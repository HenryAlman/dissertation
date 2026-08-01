

from __future__ import annotations

import warnings
from collections.abc import Collection

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy, norm
from scipy.stats.qmc import Sobol
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ribs._utils import check_batch_shape, check_finite, validate_batch
from ribs.archives import GridArchive
from ribs.emitters._emitter_base import EmitterBase
from ribs.typing import BatchData, Float, Int

# Adapted from the pyribs BayesianOptimisationEmitter for BOPElites.
# - Functionally, removed unnecessary operations (e.g. upscaling, multi-output GP -> single-output GP), 
# - redefined acquisition function to be standard Expected Improvement with jitter,
# - changed kernel to use ARD. Initially added a small added WhiteKernel for noise but have commented out for now.
# - added n_restarts_optimiser = 5 to the GP
# - warm restart for pattern search - force inject 3 best performing points so far
#       NOTE the above requires search_nrestarts > 3, and ideally set it high enough that we're still grabbing a good number of sobol-sampled points alongside the warm restart.
#       NOTE the above is also *heavily* exploitation-biased, so having a reasonably large jitter to push a bit of exploration back in should be fine.

# NOTE As noted in BOPE_emitter, we can't use lengthscale priors in sklearn, which is now the standard in BOTORCH and would be advantageous for high dimensions. See: https://arxiv.org/abs/2402.02229

# TODO: make many of the choices made in the above parameters rather than hard coded.

class BOEmitter(EmitterBase):
    def __init__(
        self,
        archive: GridArchive,
        *,
        bounds: Collection[tuple[None | Float, None | Float]] | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        search_nrestarts: Int = 5,
        num_initial_samples: Int | None = None,
        initial_solutions: ArrayLike | None = None,
        batch_size: Int = 1,
        seed: Int | None = None,
    ) -> None:
        try:
            # pylint: disable = import-outside-toplevel
            from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
            from pymoo.optimize import minimize
            from pymoo.problems.functional import FunctionalProblem
            from pymoo.termination.default import DefaultSingleObjectiveTermination
        except ImportError as e:
            raise ImportError(
                "pymoo must be installed -- please run `pip install pymoo` "
                "or `conda install pymoo`"
            ) from e
        self._pymoo_mods = {
            "PatternSearch": PatternSearch,
            "minimize": minimize,
            "FunctionalProblem": FunctionalProblem,
            "DefaultSingleObjectiveTermination": DefaultSingleObjectiveTermination,
        }

        if bounds is None and lower_bounds is None and upper_bounds is None:
            raise ValueError(
                "Bounds must be specified for BOEmitter, either "
                "with the bounds parameter or with lower_bounds and upper_bounds."
            )
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        check_finite(self.lower_bounds, "lower_bounds")
        check_finite(self.upper_bounds, "upper_bounds")

        if not isinstance(archive, GridArchive):
            raise NotImplementedError(
                f"archive type {archive.__class__.__name__} not implemented for"
                " BOEmitter. Expected GridArchive."
            )

        self._seed = seed
        self._sobol = Sobol(d=self.solution_dim, scramble=True, seed=self._seed)

        # Initializes a GP over objective function, with ARD Matern kernel.
        dimension_ranges = upper_bounds - lower_bounds
        dimensions = len(dimension_ranges)
        init_length_scales = dimension_ranges /4
        length_scale_bounds = list(zip(0.2 * dimension_ranges, 3.0 * dimension_ranges))
        self._kernel=Matern(length_scale=init_length_scales,
                      length_scale_bounds=length_scale_bounds,
                      nu=2.5
                )
        """ 
        + WhiteKernel(
                        noise_level=1e-4,
                        noise_level_bounds=(1e-6, 1e-1)
        )
        """
        self._gp = GaussianProcessRegressor(
            kernel=self._kernel, 
            alpha = 1e-10,
            normalize_y=True, 
            n_targets=1,
            n_restarts_optimizer=5
        )

        if num_initial_samples is None and initial_solutions is None:
            raise ValueError(
                "Either num_initial_samples or initial_solutions must be provided."
            )
        if num_initial_samples is not None and initial_solutions is not None:
            raise ValueError(
                "num_initial_samples and initial_solutions cannot both be provided."
            )

        if initial_solutions is not None:
            self._initial_solutions = np.asarray(
                initial_solutions, dtype=archive.dtypes["solution"]
            )
        else:
            self._initial_solutions = self._sample_n_rescale(num_initial_samples)

        check_batch_shape(
            self._initial_solutions,
            "initial_solutions",
            archive.solution_dim,
            "archive.solution_dim",
        )

        self._dataset = {
            "solution": np.empty((0, self.solution_dim), dtype=self.dtype),
            "objective": np.empty((0, 1)),
            "measures": np.empty((0, self.measure_dim)),
        }

        self._search_nrestarts = search_nrestarts

        self._batch_size = batch_size

        self._misspec = 0
        self._overspec = 0
        self._prev_numcells = len(self.archive)
        self._numitrs_noprogress = 0

    @property
    def batch_size(self) -> Int:
        """Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def num_evals(self) -> int:
        """Number of solutions stored in :attr:`_dataset`.

        This is the number of solutions that have been evaluated since the
        initialization of this emitter.
        """
        return self._dataset["solution"].shape[0]

    @property
    def measure_dim(self) -> int:
        """Number of measure functions."""
        return self.archive.measure_dim

    @property
    def num_sobol_samples(self) -> Int:
        """Number of SOBOL samples when choosing pattern search starting points in :meth:`ask`.

        .. note:: If measure function gradients are available, a potentially better way
            to do this might be to do Latin Hypercube sampling within measure space, and
            then use measure gradients to find solutions achieving those measure space
            samples. See `Kent 2024b
            <https://wrap.warwick.ac.uk/id/eprint/189556/1/WRAP_Theses_Kent_2024.pdf>`_
            Sec. 6.3 for more details.
        """
        m = 10 if self.solution_dim < 2 else 1
        return np.clip(
            m * (self.solution_dim**2) * np.prod(self.measure_dim),
            10000,
            100000,
        )

    @property
    def dtype(self) -> np.dtype:
        """Data type of solutions."""
        return self.archive.dtypes["solution"]

    @property
    def initial_solutions(self) -> np.ndarray | None:
        """Returned when the archive is empty (if :attr:`x0` is not set)."""
        return self._initial_solutions

    @EmitterBase.archive.setter
    def archive(self, new_archive: GridArchive) -> None:
        """Allows resetting the archive associated with this emitter (for archive upscaling)."""
        self._archive = new_archive


    def _sample_n_rescale(self, num_samples: int) -> np.ndarray:
        """Samples `num_samples` solutions from the SOBOL sequence.

        The solutions are also rescaled to the bounds of the search space.

        Args:
            num_samples: Number of solutions to sample.

        Returns:
            Array of shape (num_samples, :attr:`solution_dim`) containing the sampled
            solutions.
        """
        # SOBOL samples are in range [0, 1]. Need to rescale to bounds
        sobol_samples = self._sobol.random(n=num_samples)
        rescaled_samples = self.lower_bounds + sobol_samples * (
            self.upper_bounds - self.lower_bounds
        )

        return rescaled_samples
        

    def _get_ei_values(
            self, 
            samples: np.ndarray
        ) -> np.ndarray:

        mus, stds = self._gp.predict(
            samples.reshape(-1, self.solution_dim), return_std=True
        )

        # jitter parameter; higher = more exploration, less exploitation
        # https://krasserm.github.io/2018/03/21/bayesian-optimization/
        # we decrease calculated improvements by the jitter value (which scales down with range of rewards so far)
        # this means it's larger early, when we want to explore more, but becomes smaller (i.e. we become more exploitation-driven)
        # as our fitnesses improve. However, default minimum value of 0.01 means we always at least bias *slightly* to a bit of exploration.
        # which might help a little to avoid local minima.
        all_objs = self._dataset["objective"]
        best_obj = np.max(all_objs)
        worst_obj = np.min(all_objs)
        jitter = np.maximum(0.1 / (best_obj - worst_obj), 0.05)

        stds = np.maximum(stds, 1e-9) # extremely small value at least to avoid divide by zero error
        improvements = mus - best_obj - jitter
        zs = improvements / stds

        eis = improvements * norm.cdf(zs) + stds * norm.pdf(zs)
        eis = np.maximum(eis, 0.0)

        # worth noting: we warm-restart our search below with the three highest-objective points so far. 
        # This is *heavily* exploitation-biased, so having a reasonably large jitter to push a bit of exploration back in should be fine.
        return eis, mus, stds

    def ask(self) -> np.ndarray:
        
        if self.num_evals == 0:
            return np.clip(self.initial_solutions, self.lower_bounds, self.upper_bounds)
        # pymoo minimizes so need to negate
        pymoo_problem = self._pymoo_mods["FunctionalProblem"](
            n_var=self.solution_dim,
            objs=lambda x: -self._get_ei_values(x)[0],
            xl=self.lower_bounds,
            xu=self.upper_bounds,
        )

        termination = self._pymoo_mods["DefaultSingleObjectiveTermination"]()

        optimization_outcomes = {
            "optimized_samples": [],
            "optimized_eis": []
        }
        while len(optimization_outcomes["optimized_samples"]) < self.batch_size:
            samples = self._sample_n_rescale(self.num_sobol_samples)
            starting_eis, mus, stds = self._get_ei_values(samples)

            """note: AI provided this code as a diagnostic for whether above code was working:
            print(f"\n--- BO Diagnostic [Seed {self._seed}] ---")
            print(f"GP Global Mean range:   {np.min(mus):.2f} to {np.max(mus):.2f}")
            print(f"GP Global Std range:    {np.min(stds):.2f} to {np.max(stds):.2f}")
            print(f"Starting EI range:     {np.min(starting_eis):.2f} to {np.max(starting_eis):.2f}")
            """

            # force three of the search start points to be our top three performers so far
            search_starting_points = samples[
                np.argsort(starting_eis)[
                    (-self._search_nrestarts+3) :
                ]
            ]
            top3 = self._dataset["solution"][
                np.argsort(self._dataset["objective"].ravel())[-3:][::-1]
            ]
            search_starting_points = np.vstack([search_starting_points, top3])


            for x0 in search_starting_points:
                optimizer = self._pymoo_mods["PatternSearch"](x0=x0)

                # Note: Using default pymoo minimize, PatternSearch, and
                # termination.
                result = self._pymoo_mods["minimize"](
                    problem=pymoo_problem,
                    algorithm=optimizer,
                    termination=termination,
                    copy_algorithm=False,
                    seed=self._seed,
                )

                optimization_outcomes["optimized_samples"].append(result.X)
                optimization_outcomes["optimized_eis"].append(self._get_ei_values(result.X)[0].squeeze())
        
        optimized_samples = np.array(optimization_outcomes["optimized_samples"])
        optimized_eis = np.array(optimization_outcomes["optimized_eis"])

        """note: AI provided this code as a diagnostic for whether above code was working:
        print(f"Best Sobol EI:         {np.max(starting_eis):.2f}")
        print(f"Polished Peak EI:      {np.max(optimized_eis):.2f}")
        print(f"Local Search Delta:     {np.max(optimized_eis) - np.max(starting_eis):.2f}")
        """

        # identify best [batch_size] solutions
        sorted_idx = np.argsort(optimized_eis)[::-1][: self.batch_size]

        return optimized_samples[sorted_idx]

    def tell(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        add_info: BatchData,
        **fields: ArrayLike,
    ) -> np.ndarray | None:
       
        data, add_info = validate_batch(
            self.archive,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
            add_info,
        )
        # Adds new data to dataset.
        self._dataset["solution"] = np.vstack(
            (self._dataset["solution"], data["solution"])
        )
        self._dataset["objective"] = np.vstack(
            (self._dataset["objective"], data["objective"].reshape(-1, 1))
        )
        self._dataset["measures"] = np.vstack(
            (self._dataset["measures"], data["measures"])
        )

        # Updates (actually re-trains) GP with updated dataset.
        # sklearn occasionally raises LBFGS ConvergenceWarning, but this does
        # not seem to impact BOP-Elites performance too much.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._gp.fit(
                X=self._dataset["solution"],
                y=self._dataset["objective"]
            )
        return None
