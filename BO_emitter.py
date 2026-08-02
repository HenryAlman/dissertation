from __future__ import annotations

import warnings
from collections.abc import Collection

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import entropy, norm
from scipy.stats.qmc import Sobol
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf


from ribs._utils import check_batch_shape, check_finite, validate_batch
from ribs.archives import GridArchive
from ribs.emitters._emitter_base import EmitterBase
from ribs.typing import BatchData, Float, Int

# Adapted from the pyribs BayesianOptimisationEmitter for BOPElites.
# - Functionally, removed unnecessary operations (e.g. upscaling, multi-output GP -> single-output GP), 
# - and switched to BOTORCH. BOTORCH:
# - naturally uses a dimensionality-scaled prior on the kernel, see Vanilla Bayesian Optimization Performs Great in High Dimensions
# - naturally incorporates ARD with the above
# - naturally infers at least homoscedastic noise
# - comes with built in support for using qLogExpectedImprovement instead of standard EI, which helps with vanishing gradients,

# TODO:  potential improvements: using HeteroskedasticSingleTaskGP and qLogNoisyExpectedImprovement to handle the variance of maze sims
# i.e. that a small parameter change can cause a collision, drastically changing fitness etc. Would need testing though.

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
        

    def ask(self) -> np.ndarray:
        # first run only: grab sobol points
        if self.num_evals == 0:
            return np.clip(self.initial_solutions, self.lower_bounds, self.upper_bounds)

        # define acquisition function - in this case we use qLogEI
        # a recommended improvement over EI which helps avoid vanishing gradient issues
        # note we use "q" version to support batching!
        # potential future improvement as noted at top would be incorporating qLogNoiseExpectedImprovement
        # to handle noisy domains
        acq_func = qLogExpectedImprovement(
            model=self._gp,
            best_f=self._best_standardised_fitness
        )

        candidates, acq_value = optimize_acqf(
            acq_function=acq_func,
            # we normalised the X and Y data going into the GP, so we search from 0 to 1
            bounds=torch.stack([torch.zeros(self._solution_dim), torch.ones(self._solution_dim)]).to(device="cpu", dtype=torch.float64),
            q=self._batch_size,
            num_restarts=self._search_nrestarts,
            raw_samples=self.num_sobol_samples
        )

        unnormalised_candidates = []
        for candidate in candidates:
            # and now need to unnormalise them before returning
            unnormalised_candidates.append(self._unnormalise(candidate).numpy())
        
        return np.array(unnormalised_candidates)

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

        # per BOTORCH best practice: (see https://botorch.readthedocs.io/en/stable/models.html#botorch.models.gp_regression.SingleTaskGP)
        # normalise solutions to range 0-1
        normalised_solutions = self._normalise(self._dataset["solution"])
        X_train = torch.tensor(normalised_solutions, dtype=torch.float64, device="cpu")
        Y_train = torch.tensor(self._dataset["objective"], dtype=torch.float64, device="cpu")
        # standardise y
        standardised_Y = (Y_train - Y_train.mean()) / (Y_train.std() + 1e-8)
        self._best_standardised_fitness=standardised_Y.max()

        # use BOTORCH default, no explicit kernel/lengthscale,
        # as they implemented findings from Vanilla Bayesian Optimization Performs Great in High Dimensions
        # and they infer homoscedastic noise as a default as well
        self._gp = SingleTaskGP(
            train_X=X_train,
            train_Y=standardised_Y
        )

        # optimise GP parameters and fit it
        mll = ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)
        fit_gpytorch_mll(mll, max_attempts=10) #TODO: make an arg

        return None


    # utils for converting between GP normalised data, 
    # and unnormalised data we need elsewhere (e.g. to return in ask())
    def _normalise(self, x):
        return (x - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def _unnormalise(self, x_normalised):
        torch_lower = torch.tensor(self.lower_bounds, dtype=torch.float64, device=("cpu"))
        torch_upper = torch.tensor(self.upper_bounds, dtype=torch.float64, device=("cpu"))
        return torch_lower + (torch_upper - torch_lower) * x_normalised
