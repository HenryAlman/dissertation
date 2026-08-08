"""Provides the BayesianOptimizationEmitter."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import math
from scipy.stats import entropy, norm
from scipy.stats.qmc import Sobol
import torch
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from ribs._utils import check_batch_shape, check_finite, validate_batch
from ribs.archives import GridArchive, ArchiveDataFrame
from ribs.emitters import EmitterBase, GaussianEmitter
from ribs.schedulers import Scheduler
from ribs.typing import BatchData, Float, Int

from diss_utils import check_ram_usage

# Adapted from the pyribs BayesianOptimisationEmitter for BOPElites.
# See notes in BO_emitter for the benefits of switching to BOTORCH.
# Additional benefit for BOP: BOTORCH maintains different hyperparameters(lengthscales etc.)
# for *each* of the outputs, whereas I believe sklearn uses the same one for them all.

# RE: acquisition function, logic remains unchanged and we still use
# pymoo Sobol+Pattern Search over GP posterior due to EJIE discontinuities 
# at cell boundaries (rather than a BOTORCH LBFGS gradient-based optimiser.)

# Additionally added functionality to use the trained GP to make predictions over a higher-res archive cell, using existing elites in main archive as warm starts.

# TODO: many values are hardcoded when they should be args; low priority for experiment but if returning to this, fix

class CustomBayesianOptimizationEmitter(EmitterBase):
    def __init__(
        self,
        archive: GridArchive,
        *,
        bounds: Collection[tuple[None | Float, None | Float]] | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        search_nrestarts: Int = 5,
        entropy_ejie: bool = False,
        upscale_schedule: ArrayLike | None = None,
        min_obj: Float = 0,
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
                "Bounds must be specified for BayesianOptimizationEmitter, either "
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
                " BayesianOptimizationEmitter. Expected GridArchive."
            )

        if (upscale_schedule is not None) and (
            not np.isclose(archive.learning_rate, 1)
        ):
            raise NotImplementedError(
                "Archive upscaling is currently incompatible with archive "
                "learning rate. Since you have specified an upscale schedule "
                f"{upscale_schedule}, the learning rate of the input archive "
                f"must be 1 (currently {archive.learning_rate})."
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

        if upscale_schedule is None:
            self._upscale_schedule = None
        else:
            self._upscale_schedule = np.asarray(upscale_schedule)
            self._check_upscale_schedule(self._upscale_schedule)

        self._batch_size = batch_size

        self._misspec = 0
        self._overspec = 0
        self._prev_numcells = len(self.archive)
        self._numitrs_noprogress = 0

        self._entropy_norm = (
            entropy(np.ones(self.archive.cells) / self.archive.cells)
            if entropy_ejie
            else None
        )

        self._min_obj = min_obj

        self._max_ram = 0

    @property
    def batch_size(self) -> Int:
        """Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def cell_prob_cutoff(self) -> Float:
        """Cutoff value (ohm) for :meth:`_get_cell_probs`.

        Described in `Kent 2024
        <https://ieeexplore.ieee.org/abstract/document/10472301>`_ Sec.IV-D.
        There are some numerical errors involved with cell_probs, so even passing the
        same sample in different shapes/contexts can sometimes return slightly different
        cell_probs, so we return cell_prob_cutoff at a lower precision than cell_probs
        to ensure the same sample consistently passes/fails the threshold check.
        """
        return round(
            0.5
            * (2 / self.archive.cells)
            ** (
                (10 * self.solution_dim)
                / (self._misspec - 2 * self._overspec + self.num_evals + 1e-6)
            )
            ** 0.5,
            4,
        )

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
    def upscale_schedule(self) -> np.ndarray | None:
        """Archive upscale schedule.

        Defined when initializing this emitter.
        """
        return self._upscale_schedule

    @property
    def upscale_trigger_threshold(self) -> Int:
        """Maximum number of iterations the emitter is allowed to not find new cells before archive upscale is triggered.

        See `here
        <https://github.com/kentwar/BOPElites/blob/main/algorithm/BOP_Elites_UKD_beta.py#L187>`_
        for more details.
        """
        return np.floor(np.sqrt(self.archive.cells))

    @property
    def min_obj(self) -> Float:
        """The lowest possible objective value.

        Refer to the documentation for this class.
        """
        return self._min_obj

    @property
    def initial_solutions(self) -> np.ndarray | None:
        """Returned when the archive is empty (if :attr:`x0` is not set)."""
        return self._initial_solutions

    @EmitterBase.archive.setter
    def archive(self, new_archive: GridArchive) -> None:
        """Allows resetting the archive associated with this emitter (for archive upscaling)."""
        self._archive = new_archive

    def post_upscale_updates(self) -> None:
        """Runs after the scheduler upscales the archive.

        This method updates :attr:`_entropy_norm` according to new number of archive
        cells and resets :attr:`_numitrs_noprogress` to 0.
        """
        if self._entropy_norm is not None:
            self._entropy_norm = entropy(
                np.ones(self.archive.cells) / self.archive.cells
            )

        self._numitrs_noprogress = 0

    def _update_no_coverage_progress(self) -> None:
        """Potentially increments :attr:`_numitrs_noprogress`.

        Increments if number of discovered archive cells remains the same for two
        successive calls to this function. Otherwise resets :attr:`_numitrs_noprogress`
        to 0.
        """
        if len(self.archive) == self._prev_numcells:
            self._numitrs_noprogress += self.batch_size
        else:
            self._numitrs_noprogress = 0
            self._prev_numcells = len(self.archive)

    def _check_upscale_schedule(self, upscale_schedule: np.ndarray) -> None:
        """Checks that ``upscale_schedule`` is a valid upscale schedule.

        Specifically:
            1. Must be a 2D array where the second dim equals :attr:`measure_dim`.
            2. The resolutions corresponding to each measure must be non-decreasing
               along axis 0.
            3. The first resolution within the schedule must equal :attr:`archive.dims`.

        Example of valid upscale_schedule:
            [
                [5, 5],
                [5, 10],
                [10, 10]
            ]

        Example of invalid upscale_schedule:
            [
                [5, 5],
                [5, 10],
                [10, 5]  <-  resolution for measure 2 decreases
            ]

        Args:
            upscale_schedule: See ``upscale_schedule`` from :meth:`__init__`.
        """
        if upscale_schedule.ndim != 2:
            raise ValueError("upscale_schedule must have 2 dimensions.")

        if upscale_schedule.shape[1] != self.measure_dim:
            raise ValueError(
                f"Expected upscale_schedule of shape (any,{self.measure_dim}), "
                f"actually got {upscale_schedule.shape}."
            )

        if not np.all(np.diff(upscale_schedule, axis=0) >= 0):
            raise ValueError(
                "The resolutions corresponding to each measure must be "
                "non-decreasing along axis 0."
            )

        if not np.all(self.archive.dims == upscale_schedule[0]):
            raise ValueError(
                "Expected the first resolution within upscale_schedule to be "
                f"{self.archive.dims} (the resolution of this emitter's "
                f"archive), actually got {upscale_schedule[0]}."
            )

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

    def _get_expected_improvements(
        self, obj_mus: np.ndarray, obj_stds: np.ndarray
    ) -> np.ndarray:
        """Computes expected improvements predicted by :attr:`_gp`.

        The improvements are calculated for a batch of solutions over all cells in the
        current archive. This function takes in the posterior means and standard
        deviations predicted by the objective gaussian process instead of the solutions
        themselves to avoid redundant computation.

        Args:
            obj_mus: Array of shape (num_solutions,) containing the posterior objective
                means predicted by the gaussian process.
            obj_stds: Array of shape (num_solutions,) containing the posterior objective
                standard deviations predicted by the gaussian process.

        Returns:
            Array of shape (num_solutions, :attr:`archive.cells`) containing the
            expected improvements for each solution over each cell.
        """
        num_samples = obj_mus.shape[0]
        all_obj = np.full((self.archive.cells,), self.min_obj)
        elite_idx, elite_obj = self.archive.data(
            ["index", "objective"], return_type="tuple"
        )
        all_obj[elite_idx] = elite_obj

        distribution = norm(
            loc=np.repeat(all_obj[None, :], num_samples, axis=0),
            scale=np.repeat(obj_stds[:, None], self.archive.cells, axis=1),
        )

        return (obj_mus[:, None] - all_obj) * distribution.cdf(
            obj_mus[:, None]
        ) + obj_stds[:, None] * distribution.pdf(obj_mus[:, None])

    def _get_cell_probs(
        self,
        meas_mus: np.ndarray,
        meas_stds: np.ndarray,
        normalize: bool = True,
        cutoff: bool = True,
    ) -> np.ndarray:
        """Computes archive cell membership probabilities predicted by :attr:`_gp`.

        Probabilities are computed for a batch of solutions. This function takes in the
        posterior means and standard deviations predicted by the measure gaussian
        processes instead of the solutions themselves to avoid redundant computation.

        Args:
            meas_mus: Array of shape (num_solutions, :attr:`measure_dim`) containing the
                posterior measure means predicted by the gaussian process.
            meas_stds: Array of shape (num_solutions, :attr:`measure_dim`) containing
                the posterior measure standard deviations predicted by the gaussian
                process.
            normalize: If ``True``, normalizes the cell probabilities such that they sum
                to 1 for each solution.
            cutoff: If ``True``, sets cell probabilities below :attr:`cell_prob_cutoff`
                to 0.

        Returns:
            Array of shape (num_solutions, :attr:`archive.cells`) containing the
            predicted cell probabilities for each solution.
        """
        num_solutions = meas_mus.shape[0]

        cell_probs = np.ones((num_solutions, *self.archive.dims))
        for measure_idx, (mus, stds) in enumerate(
            zip(meas_mus.T, meas_stds.T, strict=True)
        ):
            distribution = norm(loc=mus, scale=stds)

            # computes the cdf values at each cell boundary, this has shape
            # (num_solutions, num_boundaries).
            cdf_vals = distribution.cdf(self.archive.boundaries[measure_idx][:, None]).T

            # takes the difference between each pair of adjacent boundaries,
            # this has shape (num_solutions, num_boundaries-1) = (num_solutions,
            # measure_resolution)
            cdf_diffs = np.diff(cdf_vals, axis=1)

            # reshapes diffs to be compatible with element-wise multiplication
            for i in range(self.measure_dim):
                if i != measure_idx:
                    # axis i+1 because first axis is num_solutions
                    cdf_diffs = np.expand_dims(cdf_diffs, axis=i + 1)

            cell_probs *= cdf_diffs

        cell_probs = cell_probs.reshape((num_solutions, self.archive.cells))

        if cutoff:
            cell_probs[cell_probs < self.cell_prob_cutoff] = 0

        if normalize:
            # with ``cutoff``, it is possible a solution has 0 prob on all
            # cells, we don't normalize on those to prevent numerical error
            cell_probs_sum = np.sum(cell_probs, axis=1)[:, None]
            cell_probs_sum[cell_probs_sum == 0] = 1
            cell_probs /= cell_probs_sum

        return cell_probs

    def _get_ejie_values(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes *Expected Joint Improvement of Elites* (EJIE) acquisition values.

        Value are computed by multiplying the predicted expected improvements and cell
        membership probabilities. Returns individual EJIE values for each cell in an
        array of shape (num_solutions, :attr:`archive.cells`). You can use
        `np.sum(result, axis=1)` to get the total EJIE on the entire archive. Also
        returns the predicted cell membership probabilities for each sample in an array
        of shape (num_solutions, :attr:`archive.cells`).

        Args:
            samples: Array of shape (num_samples, :attr:`solution_dim`) containing
                samples whose EJIE values need to be computed.

        Returns:
            Returns an array of shape (num_solutions, :attr:`archive.cells`) containing
            each solution's EJIE values for each cell. Also returns an array of shape
            (num_solutions, :attr:`archive.cells`) containing the predicted cell
            membership probabilities for each solution.
        """
        # convert samples to normalised, torch
        self._max_ram = check_ram_usage("_get_ejie_values start RAM:", self._max_ram)
        samples_norm = self._normalise(samples)
        torch_samples = torch.tensor(samples_norm.reshape(-1, self.solution_dim), dtype=torch.float64, device="cpu")
        self._max_ram = check_ram_usage("_get_ejie_values torch_samples RAM:", self._max_ram)
        # grab means and stddevs from normalised torch GP
        # we chunk the list, as BOTORCH loads them all into memory simultaneously otherwise!
        mus_torch_list = []
        stds_torch_list = []
        chunk_size = 2000
        self._max_ram = check_ram_usage("_get_ejie_values pre-posterior RAM:", self._max_ram)
        with torch.no_grad():
            for i in range(0, torch_samples.size(0), chunk_size):
                self._max_ram = check_ram_usage(f"_get_ejie_values chunk {i} RAM:", self._max_ram)
                chunk = torch_samples[i : i+chunk_size]
                posterior = self._gp.posterior(chunk)
                mus_torch_list.append(posterior.mean.clone())
                stds_torch_list.append(posterior.stddev.clone())
            mus_torch = torch.cat(mus_torch_list, dim=0)
            stds_torch = torch.cat(stds_torch_list, dim=0)

        # GP values were standardised, so need to unstandardise/unnormalise
        self._max_ram = check_ram_usage("_get_ejie_values ops RAM:", self._max_ram)
        mus_unnorm = torch.empty_like(mus_torch)
        stds_unnorm = torch.empty_like(stds_torch)
        mus_unnorm[:, 0] = mus_torch[:, 0] * self._obj_std + self._obj_mean
        stds_unnorm[:, 0] = stds_torch[:, 0] * self._obj_std
        mus_unnorm[:, 1] = mus_torch[:, 1] * self._mea0_std + self._mea0_mean
        stds_unnorm[:, 1] = stds_torch[:, 1] * self._mea0_std
        self._max_ram = check_ram_usage("_get_ejie_values ops2 RAM:", self._max_ram)
        mus_unnorm[:, 2] = mus_torch[:, 2] * self._mea1_std + self._mea1_mean
        stds_unnorm[:, 2] = stds_torch[:, 2] * self._mea1_std

        self._max_ram = check_ram_usage("_get_ejie_values backtonorm RAM:", self._max_ram)
        

        # make numpy, then pass to identical logic to the prior original sklearn version
        mus = mus_unnorm.numpy().copy()
        stds = stds_unnorm.numpy().copy()

        del mus_torch, stds_torch, mus_unnorm, stds_unnorm # del for safety

        expected_improvements = self._get_expected_improvements(mus[:, 0], stds[:, 0])

        cell_probs = self._get_cell_probs(
            mus[:, 1:], stds[:, 1:], normalize=True, cutoff=True
        )

        if self._entropy_norm is not None:
            all_zero_filter = np.all(np.isclose(cell_probs, 0), axis=1)
            entropies = np.zeros((mus.shape[0], 1))
            entropies[~all_zero_filter] = entropy(cell_probs[~all_zero_filter], axis=1)[
                :, None
            ]
            ejie_by_cell = (
                expected_improvements
                * cell_probs
                * (1 + entropies / self._entropy_norm)
            )
        else:
            ejie_by_cell = expected_improvements * cell_probs

        self._max_ram = check_ram_usage("_get_ejie_values returning RAM:", self._max_ram)

        return ejie_by_cell, cell_probs

    def ask(self) -> np.ndarray:
        if self.num_evals == 0:
            return np.clip(self.initial_solutions, self.lower_bounds, self.upper_bounds)

        # pymoo minimizes so need to negate
        pymoo_problem = self._pymoo_mods["FunctionalProblem"](
            n_var=self.solution_dim,
            objs=lambda x: -np.sum(self._get_ejie_values(x)[0], axis=1), # sum of per-cell EJIE value for each sample, i.e. total EJIE
            xl=self.lower_bounds,
            xu=self.upper_bounds,
        )

        termination = self._pymoo_mods["DefaultSingleObjectiveTermination"]()

        optimization_outcomes = {
            "optimized_samples": [],
            "optimized_ejie_by_cell": [],
            "optimized_cell_probs": [],
        }
        while len(optimization_outcomes["optimized_samples"]) < self.batch_size:
            samples = self._sample_n_rescale(self.num_sobol_samples)
            starting_ejie_by_cell, _ = self._get_ejie_values(samples)

            search_starting_points = samples[
                np.argsort(np.sum(starting_ejie_by_cell, axis=1))[
                    -self._search_nrestarts :
                ]
            ]

            # optimizes ejie values of starting points
            found_positive_ejie = False
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

                if -result.F > 0:
                    optimization_outcomes["optimized_samples"].append(result.X)
                    # retrieve the cell-wise EJIE and probs for optimized
                    # solution
                    opt_ejie_by_cell, opt_cell_probs = self._get_ejie_values(result.X)
                    optimization_outcomes["optimized_ejie_by_cell"].append(
                        opt_ejie_by_cell.squeeze()
                    )
                    optimization_outcomes["optimized_cell_probs"].append(
                        opt_cell_probs.squeeze()
                    )
                    found_positive_ejie = True

            # if didn't find any positive ejie after optimization, increments
            # over-specification count
            # (we don't increment the over-specification count if we found
            # some positive EJIEs but not enough to fill the batch)
            if not found_positive_ejie:
                self._overspec += 1
        optimized_samples = np.array(optimization_outcomes["optimized_samples"])
        ejie_by_cell = np.array(optimization_outcomes["optimized_ejie_by_cell"])
        cell_probs = np.array(optimization_outcomes["optimized_cell_probs"])

        total_ejies = np.sum(ejie_by_cell, axis=1)
        # Most likely cell for each optimized solution
        best_cell_idx = np.argmax(cell_probs, axis=1)
        best_cell_probs = cell_probs[range(cell_probs.shape[0]), best_cell_idx]

        # Computes EJIE attributions of the most likely cell for each solution
        ejie_attributions = (
            ejie_by_cell[range(ejie_by_cell.shape[0]), best_cell_idx] / total_ejies
        )

        # Sort by EJIE, take the top :attr:`batch_size` samples
        sorted_idx = np.argsort(total_ejies)[::-1][: self.batch_size]

        # NOTE: BOP-Elites Algorithm 1 implements a different mis-specification
        # check, in which a mis-specification occurs if a sample is predicted
        # to be in a cell with high confidence, but the prediction turns out
        # to be wrong.
        # We implement a new mis-specification check as recommended by the
        # author. New mis-specification checks whether most of a sample's EJIE
        # is attributed to a single cell, which has low predicted cell
        # probability. This corresponds to the (undesirable) scenario in which
        # a cell that is likely unreachable dominates EJIE.
        for best_prob, attr_val in zip(
            best_cell_probs[sorted_idx], ejie_attributions[sorted_idx], strict=True
        ):
            if best_prob < 0.5 < attr_val:
                self._misspec += 1
        return optimized_samples[sorted_idx]

    def tell(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        add_info: BatchData,
        **fields: ArrayLike,
    ) -> np.ndarray | None:
        """Updates the gaussian process and potentially upscales the archive.

        The function does the following:

        1. Adds ``solution``, ``objective``, and ``measures`` to :attr:`_dataset`.
        2. Updates :attr:`_gp` with :attr:`_dataset`.
        3. For each solution whose EJIE attribution exceeds 50%, checks whether its
           predicted cell is different from the cell it is actually assigned according
           to its evaluated measures. If so, increments :attr:`_misspec`.
        4. If :attr:`upscale_schedule` is not ``None``, and if the archive upscale
           conditions have been met, sends an upscale signal upstream by returning the
           next resolution to upscale to.

        Args:
            solution: (batch_size, :attr:`solution_dim`) array of solutions generated by
                this emitter's :meth:`ask()` method.
            objective: 1D array containing the objective function value of each
                solution.
            measures: (batch_size, :attr:`measure_dim`) array with the measure values of
                each solution.
            add_info: Data returned from the archive
                :meth:`~ribs.archives.ArchiveBase.add` method.
            fields: Additional data for each solution. Each argument should be an array
                with batch_size as the first dimension.

        Returns:
            A 1D array of shape (:attr:`measure_dim`,) containing the
            next resolution to upscale to. The actual upscaling will be done in the
            scheduler, through
            :meth:`~ribs.schedulers.BayesianOptimizationScheduler.tell`. If no upscaling
            is needed in the current step, returns ``None``.
        """
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

        self._max_ram = check_ram_usage("tell start RAM:", self._max_ram)

        # per BOTORCH best practice: (see https://botorch.readthedocs.io/en/stable/models.html#botorch.models.gp_regression.SingleTaskGP)
        # normalise solutions to range 0-1
        normalised_solutions = self._normalise(self._dataset["solution"])
        X_train = torch.tensor(normalised_solutions, dtype=torch.float64, device="cpu")
        # standardise y
        Y_train_obj = torch.tensor(self._dataset["objective"], dtype=torch.float64, device="cpu")
        standardised_Y_train_obj = (Y_train_obj - Y_train_obj.mean()) / (Y_train_obj.std() + 1e-8)
        self._max_ram = check_ram_usage("tell ops1 RAM:", self._max_ram)
        Y_train_mea_0 = torch.tensor(self._dataset["measures"][:,0], dtype=torch.float64, device="cpu")
        standardised_Y_train_mea_0 = (Y_train_mea_0 - Y_train_mea_0.mean()) / (Y_train_mea_0.std() + 1e-8)
        Y_train_mea_1 = torch.tensor(self._dataset["measures"][:,1], dtype=torch.float64, device="cpu")
        standardised_Y_train_mea_1 = (Y_train_mea_1 - Y_train_mea_1.mean()) / (Y_train_mea_1.std() + 1e-8)
        self._max_ram = check_ram_usage("tell ops2 RAM:", self._max_ram)
        

        # store relevant values for unnormalising later when needed
        self._best_standardised_fitness=standardised_Y_train_obj.max()
        self._obj_mean = Y_train_obj.mean()
        self._obj_std = Y_train_obj.std() + 1e-8 # (ensure no div by 0 error by adding tiny offset)
        self._mea0_mean = Y_train_mea_0.mean()
        self._mea0_std = Y_train_mea_0.std() + 1e-8
        self._mea1_mean = Y_train_mea_1.mean()
        self._mea1_std = Y_train_mea_1.std() + 1e-8

        self._max_ram = check_ram_usage("tell ops3 RAM:", self._max_ram)

        # create list like: [obj_0, mea0_0, mea1_0], [obj1, mea0_1, mea1-1], ...
        standardised_Y = torch.stack([standardised_Y_train_obj.squeeze(-1), standardised_Y_train_mea_0, standardised_Y_train_mea_1], dim=-1)
        self._max_ram = check_ram_usage("tell ops4 RAM:", self._max_ram)

        # note: we use SingleTaskGP with separate independent outputs rather than MultiTaskGP
        # (i.e. conditionally-dependent outputs). This is because:
        # A) efficiency/scaling O(D * N^3) vs. O((D*N)^3), 
        # B) it matches the sklearn approach used originally,
        # and C) we aren't looking to predict fitness based on behavioural descriptors so dont need
        # a shared/joint probability distribution between them.
        # a ModelListGP may be more appropriate for a *design* problem where we want to be able to
        # predict performance based on descriptors later, e.g. designing robot morphologies.

        # also note: we instantiate a new GP (rather than condition_on_observation to add data and get a "fantasy model")
        # see https://github.com/meta-pytorch/botorch/issues/533, https://github.com/meta-pytorch/botorch/discussions/2025 
        
        # use BOTORCH defaults, but with Matern 5/2 and a forcibly constrained low noise level to prevent covariance collapse in deterministic environment,
        # We want to keep defaults as they implemented findings from Vanilla Bayesian Optimization Performs Great in High Dimensions
        # and they infer homoscedastic noise as a default as well

        # this returns a Matern 5/2 with all of those default settings applied, rather than make one from scratch
        matern = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=self.solution_dim,
            use_rbf_kernel=False,
            batch_shape=torch.Size([standardised_Y.shape[-1]]) # use batch 3 here because fitting 3 GPs, over obj, mea0, mea1
        )
        # force a custom likelihood with a very small amount of noise constraint
        # because sim is deterministic, it *could* reduce noise to 0 which can cause crashes
        likelihood = GaussianLikelihood(
            noise_constraint=Interval(1e-8, 1e-6),
            batch_shape=torch.Size([standardised_Y.shape[-1]]) # use batch 3 here because fitting 3 GPs, over obj, mea0, mea1
        )

        self._gp = SingleTaskGP(
            train_X=X_train,
            train_Y=standardised_Y,
            likelihood=likelihood,
            covar_module=matern
        )
        self._max_ram = check_ram_usage("tell postGP RAM:", self._max_ram)
         # optimise GP parameters and fit it
        mll = ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)
        #default is 5 from documentation. However, it stops as soon as it works unless you set pick_best_of_all_attempts=True
        #so we can have more restarts without additional cost. Given the lengthiness of these experiments, having it fail halfway through is not worth it.
        fit_gpytorch_mll(mll, max_attempts=10)
        self._max_ram = check_ram_usage("tell postMLL RAM:", self._max_ram)
        # put GP into evaluation mode, not training mode
        self._gp.eval()
        self._gp.likelihood.eval()

        
        # Checks upscale conditions and upscales if needed
        # NOTE: BOP-Elites Algorithm 1 implements a slightly different upscale
        # condition, in which the archive upscale is triggered if either all
        # its cells have been filled or if num_evals > 2*cells. However, the
        # old condition may struggle with applications where some cells are not
        # feasible. We implement an improved condition here as recommended by
        # the original author. The new condition triggers the upscale when no
        # new cell has been found for multiple iterations.
        self._update_no_coverage_progress()
        if (
            (self.upscale_schedule is not None)
            and np.any(np.all(self.upscale_schedule > self.archive.dims, axis=1))
            and self._numitrs_noprogress > self.upscale_trigger_threshold
        ):
            # The next resolution on the schedule that is higher than the
            # current resolution along all measure dims
            next_res = self.upscale_schedule[
                np.all(self.upscale_schedule > self.archive.dims, axis=1)
            ][0]
            return next_res
        return None


    def _normalise(self, x):
        return (x - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
    
    def _unnormalise(self, x_normalised):
        torch_lower = torch.tensor(self.lower_bounds, dtype=torch.float64, device=("cpu"))
        torch_upper = torch.tensor(self.upper_bounds, dtype=torch.float64, device=("cpu"))
        return torch_lower + (torch_upper - torch_lower) * x_normalised


    # run MAPElites over
    def get_predicted_elites(
        self,
        result_archive: GridArchive, # pass in the result_archive here!
        outdir: Path,
        num_iterations: int = 5000
    ) -> list[ArrayLike]:

        predicted_elites = []
        
        dim0_boundaries = result_archive.boundaries[0]
        dim1_boundaries = result_archive.boundaries[1]

        self.archive.retessellate(result_archive.dims) # have to retesselate so get_cell_probs works correctly

        # setup MAP Elites
        prediction_archive = GridArchive(
                solution_dim=self.solution_dim,
                dims=result_archive.dims,
                ranges=[(dim0_boundaries[0], dim0_boundaries[-1]), (dim1_boundaries[0], dim1_boundaries[-1])],
            )
        prediction_emitter = GaussianEmitter(
                        archive=prediction_archive, # shared reference to the archive!
                        x0=np.zeros(self.solution_dim), # point from which to begin sampling in MAP-Elites
                        sigma=(np.asarray([self.upper_bounds]) - np.asarray([self.lower_bounds])) * 0.1,
                        lower_bounds=self.lower_bounds,
                        upper_bounds=self.upper_bounds,
                        batch_size=50, # number of samples to take for simulation at each iteration
                        seed=None,
                    )
        scheduler = Scheduler(prediction_archive, [prediction_emitter])
        iteration = 0

        # replaces our "simulation" step; we take the GP prediction as "ground truth" to build a prediction map
        # however, BOP additionally added the step of multiplying the predicted mean by the probability that the solution is in the predicted cell!
        def evaluate_gp(x):
            nonlocal dim0_boundaries, dim1_boundaries, prediction_archive

            x_norm = self._normalise(x)
            torch_x = torch.tensor(x_norm.reshape(-1, self.solution_dim), dtype=torch.float64, device="cpu")

            self._gp.eval()
            with torch.no_grad():
                # "ground truth" is straight from GP
                posterior = self._gp.posterior(torch_x)
                mu_torch = posterior.mean.clone()
                std_torch = posterior.stddev.clone()

            mu_unnorm = torch.empty_like(mu_torch)
            std_unnorm = torch.empty_like(std_torch)
            mu_unnorm[:, 0] = mu_torch[:, 0] * self._obj_std + self._obj_mean
            std_unnorm[:, 0] = std_torch[:, 0] * self._obj_std
            mu_unnorm[:, 1] = mu_torch[:, 1] * self._mea0_std + self._mea0_mean
            std_unnorm[:, 1] = std_torch[:, 1] * self._mea0_std
            mu_unnorm[:, 2] = mu_torch[:, 2] * self._mea1_std + self._mea1_mean
            std_unnorm[:, 2] = std_torch[:, 2] * self._mea1_std

            mu = mu_unnorm.numpy().copy()
            std = std_unnorm.numpy().copy()
    
            del mu_torch, std_torch, mu_unnorm, std_unnorm # del for safety

            # get probability of sample being in predicted cell
            cell_probs = self._get_cell_probs(
                        mu[:, 1:], std[:, 1:], normalize=True, cutoff=True
                    )

            cell_idx = prediction_archive.index_of(mu[:, 1:])

            return mu[:, 0] * cell_probs[:, cell_idx], mu[:, 1], mu[:, 2]


        # run MAPElites over GP
        while (iteration < num_iterations):
            sols = scheduler.ask()
            objs, meas = [], []
            predictions = [evaluate_gp(model) for model in sols]
            for pred_obj, pred_mea0, pred_mea1 in predictions:
                objs.append(pred_obj.item())
                meas.append([pred_mea0.item(), pred_mea1.item()])
            scheduler.tell(objs, meas)

            if (iteration % 50 == 0):
                print(f"Iteration {iteration} / {num_iterations}")

            iteration += 1

        archive_as_dataframe = ArchiveDataFrame(prediction_archive.data(return_type="pandas"))
        predicted_elites = np.array([elite["solution"] for elite in archive_as_dataframe.iterelites()])

        save_loc = str(outdir / "predicted_elites.npy")
        np.save(save_loc, predicted_elites)
        return predicted_elites # return to actually be simulated and combined with result_archive