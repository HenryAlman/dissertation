"""Provides the BayesianOptimizationScheduler."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ribs.archives import ArchiveBase, GridArchive
from ribs.emitters import BayesianOptimizationEmitter
from ribs.schedulers._scheduler import Scheduler

from BO_emitter import BOEmitter

# Adapted from the pyribs BayesianOptimisationScheduler for BOPElites.
# Functionally, just removed unnecessary operations (e.g. upscaling).
class BOScheduler(Scheduler):
    def __init__(
        self,
        archive: GridArchive,
        emitters: Sequence[BOEmitter],
        *,
        add_mode: Literal["batch", "single"] = "batch",
    ) -> None:
        super().__init__(archive, emitters, add_mode=add_mode)

        for i, e in enumerate(emitters):
            if not isinstance(e, BOEmitter):
                raise TypeError(
                    "All emitters must be of type BOEmitter, "
                    f"but emitter{i} has type {e.__class__.__name__}"
                )

        # Checks that ``archive`` is a GridArchive
        if not isinstance(archive, GridArchive):
            raise TypeError(
                "Archive type must be GridArchive. Actually got "
                f"{archive.__class__.__name__}"
            )

    @Scheduler.archive.setter
    def archive(self, new_archive: GridArchive) -> None:
        self._archive = new_archive

    def ask_dqd(self) -> None:
        raise NotImplementedError(
            "ask_dqd() is not supported by BOScheduler."
        )

    def tell_dqd(
        self,
        objective: ArrayLike | None,
        measures: ArrayLike,
        jacobian: ArrayLike,
        **fields: ArrayLike | None,
    ) -> None:
        raise NotImplementedError(
            "tell_dqd() is not supported by BOScheduler."
        )

    def tell(
        self,
        objective: ArrayLike | None,
        measures: ArrayLike,
        **fields: ArrayLike | None,
    ) -> None:
        """Updates :attr:`emitters` and the :attr:`archive` with new data.
        """
        if self._last_called != "ask":
            raise RuntimeError("tell() was called without calling ask().")
        self._last_called = "tell"

        data = self._validate_tell_data(
            {
                "objective": objective,
                "measures": measures,
                **fields,
            }
        )

        add_info = self._add_to_archives(data)

        pos = 0
        this_upscale_res = None
        for i, (emitter, n) in enumerate(
            zip(self._emitters, self._num_emitted, strict=True)
        ):
            end = pos + n
            emitter.tell(
                **{
                    name: None if arr is None else arr[pos:end]
                    for name, arr in data.items()
                },
                add_info={name: arr[pos:end] for name, arr in add_info.items()},
            )
            pos = end