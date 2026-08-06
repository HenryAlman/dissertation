import json
from pathlib import Path
from typing import Any
import psutil # TODO: tracking single-threadedness, test only
import os

from loguru import logger as log
import matplotlib.pyplot as plt

from ribs.archives import ArchiveBase, ArchiveDataFrame, GridArchive
from ribs.visualize import grid_archive_heatmap

def save_heatmap(archive: GridArchive, filename: str | Path, min_obj: int | float, max_obj: int | float) -> None:
    """Saves a heatmap of the scheduler's archive to the filename.

    Args:
        archive: Archive with results from an experiment.
        filename: Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=min_obj, vmax=max_obj, ax=ax) #TODO, what's our new max fitness?
    ax.set_ylabel("Final Y Position")
    ax.set_xlabel("Final X Position")
    fig.savefig(filename)


def save_metrics(
    outdir: Path, 
    metrics: dict[str, dict[str, list[int | float]]],
    prefix: str | None = None
) -> None:
    """Saves metrics to png plots and a JSON file.

    Args:
        outdir: output directory for saving files.
        metrics: Metrics as output by run_search.
    """
    # Plot metrics.
    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics[metric]["x"], metrics[metric]["y"])
        ax.set_title(metric)
        ax.set_xlabel("Elapsed Time")
        if (prefix is not None): 
            path = str(outdir / f"{prefix}_{metric.lower().replace(' ', '_')}.png")
        else: 
            path = str(outdir / f"{metric.lower().replace(' ', '_')}.png")
        fig.savefig(path)

    # Convert metrics to Python scalars by calling .item(), since each stats value is a
    # 0-D array by default, and JSON cannot serialize 0-D arrays.
    for metric in metrics:
        metrics[metric]["y"] = [
            m if isinstance(m, (int, float)) else m.item() for m in metrics[metric]["y"]
        ]

    # Save metrics to JSON.
    if (prefix is not None):
        with (outdir / f"{prefix}_metrics.json").open("w") as file:
            json.dump(metrics, file, indent=2)
    else:
        with (outdir / "metrics.json").open("w") as file:
            json.dump(metrics, file, indent=2)


def save_ccdf(archive: ArchiveBase, filename: str | Path) -> None:
    """Saves a CCDF showing the distribution of the archive's objectives.

    CCDF = Complementary Cumulative Distribution Function (see
    https://en.wikipedia.org/wiki/Cumulative_distribution_function#Complementary_cumulative_distribution_function_(tail_distribution)).
    The CCDF plotted here is not normalized to the range (0,1). This may help when
    comparing CCDF's among archives with different amounts of coverage (i.e. when one
    archive has more cells filled).

    Args:
        archive: Archive with results from an experiment.
        filename: Path to an image file.
    """
    fig, ax = plt.subplots()
    ax.hist(
        archive.data("objective"),
        50,  # Number of bins.
        histtype="step",
        density=False,
        cumulative=-1,  # CCDF rather than CDF.
    )
    ax.set_xlabel("Objectives")
    ax.set_ylabel("Num. Entries")
    ax.set_title("Distribution of Archive Objectives")
    fig.savefig(filename)


"""
# debug tool to test if single-threadedness is working
def assert_single_threaded():
    current_process = psutil.Process(os.getpid())
    
    thread_count = current_process.num_threads()

    if thread_count > 2: 
        print(f"Multithreading active, active OS threads: {thread_count}")
        cpu_usage = current_process.cpu_percent(interval=0.01)
        print(f"CPU usage: {cpu_usage:.1f}%")
    else:
        print(f"Single-threaded, active OS threads: {thread_count}")
"""

def check_ram_usage(msg="", cur_max_ram=0.0, from_main=False):
    process = psutil.Process(os.getpid())
    cur_ram = process.memory_info().rss / (1024 ** 3) # in GB
    if (cur_ram > cur_max_ram and from_main is False):
        print(msg, cur_ram)
        return cur_ram
    if (from_main):
        return cur_ram
    
    return cur_max_ram