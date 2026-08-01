from __future__ import annotations

# force single-threaded for fair performance across trials
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# if issues with graphics drivers with gladGL error, try the below:
"""
os.environ["MUJOCO_GL"] = "egl"
"""

# util imports
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import math
from typing import Any
import pickle
import imageio

# libs
import fire # cmd line parameter/arguments for main
#import psutil # tracking single-threadedness, test only
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd # used for saving/reading archive CSVs and dataframes
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
from loguru import logger as log

# pyribs and customised versions
from ribs.archives import ArchiveBase, ArchiveDataFrame, GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler
from BOPE_emitter import CustomBayesianOptimizationEmitter # import from custom file, not from pyribs. We made a couple small changes.
from BOPE_scheduler import CustomBayesianOptimizationScheduler # same as the default, just changed to expect CustomBayesianOptimizationEmitter not pyribs one
from BO_emitter import BOEmitter # custom BO-only emitter and scheduler
from BO_scheduler import BOScheduler
from ribs.visualize import grid_archive_heatmap

from gymnasium_robotics.envs.maze.maps import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE

# custom environment wrapper over AntMaze
from mujoco_env_wrapper import MujocoEnvWrapper


def simulate(
    model: np.ndarray, # The array of weights/biases for the controller policy.
    maze_params: dict[str, Any], # dict for maze parameters; dictated by maze input
    xml_file: str, # path to XML to use
    seed: int | None = None, # environment seed
    record_video: bool = False, # If passed in, this will be used instead of creating a new env; mainly for recording video, see run_evaluation().
    record_video_idx: int | None = None,
    record_outdir: Path | None = None
) -> tuple[float, float, float, bool]: # total_reward (objective), avg linear velocity and avg z-axis rotation (measures), and whether the goal was reached.

    maze = maze_params["maze"]
    max_episode_steps = maze_params["max_episode_steps"]
    maze_options = maze_params["maze_options"]
    maze_max_dist = maze_params["maze_max_dist"]

    # if running search:
    if not record_video:
        base_env = gym.make("AntMaze_UMazeDense-v5", maze_map=maze, continuing_task=False, xml_file=xml_file, max_episode_steps=max_episode_steps)
    else:
        base_env = gym.make("AntMaze_UMazeDense-v5", maze_map=maze, continuing_task=False, xml_file=xml_file, max_episode_steps=max_episode_steps, render_mode="rgb_array")
    base_env.unwrapped.include_sensors = True
    env = MujocoEnvWrapper(base_env, max_episode_steps)
    

    total_reward = 0.0
    linear_x_velocities = []
    linear_y_velocities = []
    angular_z_axis_velocities = []
    reached_goal = False

    if (record_video):
        frames=[]

    obs, _ = env.reset(seed=seed, options=maze_options) # reset has to be called first per Gym/Mujoco Documentation!
    done = False # track whether session was terminated or truncated
    
    # initialise
    if (xml_file == "/users/40795510/sharedscratch/dissertation/new_racecar.xml"):
        # uses simple/direct policy
        w_end = 8
        weights = model[:w_end].reshape(4, 2)
        biases = model[w_end:]
    else:
        # Ant/Hexapod uses CPGRBF network with 10 RBFs, equally spaced around unit circle
        # each RBF drives hips and ankles uniformly - indirect encoding, see https://mathiasthor.github.io/assets/pdf/generic_neural_locomotion_control_framework.pdf
        # we hardcode phases into the legs (i.e. some legs are at state cpg_state, some are at cpg_state * rotation matrix i.e. opposite side of unit circle)
        # this produces symmetric movement. For regularised robots (as our Ant and Hexapod are), their research showed this can be highly effective

        # unpack weights
        w_cpg = model[0]
        w_locomotion = model[1:-4]
        w_locomotion = w_locomotion.reshape(10, 2)
        w_hip_centresensor, w_ankle_centresensor, w_hip_sensordiff, w_ankle_sensordiff = model[-4:]

        # initialise CPG start state, RBFs placed equilaterally around circle, phase gaits, and torque sign masks
        cpg_state = np.array([0.1, 0.0]) 
        # rbfs equally spaced around unit circle
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)
        rbf_centers = np.column_stack((xs, ys))
        # 75% of the distance between each point
        # ensures a smooth blending of activations as we progress around unit circle
        rbf_sigma = ((math.sqrt(5)-1)/2) * (0.75)

        # virtual sign mask - torque control/motion is based on leg orientation in XML,
        # this acts as a "translator" to the correct signs
        # so the model outputs "conceptual" actions (like "left hip forward") and this handles the signs
        sign_mask = np.array([
            1.0, # frontright hip, +ve torque is forward
            1.0, # frontright ankle, +ve torque is down
            -1.0, # leftright hip, -ve torque is forward
            1.0, # leftright ankle, +ve torque is down
            1.0, # backright hip, +ve torque is forward
            -1.0, # backright ankle, -ve torque is down
            -1.0, # backleft hip, -ve torque is forward
            -1.0 # backleft hip, -ve torque is down
        ])

        # sensor reflex map in virtual/conceptual space; we inject weighted sensor data directly into the torques
        # see: https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2021.743888/full
        # we use a heavily simplified version - rather than shunting inhibition nodes,
        # we just add/subtract weighted sensor output from the torques.
        sensor_reflex_map = np.array([
            # frontright hip gets hip signal and inverts sign of sensordiff 
            # (as sensor diff is calculated as left - right, with closer being higher, so if positive then left wall is closer
            # and we want to turn right, i.e. dampen right motion and exaggerate left motion).
            # same logic for the others.
            [w_hip_centresensor, -w_hip_sensordiff], # frontright hip
            [w_ankle_centresensor, -w_ankle_sensordiff], # frontright ankle
            [w_hip_centresensor, w_hip_sensordiff], # frontleft hip
            [w_ankle_centresensor, w_ankle_sensordiff], # frontleft ankle
            [w_hip_centresensor, -w_hip_sensordiff], # backright hip
            [w_ankle_centresensor, -w_ankle_sensordiff], # backright ankle
            [w_hip_centresensor, w_hip_sensordiff], # backleft hip
            [w_ankle_centresensor, w_ankle_sensordiff], # backleft ankle
        ])

        # ant gait phases (we overwrite below for hexapod). Pair diagonally opposing.
        gait_phases = np.array([
                    0.0, # frontright
                    np.pi, # frontleft
                    np.pi, # backright
                    0.0 # backleft
                ])

        # adding middle right hip/ank, middle left hip/ank
        if (xml_file == "/users/40795510/sharedscratch/dissertation/rangefinder_hex.xml"):
            # torque directions work exactly the same as back legs
            sign_mask = np.append(sign_mask, [
                1.0, # midright hip
                -1.0,  # midright ank
                -1.0,  # midleft hip
                -1.0 # midleft ank
            ])
            # extend sensor signals to middle legs as well
            sensor_reflex_map = np.append(sensor_reflex_map, [
                [w_hip_centresensor, -w_hip_sensordiff], # middleright hip
                [w_ankle_centresensor, -w_ankle_sensordiff], # middleright ankle
                [w_hip_centresensor, w_hip_sensordiff], # middleleft hip
                [w_ankle_centresensor, w_ankle_sensordiff], # middleleft ankle
            ], axis=0)

            # different gait for hex - front/back legs on one side paired with mid on other side
            gait_phases = np.array([
                0.0, # frontright
                np.pi, # frontleft
                0.0, # backright
                np.pi, #backleft
                np.pi, # midright
                0.0 # midleft
            ])

        # cpg_state rotation matrices for each leg, depending on gait_phases
        # a gait phase of pi means that leg is being driven by the RBF on the *opposite side* than the RBF driving a leg with phase 0.0
        rotation_matrices = np.array([
            [[np.cos(phi), -np.sin(phi)], 
            [np.sin(phi),  np.cos(phi)]] 
            for phi in gait_phases
        ])

    # run sim
    while not done:
        if (record_video):
            ifframe = env.render()
            frames.append(ifframe)

        mujoco_data = env.unwrapped.data

        # get sensor data
        rangefinders_data = mujoco_data.sensordata[:3].copy()
        # rangefinders return -1 return if hit nothing; we want this to engender behaviour like a far away wall, not a close wall,
        # so for any -1 returns we instead set to the maze_max_dist
        rangefinders_data[rangefinders_data == -1.0] = maze_max_dist
        # we then shift so close walls send strong signal (1) while far walls send a weak signal (0)
        rangefinders = 1.0 - np.clip(rangefinders_data / maze_max_dist, 0.0, 1.0)

        # CAR:
        if (xml_file == "/users/40795510/sharedscratch/dissertation/new_racecar.xml"):
            velocimeter_x = mujoco_data.sensordata[3].copy()
            velocimeter_y = mujoco_data.sensordata[4].copy()
            xy_speed = math.sqrt(velocimeter_x**2 + velocimeter_y**2)
            # direct policy between obs and actions based on pure model weights (range -1 to 1)
            pruned_obs = [rangefinders[0], rangefinders[1], rangefinders[2], xy_speed]
            unique_actions = np.dot(pruned_obs, weights) + biases
            unique_actions = np.tanh(unique_actions)
            # multiply output by car XML control range to get actual actions
            unique_actions[0] = unique_actions[0] * 30
            unique_actions[1] = unique_actions[1] * 0.785
        # ANT:
        else:
            # SO(2) oscillator for discrete time steps, with weight w_cpg taking place
            # of phi parameter. Original paper set it to exactly .01*pi, here we learn it.
            x, y = cpg_state
            next_x = x * np.cos(w_cpg) - y * np.sin(w_cpg)
            next_y = x * np.sin(w_cpg) + y * np.cos(w_cpg)
            # note: original paper multiplies above by 1.01 and takes tanh, rather than dividing by norm.
            # The *purpose* is to keep things in a stable loop, which dividing by the norm does just as well
            # by holding it to the unit circle (around which our rbf centers are evenly spaced).
            # the alpha/tanh formulation from the paper allows for regaining cpg stability if it is disrupted
            # by injected e.g. sensor data, but we inject our sensor data directly to the torques for simplicity
            # so it isn't needed here.
            # note: 1e-8 there as safety buffer to avoid div by zero error if state is exactly 0
            cpg_state = np.array([next_x, next_y]) / (np.linalg.norm(cpg_state) + 1e-8)

            # for each leg, get rotated cpg state per its phase in gait_phases
            leg_gait_torques = []
            for i in range(len(gait_phases)):
                rotated_cpg = np.dot(rotation_matrices[i], cpg_state)
                # calc RBF activations based on distance b/w them and rotated cpg
                distances = np.linalg.norm(rbf_centers - rotated_cpg, axis=1)
                rbf_activations = np.exp(- (distances**2) / (2 * (rbf_sigma**2)))
                leg_torque = np.dot(rbf_activations, w_locomotion)
                leg_gait_torques.extend(leg_torque)

            virtual_gait_torques = np.array(leg_gait_torques)

            # steering signal swings positive/negative, and size of signal changes, depending on whether 
            # left or right is closer (and by how much)
            left_right_differential = rangefinders[0] - rangefinders[2]
            sensor_vector = np.array([rangefinders[1], left_right_differential])
            # combine with front sensor, and map to sensor torque modifiers via sensormap
            virtual_steering_torques = np.dot(sensor_reflex_map, sensor_vector)

            virtual_torques = virtual_gait_torques + virtual_steering_torques
            xml_torques = virtual_torques * sign_mask

            # pass through tanh activation function
            # note: ant/hex control ranges in XML are already -1 to 1 so can use these as is
            unique_actions = np.tanh(xml_torques)

        # takes a timestep with the robot carrying out the actions above.
        # note this goes through the step() defined in mujoco_env_wrapper for custom reward/measure calcs
        obs, reward, terminated, truncated, info = env.step(unique_actions)
        done = terminated or truncated
        linear_x_velocities.append(info["linear_x_velocity"])
        linear_y_velocities.append(info["linear_y_velocity"])
        angular_z_axis_velocities.append(info["angular_z_axis_velocity"])

        # fitness calculated as a sum over timesteps, rather than a single reward at the end
        # this lets us incorporate e.g. control cost penalties for high torques, collision penalties, etc.
        # see mujoco_env_wrapper for details
        # total_reward += reward

        #current_step += 1

        # tracking if goal was reached; if still false, take on value from this step. If ever true, stop checking so we don't overwrite!
        if (not reached_goal):
            reached_goal = info["success"]

        # measures extracted at final step
        if done:
            total_reward += reward

    env.close() # must close env after every sim!

    # final measures: average of the xy-plane velocities, average (absolute) turning/yaw velocity
    avg_forward_vel = np.mean(np.sqrt(np.array(linear_x_velocities)**2 + np.array(linear_y_velocities)**2))
    # we use absolute because we care about *how much* it turned throughout the run rather than which way it turned more/less
    avg_angular_vel = np.mean(np.abs(angular_z_axis_velocities)) 

    if (record_video):
        path = str(record_outdir / f'videos/{record_video_idx}.mp4')
        imageio.mimsave(path, frames, fps=30)

    return total_reward, avg_forward_vel, avg_angular_vel, reached_goal


def create_scheduler(
    seed: int | None, # seed for emitters/archives/etc. to use for sampling/randomness
    n_emitters: int, # number of emitters to use - in this experiment, will be 1 for all
    solution_dim: int, # dimensionality of controller
    lower_bounds: ArrayLike, # lower bounds on each solution dimension
    upper_bounds: ArrayLike, # upper bounds on each solution dimension
    dims: list[int], # defines number of bins per dimension in archive
    ranges: list[tuple[int,int]], # defines min/max ranges per dimension in archive
    algorithm: str,
    algorithm_params: dict[str, float | bool], # algorithm parameters
    batch_size: int, # number of samples to take for simulation at each iteration (per emitter!)
    qd_score_offset: int | float = 0.0,
) -> Scheduler: # pyribs scheduler which handles emitters. Note emitter and scheduler type vary by algorithm.

    # start from center of bounds
    # in cases of weights, this is generally 0, but e.g. w_cpg for cpg frequency ranges from 0.05 to 0.35
    initial_model = np.zeros(solution_dim)

    if (algorithm == "BOPElites"):
        # smaller starting archive for BOPElites based pon upscale_schedule
        main_archive = GridArchive(
            solution_dim=solution_dim,
            dims=algorithm_params["upscale_schedule"][0],
            ranges=ranges,
            seed=seed,
            qd_score_offset=qd_score_offset,
        )

    # for MAP/Bayes, this is only archive. For BOP, it's the "result archive" at the intended final dimensionality
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=dims,
        ranges=ranges,
        seed=seed,
        qd_score_offset=qd_score_offset,
    )

    # Seeds for emitters - note None means a random one is generated.
    # Otherwise, we want them to differ per emitter, but be replicable.
    seeds = (
        [None] * n_emitters if seed is None else [seed + i for i in range(n_emitters)]
    )

    # Creation of emitters and schedulers. This, along with the archive and scheduler, defines the algorithm used in pyribs.
    if (algorithm == "MAPElites"):
        emitters = [
            GaussianEmitter(
                archive=archive, # shared reference to the archive!
                x0=initial_model.flatten(), # point from which to begin sampling in MAP-Elites
                sigma=algorithm_params["sigmas"], # standard deviation for Gaussian mutation/sampling per dimension
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                #lower_bounds=np.full(solution_dim, -1), # lower bounds on each dimension of the solution (i.e. weight/bias)
                #upper_bounds=np.full(solution_dim, 1), # lower bounds on each dimension of the solution (i.e. weight/bias)
                batch_size=batch_size, # number of samples to take for simulation at each iteration
                seed=s,
            )
            for s in seeds
        ]
        scheduler = Scheduler(archive, emitters)
    elif (algorithm == "BayesOpt"):
        emitters = [
            BOEmitter(
                archive=archive,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                search_nrestarts=algorithm_params["search_nrestarts"], # ENSURE > 3 as we use warm start(!) Number of pattern search restarts for optimising acquisition function
                num_initial_samples=20*solution_dim, # doubled from BOP-Elites paper here: https://inria.hal.science/hal-04537563/file/main.pdf. We double b/c our simulations aren't that expensive and we're interested in wall-clock time.
                batch_size=batch_size,
                seed=s,
            )
            for s in seeds
        ]
        scheduler = BOScheduler(archive, emitters)
    elif (algorithm == "BOPElites"):
        emitters = [
            CustomBayesianOptimizationEmitter(
                archive=main_archive,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                search_nrestarts=algorithm_params["search_nrestarts"], # ENSURE > 3 as we use warm start(!) Number of pattern search restarts for optimising acquisition function
                entropy_ejie=algorithm_params["entropy_ejie"], # bool, whether to use entropy variant of EJIE
                upscale_schedule=algorithm_params["upscale_schedule"], # list[tuple[int,int]] of successively more granular archive dims
                min_obj=qd_score_offset, # precalculated minimum objective per maze; needed to evaluate expected improvement for empty cells in archive when calculating EJIE acquisition value
                num_initial_samples=20*solution_dim,#doubled from BOP-Elites paper here: https://inria.hal.science/hal-04537563/file/main.pdf. We double b/c our simulations aren't that expensive and we're interested in wall-clock time.
                batch_size=batch_size,
                seed=s,
            )
            for s in seeds
        ]
        scheduler = CustomBayesianOptimizationScheduler(main_archive, emitters, result_archive=archive)
    else:
        raise ValueError("Unknown algorithm!")

    return scheduler


def run_search(
    scheduler: Scheduler, # Scheduler for managing getting samples from emitters and informing them of results.
    maze_params: dict[str, Any], # maze params to use
    xml_file: str, # path to XML to use
    env_seed: int, # seed for creating the simulation environment
    time_to_run: int, # how long to run for
    log_freq: int # log metrics every X iterations
) -> tuple[dict[str, dict[str, list[int | float]]], dict[str, dict[str, list[int | float]]] | None]: # metrics to log etc.; used for graph generation later

    log.info(
        "> Starting search.\n"
    )

    metrics = {
        "Max Score": {
            "x": [],
            "y": [],
        },
        "Archive Size": {
            "x": [0],
            "y": [len(scheduler.archive)],
        },
        "QD Score": {
            "x": [0],
            "y": [scheduler.archive.stats.qd_score],
        },
    }

    # if BOP-Elites (or otherwise there is a passive/result_archive, log metrics for this as well)
    if (scheduler.result_archive != scheduler.archive):
        passive_metrics = {
                "Max Score": {
                    "x": [],
                    "y": [],
                },
                "Archive Size": {
                    "x": [0],
                    "y": [len(scheduler.result_archive)],
                },
                "QD Score": {
                    "x": [0],
                    "y": [scheduler.result_archive.stats.qd_score],
                },
            }
    else: passive_metrics = None

    start_time = time.time()
    goal_reached_counter = 0 # unimportant, just tracks how many runs reached the goal successfully in total
    elapsed_time = 0
    iteration = 0
    while elapsed_time < time_to_run:
        iteration += 1 # logs are every x iterations so we track it

        # Request models from the scheduler.
        sols = scheduler.ask()

        # Evaluate the models and record the objectives and measures.
        objs, meas = [], []

        # Ask the Dask client to distribute the simulations among the Dask workers, then
        # gather the results of the simulations.
        results = [simulate(model, maze_params, xml_file, env_seed) for model in sols]

        # Process the results.
        for obj, linear_vel, angular_vel, reached_goal in results:
            objs.append(obj)
            meas.append([linear_vel, angular_vel])
            if (reached_goal):
                goal_reached_counter += 1

        # Send the results back to the scheduler. It will pass them onto each emitter.
        scheduler.tell(objs, meas)
        #assert_single_threaded() #TODO: temp debug

        # Metrics.
        elapsed_time = time.time() - start_time
        metrics["Max Score"]["x"].append(elapsed_time)
        metrics["Max Score"]["y"].append(scheduler.archive.stats.obj_max)
        metrics["Archive Size"]["x"].append(elapsed_time)
        metrics["Archive Size"]["y"].append(len(scheduler.archive))
        metrics["QD Score"]["x"].append(elapsed_time)
        metrics["QD Score"]["y"].append(scheduler.archive.stats.qd_score)
        if (passive_metrics is not None):
            passive_metrics["Max Score"]["x"].append(elapsed_time)
            passive_metrics["Max Score"]["y"].append(scheduler.result_archive.stats.obj_max)
            passive_metrics["Archive Size"]["x"].append(elapsed_time)
            passive_metrics["Archive Size"]["y"].append(len(scheduler.result_archive))
            passive_metrics["QD Score"]["x"].append(elapsed_time)
            passive_metrics["QD Score"]["y"].append(scheduler.result_archive.stats.qd_score)

        # Logging.
        if (passive_metrics is not None): metrics_to_use = passive_metrics # for BOP-Elites, log the progress of the result_archive as that's our final result
        else: metrics_to_use = metrics

        # log first/last iteration, and every X iterations
        if iteration % log_freq == 0 or iteration == 1 or elapsed_time >= time_to_run:
            log.info(
                "> {} itrs completed after {:.2f} s\n"
                "  - Max Score: {}\n"
                "  - Archive Size: {}\n"
                "  - QD Score: {}\n"
                "  - Goal Reached Count: {}\n"
                "  - Archive Dims: {}",
                iteration,
                elapsed_time,
                metrics_to_use["Max Score"]["y"][-1],
                metrics_to_use["Archive Size"]["y"][-1],
                metrics_to_use["QD Score"]["y"][-1],
                goal_reached_counter,
                scheduler.archive.dims # track archive dimensions if upscaling with BOPElites
            )
    
    return metrics, passive_metrics


def save_heatmap(archive: GridArchive, filename: str | Path, min_obj: int | float) -> None:
    """Saves a heatmap of the scheduler's archive to the filename.

    Args:
        archive: Archive with results from an experiment.
        filename: Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=min_obj, vmax=0, ax=ax)
    ax.set_ylabel("Average angular velocity")
    ax.set_xlabel("Average xy-speed")
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
        50,  # Number of cells.
        histtype="step",
        density=False,
        cumulative=-1,  # CCDF rather than CDF.
    )
    ax.set_xlabel("Objectives")
    ax.set_ylabel("Num. Entries")
    ax.set_title("Distribution of Archive Objectives")
    fig.savefig(filename)


def run_evaluation(
        outdir: Path, # directory containing archive.csv to run evaluations on
        num_to_sim: int, # top X elites to evaluate
        maze_params: dict[str, Any], # maze params to use
        xml_file: str, # path to XML to use
        env_seed: int, # seed for creating the simulation environment
        seed: int | None,
        use_saved_emitter_0: bool = False
    ) -> None:

    if (use_saved_emitter_0):
        emitter = pickle.load(open(outdir / "emitter0.sav"))
        df = ArchiveDataFrame(emitter.archive.data(return_type="pandas"))
    else:
        df = ArchiveDataFrame(pd.read_csv(outdir / "archive.csv"))

    df_top = ArchiveDataFrame(df.sort_values(by="objective", ascending=False).head(num_to_sim))

    video_folder = outdir / "videos"
    video_folder.mkdir(parents=True, exist_ok=True)

    # iterate through the top X elites, simulate them, and log stats
    for idx, elite in enumerate(df_top.iterelites()):
        model = elite["solution"]
        archive_idx = df_top.index[idx]

        reward, final_lin_vel, final_ang_vel, _ = simulate(model, maze_params, xml_file, env_seed, record_video=True, record_video_idx=idx, record_outdir=outdir)
        log.info(
            "=== Index {} ===\n"
            "Model:\n"
            "{}\n"
            "Reward: {}\n"
            "Original Reward: {}\n"
            "Final speeds (lin/ang): {} / {} \n"
            "Original Final speeds: {}\n",
            archive_idx,
            model,
            reward,
            elite["objective"],
            final_lin_vel, final_ang_vel,
            elite["measures"]
        )


def mujoco_main(
    algorithm: str = "MAPElites", # algorithm to use: MAPElites, BayesOpt, BOPElites
    algorithm_params: dict[str, float | bool | list[tuple[int, int]]] = {"sigma0": 0.1}, # paramater dict for algorithm. See create_scheduler for details per algorithm.
    save_emitter_0: bool = True, # pickle emitter 0 to reuse later. Will include a copy of archive, GP, etc.
    maze_str: str = "MEDIUM_MAZE", # OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE
    xml_file: str = "/users/40795510/sharedscratch/dissertation/rangefinder_hex.xml", # path to XML to use. Ensure compatibility with script (e.g. 3x rangefinders expected)
    env_seed: int = 52, # seed for creating the simulation environment
    time_to_run: int = 60, # seconds to run for
    log_freq: int = 5, # log metrics every X iterations
    n_emitters: int = 1, # number of emitters to use
    batch_size: int = 5, # number of samples to take for simulation at each iteration (per emitter!)
    # NOTE: while batch size = 1 is best for sample efficiency, a higher batch count can be better for wall-clock time. See (and [80, 143]): https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
    seed: int | None = None, # seed for emitters/archives/etc. to use for sampling/randomness
    outdir: str | None = None, # directory to which to save outputs (if running search) - and read archive and create videos of solutions (if running eval)
    run_eval: bool = False, # set true and provide outdir to eval and create videos of existing archive elites
    run_eval_num_to_sim: int | None = None, # top X elites to evaluate and create videos of, if run_eval true
    run_eval_use_saved_emitter_0: bool = False, # if run_eval true, whether to use the saved emitter 0 (if it exists) or just read the archive.csv. For floating point precision.
) -> None:

    # if printing, be detailed - otherwise floating point truncation
    np.set_printoptions(precision=17, suppress=True) 
    # register gymnasium robotics environments, i.e. AntMaze
    gym.register_envs(gymnasium_robotics)

    # MAZE-BASED PARAMS:
    if (maze_str == "OPEN"):
        maze = OPEN
        max_episode_steps = 1800
        archive_dims = [20, 20]
        archive_ranges = [(0, 0.8), (0, 2.0)]
        maze_options = { "goal_cell": np.array([1, 4], dtype=int),
                        "reset_cell": np.array([3, 1], dtype=int)
                        }
        maze_max_dist = 23.0
        min_obj = -17.0
    elif (maze_str == "U_MAZE"):
        maze = U_MAZE
        max_episode_steps = 1800
        archive_dims = [20, 20]
        archive_ranges = [(0, 0.8), (0, 2.0)]
        maze_options = { "goal_cell": np.array([1, 1], dtype=int),
                        "reset_cell": np.array([3, 1], dtype=int)
                        }
        maze_max_dist = 12.0
        min_obj = -14.0
    elif (maze_str == "MEDIUM_MAZE"):
        maze = MEDIUM_MAZE
        max_episode_steps = 3000
        archive_dims = [20, 20]
        archive_ranges = [(0, 0.8), (0, 2.0)]
        maze_options = { "goal_cell": np.array([6, 5], dtype=int),
                        "reset_cell": np.array([6, 1], dtype=int)
                        }
        maze_max_dist = 28.0
        min_obj = -28.0
    elif (maze_str == "LARGE_MAZE"):
        maze = LARGE_MAZE
        max_episode_steps = 4000
        archive_dims = [20, 20]
        archive_ranges = [(0, 0.8), (0, 2.0)]
        maze_options = { "goal_cell": np.array([3, 8], dtype=int),
                        "reset_cell": np.array([7, 1], dtype=int)
                        }
        maze_max_dist = 40.0
        min_obj = -35.0
    else:
        raise ValueError("Unknown map!")

    maze_params = {
        "maze": maze,
        "max_episode_steps": max_episode_steps,
        "maze_options": maze_options,
        "maze_max_dist": maze_max_dist,
        "min_obj": min_obj
    }


    # XML-BASED PARAMS:
    # FUTURE WORK/TODO: parametrise this, allow users to input e.g. how many RBFs and let it crash if specified wrong
    if (xml_file == "/users/40795510/sharedscratch/dissertation/rangefinder_ant.xml"):
        solution_dim = 25
        lower_bounds=np.hstack([0.05, np.full(20, -0.8), np.full(4, -1.5)]) # cpg freq, rbf-torque weights, sensor reflex weights
        upper_bounds=np.hstack([0.25, np.full(20, 0.8), np.full(4, 1.5)])
        xml_str = "ant"
    elif(xml_file == "/users/40795510/sharedscratch/dissertation/rangefinder_hex.xml"):
        solution_dim = 25
        lower_bounds=np.hstack([0.05, np.full(20, -0.8), np.full(4, -1.5)]) # cpg freq, rbf-torque weights, sensor reflex weights
        upper_bounds=np.hstack([0.25, np.full(20, 0.8), np.full(4, 1.5)])
        xml_str = "hex"
    elif (xml_file == "/users/40795510/sharedscratch/dissertation/new_racecar.xml"):
        solution_dim = 10
        lower_bounds=np.full(solution_dim, -1) # raw weights from inputs to outputs. Note outputs get multiplied by ctrl_range (20, 0.785 from XML)
        upper_bounds=np.full(solution_dim, 1) # raw weights. Note outputs get multiplied by ctrl_range (20, 0.785 from XML)
        xml_str = "car"
    else:
        raise ValueError("Unknown XML!")
    


    # ALGORITHM-BASED PARAMS:
    # for MAP-Elites, calculate the sigma scale per dimension based on passed in sigma0 scaler and the solution bounds
    if (algorithm == "MAPElites"):
        algorithm_params["sigmas"] = (np.asarray([upper_bounds]) - np.asarray([lower_bounds])) * algorithm_params["sigma0"]


    # if running evaluation...
    if run_eval:
        if outdir is None:
            raise ValueError("outdir must be provided to run an eval.")
        outdir = Path(outdir)
        # switch log
        log.remove()
        log.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", colorize=True) 
        log.add(outdir / "eval.log")
        log.info("Evaluating solutions in {}", outdir)
        run_evaluation(outdir, run_eval_num_to_sim, maze_params, xml_file, env_seed, seed, use_saved_emitter_0=run_eval_use_saved_emitter_0)
        return



    # if running search...

    # Initialize output directory.
    outdir = (
        (
            Path("dissertation_logs")
            / Path(__file__).stem
            / f"{algorithm}_{maze_str}_{xml_str}"
            / datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_seed-{seed}")
        )
        if outdir is None
        else Path(outdir)
    )
    outdir.mkdir(parents=True, exist_ok=False)

    # Initialize loggers --
    log.remove()
    log.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", colorize=True)
    log.add(outdir / "out.log")  # Save logs in outdir.
    log.info("Saving outputs to: {}", outdir)

    # set up archive, scheduler, and emitters, which define the algorithm.
    scheduler = create_scheduler(seed, n_emitters, solution_dim, lower_bounds, upper_bounds, archive_dims, archive_ranges, algorithm, algorithm_params, batch_size, qd_score_offset=maze_params["min_obj"])

    # run search, returning dict(s) of metrics
    metrics, passive_metrics = run_search(scheduler, maze_params, xml_file, env_seed, time_to_run, log_freq)

    # Use metrics to create output graphs etc.
    scheduler.archive.data(return_type="pandas").to_csv(outdir / "archive.csv")
    save_ccdf(scheduler.archive, outdir / "archive_ccdf.png")
    save_heatmap(scheduler.archive, outdir / "archive_heatmap.png", maze_params["min_obj"])
    save_metrics(outdir, metrics)
    if (passive_metrics is not None):
        scheduler.result_archive.data(return_type="pandas").to_csv(outdir / "result_archive.csv")
        save_ccdf(scheduler.result_archive, outdir / "result_archive_ccdf.png")
        save_heatmap(scheduler.result_archive, outdir / "result_archive_heatmap.png", maze_params["min_obj"])
        save_metrics(outdir, passive_metrics, "result")

    # save emitter (including its archive, GP, etc.) to load and use later.
    if (save_emitter_0):
        emitter_filepath=str(outdir / "emitter0.sav")
        pickle.dump(scheduler._emitters[0], open(emitter_filepath, 'wb'))

    # if enabled, use BOP-Elites prediction map to predict and then actually assess elites at the final result_archive resolution
    if (algorithm == "BOPElites" and algorithm_params["make_prediction_archive"]):
        create_predicted_archive(
            solution_dim=solution_dim,
            archive_dims=archive_dims,
            archive_ranges=archive_ranges,
            maze_params=maze_params,
            xml_file=xml_file,
            env_seed=env_seed,
            scheduler=scheduler,
            seed=seed,
            outdir=outdir
        )
            

"""
# debug tool to test if single-threadedness is working
def assert_single_threaded():
    current_process = psutil.Process(os.getpid())
    
    thread_count = current_process.num_threads()

    if thread_count > 2: 
        print(f"Multithreading active, active OS threads: {thread_count}")
    else:
        print(f"Single-threaded, active OS threads: {thread_count}")
"""

def create_predicted_archive(
        solution_dim: int,
        archive_dims: list[int],
        archive_ranges: list[tuple[int,int]],
        maze_params: dict[str, Any],
        xml_file: str,
        env_seed: int,
        scheduler: Scheduler,
        seed: int | None,
        outdir: Path,
) -> None:
    log.info("Beginning Predictive Map")
    sols = scheduler.emitters[0].get_predicted_elites(scheduler.result_archive.boundaries)
    log.info("Finished Predictive Map. Simulating...")

    predicted_archive = GridArchive(
                    solution_dim=solution_dim,
                    dims=archive_dims,
                    ranges=archive_ranges,
                    seed=seed,
                    qd_score_offset=maze_params["min_obj"],
                )

    results = [simulate(model, maze_params, xml_file, env_seed) for model in sols]
    
    objs = []
    meas = []
    predicted_goal_reached_counter = 0

    for obj, linear_vel, angular_vel, reached_goal in results:
        objs.append(obj)
        meas.append([linear_vel, angular_vel])
        if (reached_goal == True):
            predicted_goal_reached_counter += 1

    predicted_archive.add(sols, objs, meas)

    # save, graphs, etc.
    predicted_archive.data(return_type="pandas").to_csv(outdir / "predicted_archive.csv")
    save_ccdf(predicted_archive, outdir / "predicted_archive_ccdf.png")
    save_heatmap(predicted_archive, outdir / "predicted_archive_heatmap.png", maze_params["min_obj"])
    log.info(
                "=== Predicted Archive Stats ===\n"
                "QD-Score: {}\n"
                "Max Reward: {}\n"
                "Coverage: {}\n"
                "Reached Goal Counter: {}",
                predicted_archive.stats.qd_score,
                predicted_archive.stats.obj_max,
                predicted_archive.stats.coverage,
                predicted_goal_reached_counter
            )

if __name__ == "__main__":
    fire.Fire(mujoco_main)
