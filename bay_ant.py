from __future__ import annotations

# force single-threaded for fair performance across trials
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 
# if issues with graphics drivers with gladGL error when trying to create videos, try the below:
"""
os.environ["MUJOCO_GL"] = "egl"
"""

import torch
torch.set_num_threads(1)
# if printing, be detailed - otherwise floating point truncation
import numpy as np
np.set_printoptions(precision=17, suppress=True) 

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
from numpy.typing import ArrayLike
from numpy import random
import pandas as pd
import gymnasium as gym
import gymnasium_robotics
from loguru import logger as log

# pyribs and customised versions
from ribs.archives import ArchiveBase, ArchiveDataFrame, GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler
from BOPE_emitter import CustomBayesianOptimizationEmitter # import from custom file, not from pyribs. We made a couple small changes.
from BOPE_scheduler import CustomBayesianOptimizationScheduler # same as the default, just changed to expect CustomBayesianOptimizationEmitter not pyribs one
from BO_emitter import BOEmitter # custom BO-only emitter and scheduler
from BO_scheduler import BOScheduler

from gymnasium_robotics.envs.maze.maps import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE
# custom environment wrapper over AntMaze
from mujoco_env_wrapper import MujocoEnvWrapper

from legged_controller import LeggedController

from diss_utils import save_heatmap, save_ccdf, save_metrics, check_ram_usage #TODO: temp debugtool, assert_single_threaded


def simulate(
    model: np.ndarray, # The array of weights/biases for the controller policy.
    maze_params: dict[str, Any], # dict for maze parameters; dictated by maze input
    xml_file: str, # path to XML to use
    controller_params: dict[str, Any] | None = None, # none defaults to Ant with 10 RBFs in unilateral mode
    seed: int | None = None, # environment seed
    record_video: bool = False, # If passed in, this will be used instead of creating a new env; mainly for recording video, see run_evaluation().
    record_video_idx: int | None = None,
    record_outdir: Path | None = None
) -> tuple[float, float, float, bool]: # total_reward (objective), avg linear velocity and avg z-axis rotation (measures), and whether the goal was reached.

    maze = maze_params["maze"]
    max_episode_steps = maze_params["max_episode_steps"]
    maze_options = maze_params["maze_options"]
    maze_max_dist = maze_params["maze_max_dist"]

    if not record_video:
        base_env = gym.make("AntMaze_UMazeDense-v5", maze_map=maze, continuing_task=False, xml_file=xml_file, max_episode_steps=max_episode_steps)
    else:
        base_env = gym.make("AntMaze_UMazeDense-v5", maze_map=maze, continuing_task=False, xml_file=xml_file, max_episode_steps=max_episode_steps, render_mode="rgb_array")
        frames=[]
    base_env.unwrapped.include_sensors = True
    env = MujocoEnvWrapper(base_env, max_episode_steps)

    # fitness
    total_reward = 0.0
    path_distances = []
    #measures:
    control_costs = []
    torso_heights = []

    #characteristics we track for interest:
    linear_x_velocities = []
    linear_y_velocities = []
    min_front_rangefinder = maze_max_dist

    obs, _ = env.reset(seed=seed, options=maze_options) # reset has to be called first per Gym/Mujoco Documentation!
    done = False # track whether session was terminated or truncated
    
    # initialise
    if (xml_file == "/home/henry/dissertation_5thAug/new_racecar.xml"):
        # uses simple/direct policy
        w_end = 8
        weights = model[:w_end].reshape(4, 2)
        biases = model[w_end:]
    else:
        if (controller_params is None):
            controller = LeggedController(model) # defaults to ant controller with 10 RBFs and a wide RBF sigma
        else:
            controller = LeggedController(
                model,
                num_legs=controller_params["num_legs"],
                gait_phases=controller_params["gait_phases"],
                sign_mask=controller_params["sign_mask"],
                num_rbfs=controller_params["num_rbfs"],
                rbf_sigma=controller_params["rbf_sigma"],
                sensor_mode=controller_params["sensor_mode"]
            )

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
        # save after doing that flip, otherwise min_rangefinder will be -1 for those who didn't hit anything at any point
        if (rangefinders_data[1] < min_front_rangefinder):
            min_front_rangefinder = rangefinders_data[1]
        # we then shift so close walls send strong signal (1) while far walls send a weak signal (0)
        rangefinders = 1.0 - np.clip(rangefinders_data / maze_max_dist, 0.0, 1.0)

        # CAR:
        if (xml_file == "/home/henry/dissertation_5thAug/new_racecar.xml"):
            velocimeter_x = mujoco_data.sensordata[3].copy()
            velocimeter_y = mujoco_data.sensordata[4].copy()
            xy_speed_ratio = np.clip((math.sqrt(velocimeter_x**2 + velocimeter_y**2) / 1.5), 0.0, 1.0) # top speed is ~1.5
            # direct policy between obs and actions based on pure model weights (range -1 to 1)
            pruned_obs = [rangefinders[0], rangefinders[1], rangefinders[2], xy_speed_ratio]
            unique_actions = np.dot(pruned_obs, weights) + biases
            unique_actions = np.tanh(unique_actions)
            # multiply output by car XML control range to get actual actions
            unique_actions[0] = unique_actions[0] * 30
            unique_actions[1] = unique_actions[1] * 0.785
        # ANT:
        else:
            unique_actions = controller.get_action(rangefinders)

        # takes a timestep with the robot carrying out the actions above.
        # note this goes through the step() defined in mujoco_env_wrapper for custom reward/measure calcs
        obs, reward, terminated, truncated, info = env.step(unique_actions)
        done = terminated or truncated
        linear_x_velocities.append(info["linear_x_velocity"])
        linear_y_velocities.append(info["linear_y_velocity"])
        torso_heights.append(info["torso_height"])
        control_costs.append(info["control_cost"])
        path_distances.append(info["path_dist"])

        # characteristics extracted at final step
        if done:
            final_x_pos = info["x_pos"]
            final_y_pos = info["y_pos"]

    env.close() # must close env after every sim!

    # fitness: distance travelled
    total_reward = np.sum(path_distances)

    # measures
    avg_torso_height = np.mean(torso_heights)
    total_control_cost = np.sum(control_costs)

    # characteristic: average of the xy-plane velocities
    avg_forward_vel = np.mean(np.sqrt(np.array(linear_x_velocities)**2 + np.array(linear_y_velocities)**2))
    # other characteristic, min_rangefinder, is ready to go from loop
    
    if (record_video):
        path = str(record_outdir / f'videos/{record_video_idx}.mp4')
        imageio.mimsave(path, frames, fps=30)

    return total_reward, avg_torso_height, total_control_cost, final_x_pos, final_y_pos, avg_forward_vel, min_front_rangefinder


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

    if (algorithm == "BOPElites"):
        # smaller starting archive for BOPElites based pon upscale_schedule
        main_archive = GridArchive(
            solution_dim=solution_dim,
            dims=algorithm_params["upscale_schedule"][0],
            ranges=ranges,
            seed=seed,
            qd_score_offset=qd_score_offset,
            extra_fields={"chars": ((4,), np.float64)} # store other characteristics as well
        )

    # for MAP/Bayes, this is only archive. For BOP, it's the "result archive" at the intended final dimensionality
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=dims,
        ranges=ranges,
        seed=seed,
        qd_score_offset=qd_score_offset,
        extra_fields={"chars": ((4,), np.float64)} # store other characteristics as well
    )

    # Seeds for emitters - note None means a random one is generated.
    # Otherwise, we want them to differ per emitter, but be replicable.
    seeds = (
        [None] * n_emitters if seed is None else [seed + i for i in range(n_emitters)]
    )

    # Creation of emitters and schedulers. This, along with the archive and scheduler, defines the algorithm used in pyribs.
    if (algorithm == "MAPElites"):
        # start from zero, as almost all weights centre there
        # w_cpg for legged controllers is different, ranges 0.05 to 0.25, but will get clipped up to 0.05 min due to bounds we pass in
        # starting at a slow 0.05 is good for ensuring early iterations don't immediately crash and burn
        initial_model = np.zeros(solution_dim)

        emitters = [
            GaussianEmitter(
                archive=archive, # shared reference to the archive!
                x0=initial_model.flatten(), # point from which to begin sampling in MAP-Elites
                sigma=algorithm_params["sigmas"], # standard deviation for Gaussian mutation/sampling per dimension
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
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
                num_initial_samples=10*solution_dim, #  from BOP-Elites paper here: https://inria.hal.science/hal-04537563/file/main.pdf.
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
                num_initial_samples=10*solution_dim,# from BOP-Elites paper here: https://inria.hal.science/hal-04537563/file/main.pdf.
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
    controller_params: dict[str, Any] | None, # controller params to use
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
        "Mean Score": {
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
        "Coverage": {
            "x": [0],
            "y": [scheduler.archive.stats.coverage]
        },
        "Sim Time": {
            "x": [],
            "y": []
        },
        "RAM Usage": {
            "x": [],
            "y": []
        }
    }

    # if BOP-Elites (or otherwise there is a passive/result_archive, log metrics for this as well)
    if (scheduler.result_archive != scheduler.archive):
        passive_metrics = {
                "Max Score": {
                    "x": [],
                    "y": [],
                },
                "Mean Score": {
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
                "Coverage": {
                    "x": [0],
                    "y": [scheduler.result_archive.stats.coverage]
                },
            }
    else: passive_metrics = None

    start_time = time.time()
    total_elapsed_time = 0
    sim_time = 0
    alg_time = 0
    iteration = 0
    ram_usage = 0
    while total_elapsed_time < time_to_run:
        iteration += 1 # logs are every x iterations so we track it

        loop_time = time.time()
        # Request models from the scheduler.
        sols = scheduler.ask()
        alg_time += time.time() - loop_time

        # Evaluate the models and record the objectives and measures.
        objs, meas, chars = [], [], []
        
        loop_time = time.time()
        # Simulate suggested models/parameters
        results = [simulate(model, maze_params, xml_file, controller_params, seed=env_seed) for model in sols]
        sim_time += time.time() - loop_time

        # Process the results.
        for obj, avg_torso_height, total_control_cost, final_x_pos, final_y_pos, avg_lin_vel, min_rangefinder in results:
            objs.append(obj)
            meas.append([avg_torso_height, total_control_cost])
            chars.append([final_x_pos, final_y_pos, avg_lin_vel, min_rangefinder])

        loop_time = time.time()
        # Send the results back to the scheduler. It will pass them onto each emitter.
        scheduler.tell(objs, meas, chars=chars)
        alg_time += time.time() - loop_time

        #assert_single_threaded() #TODO: temp debug
        ram_usage = check_ram_usage(from_main=True)

        # Metrics.
        total_elapsed_time = time.time() - start_time
        metrics["Max Score"]["x"].append(total_elapsed_time)
        metrics["Max Score"]["y"].append(scheduler.archive.stats.obj_max)
        metrics["Mean Score"]["x"].append(total_elapsed_time)
        metrics["Mean Score"]["y"].append(scheduler.archive.stats.obj_mean)
        metrics["Archive Size"]["x"].append(total_elapsed_time)
        metrics["Archive Size"]["y"].append(len(scheduler.archive))
        metrics["Coverage"]["x"].append(total_elapsed_time)
        metrics["Coverage"]["y"].append(scheduler.archive.stats.coverage)
        metrics["QD Score"]["x"].append(total_elapsed_time)
        metrics["QD Score"]["y"].append(scheduler.archive.stats.qd_score)
        metrics["Sim Time"]["x"].append(total_elapsed_time)
        metrics["Sim Time"]["y"].append(sim_time)
        metrics["RAM Usage"]["x"].append(total_elapsed_time)
        metrics["RAM Usage"]["y"].append(ram_usage)
        if (passive_metrics is not None):
            passive_metrics["Max Score"]["x"].append(total_elapsed_time)
            passive_metrics["Max Score"]["y"].append(scheduler.result_archive.stats.obj_max)
            passive_metrics["Mean Score"]["x"].append(total_elapsed_time)
            passive_metrics["Mean Score"]["y"].append(scheduler.result_archive.stats.obj_mean)
            passive_metrics["Archive Size"]["x"].append(total_elapsed_time)
            passive_metrics["Archive Size"]["y"].append(len(scheduler.result_archive))
            passive_metrics["QD Score"]["x"].append(total_elapsed_time)
            passive_metrics["QD Score"]["y"].append(scheduler.result_archive.stats.qd_score)
            passive_metrics["Coverage"]["x"].append(total_elapsed_time)
            passive_metrics["Coverage"]["y"].append(scheduler.result_archive.stats.coverage)


        # Logging.

        # for BOP-Elites, log the progress of the result_archive as that's our final result
        if (passive_metrics is not None): metrics_to_use = passive_metrics 
        else: metrics_to_use = metrics

        # log first/last iteration, and every X iterations
        if iteration % log_freq == 0 or iteration == 1 or total_elapsed_time >= time_to_run:
            log.info(
                "> {} itrs completed after {:.2f} s\n"
                "  - Max Score: {}\n"
                "  - Archive Size: {}\n"
                "  - QD Score: {}\n"
                "  - Archive Dims: {}\n"
                "  - Alg Time: {:.2f} s\n"
                "  - Sim Time: {:.2f} s\n"
                "  - RAM Usage: {} GB",
                iteration,
                total_elapsed_time,
                metrics_to_use["Max Score"]["y"][-1],
                metrics_to_use["Archive Size"]["y"][-1],
                metrics_to_use["QD Score"]["y"][-1],
                scheduler.archive.dims, # track archive dimensions if upscaling with BOPElites!
                alg_time,
                sim_time,
                ram_usage
            )
    
    return metrics, passive_metrics


def mujoco_main(
    algorithm: str = "BayesOpt", # algorithm to use: MAPElites, BayesOpt, BOPElites
    #TODO: return this to empty dict string thing when uploading to cluster
    algorithm_params = {"search_nrestarts": 10}, # paramater dict for algorithm. See create_scheduler for details per algorithm. Pass in as JSON.
    save_emitter_0: bool = True, # pickle emitter 0 to reuse later. Will include a copy of archive, GP, etc.
    maze_str: str = "MEDIUM_MAZE", # OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE
    xml_file: str = "/home/henry/dissertation_5thAug/rangefinder_ant.xml", # path to XML to use. Ensure compatibility with script (e.g. 3x rangefinders expected)
    sensor_mode: str = "unilateral", # sensor mode to use
    num_rbfs: int = 10, # number of rbfs to use in legged controller
    #TODO: consider change default here and in legged_controller to a value based on the General Neural Locomotion Framework paper; here, it's 75% of dist between each RBF
    rbf_sigma: float = ((math.sqrt(5)-1)/2) * (0.75), # rbf_sigma, see LeggedController for explanation.
    env_seed: int = 52, # seed for creating the simulation environment
    time_to_run: int = 36000, # seconds to run for, default 60 to avoid costly mistakes!
    log_freq: int = 5, # log metrics every X iterations
    n_emitters: int = 1, # number of emitters to use
    batch_size: int = 5, # number of samples to take for simulation at each iteration (per emitter!)
    # NOTE: while batch size = 1 is best for sample efficiency, a higher batch count can be better for wall-clock time. See parallelisation at (and [80, 143]): https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
    seed: int | None = None, # seed for emitters/archives/etc. to use for sampling/randomness
    outdir: str | None = None, # directory to which to save outputs (if running search) - and read archive and create videos of solutions (if running eval)
    run_eval: bool = False, # set true and provide outdir to eval and create videos of existing archive elites
    run_eval_num_to_sim: int | None = None, # top X elites to evaluate and create videos of, if run_eval true
    run_eval_use_saved_emitter_0: bool = True, # if run_eval true, whether to use the saved emitter 0 (if it exists) or just read the archive.csv. For floating point precision, makes a BIG difference. With this option enabled, eval results = search results.
) -> None:

    # on HPC we pass algorithm_params in as a JSON string, see params.text
    if (isinstance(algorithm_params, str)):
        algorithm_params = json.loads(algorithm_params)

    # register gymnasium robotics environments, i.e. AntMaze
    gym.register_envs(gymnasium_robotics)

    # MAZE-BASED PARAMS:
    if (maze_str == "OPEN"):
        maze = OPEN
        max_episode_steps = 1800
        maze_options = { "goal_cell": np.array([1, 4], dtype=int),
                        "reset_cell": np.array([3, 1], dtype=int)
                        }
        maze_max_dist = 23.0 # approx max rangefinder dist possible
        min_obj = 0.0 #if switching to goal again, -17.0 # approx max distance from goal
        archive_dims = [12, 25] #TODO: tune
        archive_ranges = [(0.2, 0.8), (0, 500)] # TODO: tune
    elif (maze_str == "U_MAZE"):
        maze = U_MAZE
        max_episode_steps = 1800
        maze_options = { "goal_cell": np.array([1, 1], dtype=int),
                        "reset_cell": np.array([3, 1], dtype=int)
                        }
        maze_max_dist = 12.0
        min_obj = 0.0 #if switching to goal again, -14.0
        archive_dims = [12, 25] #TODO: tune
        archive_ranges = [(0.2, 0.8), (0, 500)] # TODO: tune
    elif (maze_str == "MEDIUM_MAZE"):
        maze = MEDIUM_MAZE
        max_episode_steps = 3000
        maze_options = { "goal_cell": np.array([6, 5], dtype=int),
                        "reset_cell": np.array([6, 1], dtype=int)
                        }
        maze_max_dist = 28.0
        min_obj = 0.0 #if switching to goal again, -28.0
        archive_dims = [12, 25] #TODO: tune
        archive_ranges = [(0.2, 0.8), (0, 500)] # TODO: tune
    elif (maze_str == "LARGE_MAZE"):
        maze = LARGE_MAZE
        max_episode_steps = 4000
        maze_options = { "goal_cell": np.array([3, 8], dtype=int),
                        "reset_cell": np.array([7, 1], dtype=int)
                        }
        maze_max_dist = 40.0
        min_obj = 0.0 #if switching to goal again, -35.0
        archive_dims = [12, 25] #TODO: tune
        archive_ranges = [(0.2, 0.8), (0, 500)] # TODO: tune
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
    if (xml_file == "/home/henry/dissertation_5thAug/new_racecar.xml"):
        solution_dim = 10
        lower_bounds=np.full(solution_dim, -1) # raw weights from inputs to outputs. Note outputs get multiplied by ctrl_range (20, 0.785 from XML)
        upper_bounds=np.full(solution_dim, 1) # raw weights. Note outputs get multiplied by ctrl_range (20, 0.785 from XML)
        xml_str = "car"
        controller_params = None # doesn't get used for car test case
    elif (xml_file == "/home/henry/dissertation_5thAug/rangefinder_ant.xml"):
        num_legs = 4
        gait_phases = np.array([
                                0.0, # frontright
                                np.pi, # frontleft
                                np.pi, # backright
                                0.0 # backleft
                            ])
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
        xml_str = "ant"
        if (sensor_mode == "unilateral"):
            rbfs_num_weights = int(2 * num_rbfs)
            solution_dim = 5 + rbfs_num_weights
            lower_bounds=np.hstack([0.05, np.full(rbfs_num_weights, -0.8), np.full(4, -1.5)]) # cpg freq, rbf-torque weights, sensor reflex weights
            upper_bounds=np.hstack([0.25, np.full(rbfs_num_weights, 0.8), np.full(4, 1.5)])
        else:
            #TODO!
            raise NotImplementedError("Whoops!")
    elif(xml_file == "/home/henry/dissertation_5thAug/rangefinder_hex.xml"):
        num_legs = 6
        gait_phases = np.array([
                                0.0, # frontright
                                np.pi, # frontleft
                                0.0, # backright
                                np.pi, #backleft
                                np.pi, # midright
                                0.0 # midleft
                            ])
        sign_mask = np.array([
                                1.0, # frontright hip, +ve torque is forward
                                1.0, # frontright ankle, +ve torque is down
                                -1.0, # leftright hip, -ve torque is forward
                                1.0, # leftright ankle, +ve torque is down
                                1.0, # backright hip, +ve torque is forward
                                -1.0, # backright ankle, -ve torque is down
                                -1.0, # backleft hip, -ve torque is forward
                                -1.0, # backleft hip, -ve torque is down
                                1.0, # midright hip, mid hips behave same as back
                                -1.0,  # midright ank
                                -1.0,  # midleft hip
                                -1.0 # midleft ank
                            ])
        xml_str = "hex"
        if (sensor_mode == "unilateral"):
            rbfs_num_weights = int(2 * num_rbfs)
            solution_dim = 5 + rbfs_num_weights
            lower_bounds=np.hstack([0.05, np.full(rbfs_num_weights, -0.8), np.full(4, -1.5)]) # cpg freq, rbf-torque weights, sensor reflex weights
            upper_bounds=np.hstack([0.25, np.full(rbfs_num_weights, 0.8), np.full(4, 1.5)])
        else:
            #TODO!
            raise NotImplementedError("Whoops!")

    else:
        raise ValueError("Unknown XML!")

    controller_params = {
        "num_legs": num_legs,
        "gait_phases": gait_phases,
        "sign_mask": sign_mask,
        "num_rbfs": num_rbfs,
        "rbf_sigma": rbf_sigma,
        "sensor_mode": sensor_mode,
    }
    
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
        run_evaluation(outdir, run_eval_num_to_sim, xml_file, env_seed, seed, use_saved_emitter_0=run_eval_use_saved_emitter_0)
        return




    # if running search...

    # Initialize output directory.
    outdir = (
        (
            Path("dissertation_logs")
            / Path(__file__).stem
            / f"{algorithm}_{maze_str}_{xml_str}"
            / datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_seed-{seed}_{random.randint(100000)}") # add randint just in case two scripts get kicked off at exactly same time in batch
        )
        if outdir is None
        else Path(outdir)
    )
    outdir.mkdir(parents=True, exist_ok=False)

    # save parameters!
    with open(outdir / "controller_params.sav", "wb") as f:
        pickle.dump(controller_params, f)
    with open(outdir / "maze_params.sav", "wb") as f:
        pickle.dump(maze_params, f)
    with open(outdir / "algorithm_params.sav", "wb") as f:
        pickle.dump(algorithm_params, f)

    # Initialize loggers --
    log.remove()
    log.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", colorize=True)
    log.add(outdir / "out.log")  # Save logs in outdir.
    log.info("Saving outputs to: {}", outdir)

    # set up archive, scheduler, and emitters, which define the algorithm.
    scheduler = create_scheduler(seed, n_emitters, solution_dim, lower_bounds, upper_bounds, archive_dims, archive_ranges, algorithm, algorithm_params, batch_size, qd_score_offset=maze_params["min_obj"])

    # run search, returning dict(s) of metrics
    metrics, passive_metrics = run_search(scheduler, maze_params, controller_params, xml_file, env_seed, time_to_run, log_freq)

    # Use metrics to create output graphs etc.
    scheduler.archive.data(return_type="pandas").to_csv(outdir / "archive.csv")
    save_ccdf(scheduler.archive, outdir / "archive_ccdf.png")
    save_heatmap(scheduler.archive, outdir / "archive_heatmap.png", maze_params["min_obj"], scheduler.archive.stats.obj_max)
    save_metrics(outdir, metrics)
    if (passive_metrics is not None):
        scheduler.result_archive.data(return_type="pandas").to_csv(outdir / "result_archive.csv")
        save_ccdf(scheduler.result_archive, outdir / "result_archive_ccdf.png")
        save_heatmap(scheduler.result_archive, outdir / "result_archive_heatmap.png", maze_params["min_obj"], scheduler.result_archive.stats.obj_max)
        save_metrics(outdir, passive_metrics, "result")

    # save emitter (including its archive, GP, etc.) to load and use later.
    if (save_emitter_0):
        with open(outdir / "emitter0.sav", "wb") as f:
            pickle.dump(scheduler._emitters[0], f)

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

def run_evaluation(
        outdir: Path, # directory containing archive.csv to run evaluations on
        num_to_sim: int, # top X elites to evaluate
        xml_file: str, # path to XML to use
        env_seed: int, # seed for creating the simulation environment
        seed: int | None,
        use_saved_emitter_0: bool = False
    ) -> None:

    # retrieve parameters
    with open(outdir / "controller_params.sav", "rb") as f:
        controller_params = pickle.load(f)
    with open(outdir / "maze_params.sav", "rb") as f:
        maze_params = pickle.load(f)

    if (use_saved_emitter_0):
        with open(outdir / "emitter0.sav", "rb") as f:
            emitter = pickle.load(f)
        df = ArchiveDataFrame(emitter.archive.data(return_type="pandas"))
    else:
        df = ArchiveDataFrame(pd.read_csv(outdir / "archive.csv"))

    df_top = ArchiveDataFrame(df.sort_values(by="objective", ascending=False).head(num_to_sim))

    video_folder = outdir / "videos"
    video_folder.mkdir(parents=True, exist_ok=True) # will overwrite existing videos!

    # iterate through the top X elites, simulate them, and log stats
    for idx, elite in enumerate(df_top.iterelites()):
        model = elite["solution"]
        archive_idx = df_top.index[idx]

        reward, avg_torso_height, total_control_cost, _, _, _, _ = simulate(model, maze_params, xml_file, controller_params=controller_params, seed=env_seed, record_video=True, record_video_idx=idx, record_outdir=outdir)
        log.info(
            "=== Index {} ===\n"
            "Model:\n"
            "{}\n"
            "Reward: {}\n"
            "Original Reward: {}\n"
            "Final meas: {} / {} \n"
            "Original meas: {}\n",
            archive_idx,
            model,
            reward,
            elite["objective"],
            avg_torso_height, total_control_cost,
            elite["measures"]
        )


# NOTE: this is very very slow!
# NOTE: this updates scheduler.result_archive by adding the simulated predicted elites in! Do not call until after you've exported/saved result_archive!
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
    sols = scheduler.emitters[0].get_predicted_elites(scheduler.result_archive.boundaries, outdir)
    log.info("Finished Predictive Map. Simulating...")

    results = [simulate(model, maze_params, xml_file, seed=env_seed) for model in sols]
    
    objs = []
    meas = []
    chars = []

    for obj, avg_torso_height, total_control_cost, final_x_pos, final_y_pos, avg_lin_vel, min_rangefinder in results:
        objs.append(obj)
        meas.append([avg_torso_height, total_control_cost])
        chars.append([final_x_pos, final_y_pos, avg_lin_vel, min_rangefinder])

    scheduler.result_archive.add(sols, objs, meas, chars=chars)

    # save, graphs, etc.
    scheduler.result_archive.data(return_type="pandas").to_csv(outdir / "obs_and_predicted_archive.csv")
    save_ccdf(scheduler.result_archive, outdir / "obs_and_predicted_archive_ccdf.png")
    save_heatmap(scheduler.result_archive, outdir / "obs_and_predicted_archive_heatmap.png", maze_params["min_obj"], scheduler.result_archive.obj_max)
    log.info(
                "=== Predicted Archive Stats ===\n"
                "QD-Score: {}\n"
                "Max Reward: {}\n"
                "Coverage: {}\n",
                scheduler.result_archive.stats.qd_score,
                scheduler.result_archive.stats.obj_max,
                scheduler.result_archive.stats.coverage,
            )

if __name__ == "__main__":
    fire.Fire(mujoco_main)
