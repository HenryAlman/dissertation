from mapelites_mujoco_test_cpg_3 import simulate
import numpy as np
from gymnasium_robotics.envs.maze.maps import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE

model = np.load("best_model.npy")

maze = MEDIUM_MAZE
max_episode_steps = 3000
archive_dims = [20, 20]
archive_ranges = [(0, 0.8), (0, 2.0)]
maze_options = { "goal_cell": np.array([6, 5], dtype=int),
                "reset_cell": np.array([6, 1], dtype=int)
                }
maze_max_dist = 28.0
min_obj = -28.0

maze_params = {
        "maze": maze,
        "max_episode_steps": max_episode_steps,
        "maze_options": maze_options,
        "maze_max_dist": maze_max_dist,
        "min_obj": min_obj
    }

xml_file = "/home/henry/PythonDissertation/rangefinder_ant.xml"

env_seed = 52

r1 = simulate(model, maze_params, xml_file, env_seed)
r2 = simulate(model, maze_params, xml_file, env_seed)
print(r1)
print(r2)