from diss_script_refac import simulate
import numpy as np
from gymnasium_robotics.envs.maze.maps import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE

model = np.asarray([ 0.0680719788898192,  0.117828872521685,   0.4720476297957744,
 -0.2232343007427087,  0.1335773609590335,  0.2713327989102389,
  0.0113647739464679,  0.4997000485572417, -0.2265914755425285,
  0.1163369841507073, -0.4267134242464514,  0.0557521340566743,
 -0.3635117875188146, -0.2595310806955755, -0.0941735247046031,
  0.2750145736754608, -0.48225731941278,    0.1978890398566982,
  0.1147691364663208, -0.4868341309035991,  0.2584122314622381,
  0.8385854467153564, -0.3407948637049618,  1.0267101741390523,
  0.1612546962303887])

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

xml_file = "/home/henry/dissertation_5thAug/rangefinder_ant.xml"

env_seed = 52

r1 = simulate(model, maze_params, xml_file, seed=env_seed)
r2 = simulate(model, maze_params, xml_file, seed=env_seed)
print(r1)
print(r2)

list = [1,2,3,4,5]
print(list[1:3])