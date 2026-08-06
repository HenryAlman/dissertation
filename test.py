from bay_hex import simulate
import numpy as np
from gymnasium_robotics.envs.maze.maps import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE
import math

model = np.asarray( [ 0.25,                 0.8,                 -0.8,
 -0.8,                  0.8,                  0.8,
  0.8,                  0.8,                 -0.26548861419458925,
 -0.8,                 -0.8,                  0.1195648020489729,
 -0.8,                 -0.8,                 -0.28437861565776446,
 -0.8,                 -0.8,                  0.8,
  0.8,                  0.8,                  0.8,
  1.5,                -1.5,                  1.5,
 -1.5                
])

maze = MEDIUM_MAZE
max_episode_steps = 3000
maze_options = { "goal_cell": np.array([6, 5], dtype=int),
                "reset_cell": np.array([6, 1], dtype=int)
                }
maze_max_dist = 28.0
min_obj = 0.0 #if switching to goal again, -28.0
archive_dims = [12, 25] #TODO: tune
archive_ranges = [(0.2, 0.8), (0, 500)] # TODO: tune

maze_params = {
        "maze": maze,
        "max_episode_steps": max_episode_steps,
        "maze_options": maze_options,
        "maze_max_dist": maze_max_dist,
        "min_obj": min_obj
    }

xml_file = "/home/henry/dissertation_5thAug/rangefinder_hex.xml"

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
num_rbfs=10
rbf_sigma=((math.sqrt(5)-1)/2) * (0.75)
sensor_mode="unilateral"
leg_geom_names = [  
                            "frontright_leg_geom", # frontright hip
                            "frontright_ankle_geom",  # frontright ankle
                            "frontleft_leg_geom", #frontleft hip
                            "frontleft_ankle_geom", # frontleft ankle
                            "backright_leg_geom", #backright hip
                            "backright_ankle_geom", # backright ankle
                            "backleft_leg_geom", # backleft hip
                            "backleft_ankle_geom", # backleft ankle,
                            "midright_leg_geom", #midright hip
                            "midright_ankle_geom", #midright ankle
                            "midleft_leg_geom", #midleft hip
                            "midleft_ankle_geom", #midleft ankle
                        ]
controller_params = {
        "num_legs": num_legs,
        "gait_phases": gait_phases,
        "sign_mask": sign_mask,
        "num_rbfs": num_rbfs,
        "rbf_sigma": rbf_sigma,
        "sensor_mode": sensor_mode,
        "leg_geom_names": leg_geom_names
    }

env_seed = 52

r1 = simulate(model, maze_params, xml_file, controller_params=controller_params, seed=env_seed)
print(r1)


