import gymnasium as gym
import numpy as np
import math

class MujocoEnvWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            max_episode_steps: int,
    ):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        mj_model = env.unwrapped.model
        self.torso_id = mj_model.body("torso").id # note car xml had chassis name changed to "torso" as well

        self.prior_x = 0
        self.prior_y = 0
        self.standing_still_counter = 0



    def reset(self, seed, options):
        self.prior_x = 0
        self.prior_y = 0
        self.current_step = 0
        self.standing_still_counter = 0
        return self.env.reset(seed=seed, options=options)


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = False
        mujoco_data = self.env.unwrapped.data

        # default reward is e^-(distance^2). This transforms it into just -distance.
        # note the * 4.0 turns it into a metre distance based on default 4.0 maze_scaling for AntMaze
        # rather than a "cell distance"
        # note we also add 2.
        # Some of the models like hexapod are quite large, the qpos tracks its central position only
        # so this effectively means that if the *robot center gets within cell width of the goal* its a success with a score of 0
        reward = -math.sqrt(-math.log(reward)) * 4.0 + 2.0

        # measures
        linear_x_velocity = mujoco_data.qvel[0]
        linear_y_velocity = mujoco_data.qvel[1]

        # additional field for final position to heatmap it later
        x_pos = mujoco_data.qpos[0]
        y_pos = mujoco_data.qpos[1]
        
        # if within 2m (1/2 of a cell), count goal as reached
        if (reward >= 0.0): 
            success = True # track that goal was reached and return it
            terminated = True # terminate if goal reached

        # terminate if the torso falls over
        upright_alignment = mujoco_data.xmat[self.torso_id][8]
        if upright_alignment < 0.2:
            terminated = True

        # track movement; terminate if stalled
        current_x = mujoco_data.qpos[0]
        current_y = mujoco_data.qpos[1]
        if (self.current_step == 0):
            self.prior_x = current_x
            self.prior_y = current_y
        elif (math.sqrt((current_x - self.prior_x)**2 + (current_y - self.prior_y)**2) < 0.03):
            self.standing_still_counter += 1
            if (self.standing_still_counter > 50):
                terminated = True
        else:
            self.standing_still_counter = 0
            # note - only update the "priors" once there's been some movement. 
            # Effectively acting as a "waypoint" or "landmark" we occasionally update.
            # Otherwise we would be effectively demanding a constant *rate* of change (at least so much *per timestep*) rather than just trying to identify that it's stalled.
            self.prior_x = current_x
            self.prior_y = current_y

        self.current_step += 1

        info["success"] = success
        info["linear_x_velocity"] = linear_x_velocity
        info["linear_y_velocity"] = linear_y_velocity
        info["x_pos"] = x_pos
        info["y_pos"] = y_pos
        
        return obs, reward, terminated, truncated, info