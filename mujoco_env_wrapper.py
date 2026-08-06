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

        self.start_x = 0
        self.start_y = 0


    # reset in case of bleed-over - shouldn't be possible but better safe than sorry!
    def reset(self, seed, options):
        self.prior_x = 0
        self.prior_y = 0
        self.current_step = 0
        self.standing_still_counter = 0
        return self.env.reset(seed=seed, options=options)


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # note we entirely ignore the default reward as it's defined relative to the goal

        mujoco_data = self.env.unwrapped.data

        # don't end on default condition of maze goal reached!
        if (terminated):
            terminated = False

        # measures
        x_pos = mujoco_data.qpos[0]
        y_pos = mujoco_data.qpos[1]

        if (self.current_step == 0):
            self.prior_x = x_pos
            self.prior_y = y_pos

        dist_since_last_checkpoint = math.sqrt((x_pos - self.prior_x)**2 + (y_pos - self.prior_y)**2)

        # only count path distance after a metre of travel, otherwise "bobbing" of the ant in gait counts
        if (dist_since_last_checkpoint > 1):
            # update checkpoint
            self.prior_x = x_pos
            self.prior_y = y_pos
            # return the distance to add to path distance
            info["path_dist"] = dist_since_last_checkpoint
        else:
            info["path_dist"] = 0

        # control cost from original Gymnasium Ant
        # their default cost is 0.5, but this produced consistently "slothlike" Ants. 
        # We want to value distance a bit more, so we halve the control cost estimation.
        # (we equally could double the distance score in terms of value)
        control_cost = 0.5 * np.sum(np.square(action))

        # also return info for characteristics
        linear_x_velocity = mujoco_data.qvel[0]
        linear_y_velocity = mujoco_data.qvel[1]
        torso_height = mujoco_data.qpos[2]

        # we need to let ineffective controllers burn energy going nowhere without ending early.
        # hit on sim time due to no truncation, sadly
        """
        # terminate if the torso just falls over
        upright_alignment = mujoco_data.xmat[self.torso_id][8]
        if upright_alignment < 0.2:
            terminated = True
        
        # track movement; terminate if completely stalled
        if (self.current_step == 0):
            self.prior_x = x_pos
            self.prior_y = y_pos
        # note, this is a very "generous" stall detector; only triggered by < 3cm of distance over 50 time steps
        # it's meant to detect ants standing totally still (outputting 0s), rather than ones e.g. walking into a wall
        elif (math.sqrt((x_pos - self.prior_x)**2 + (y_pos - self.prior_y)**2) < 0.03):
            self.standing_still_counter += 1
            if (self.standing_still_counter > 50):
                terminated = True
        else:
            self.standing_still_counter = 0
            # note - only update the "priors" once there's been some movement. 
            # Effectively acting as a "waypoint" or "landmark" we occasionally update.
            # Otherwise we would be effectively demanding a constant *rate* of change (at least so much *per timestep*) rather than just trying to identify that it's stalled.
            self.prior_x = x_pos
            self.prior_y = y_pos
        """

        self.current_step += 1

        info["linear_x_velocity"] = linear_x_velocity
        info["linear_y_velocity"] = linear_y_velocity
        info["torso_height"] = torso_height
        info["x_pos"] = x_pos
        info["y_pos"] = y_pos
        info["control_cost"] = control_cost
        
        return obs, reward, terminated, truncated, info