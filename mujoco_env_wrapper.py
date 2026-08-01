import gymnasium as gym
import numpy as np
import math

class MujocoEnvWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            max_episode_steps: int, # used to scale reward on reaching goal based on number of steps remaining
            goal_reward_weight: int = 100, # reward for reaching the goal
            healthy_reward_weight: float = 0.1, # reward per timestep for staying alive
            ctrl_cost_weight: float = 0.005, # weight for penalising high torque, aggressive/chaotic movements
            standing_still_penalty_weight: float = 1.0, # penalty for standing still. Triggers each step if >10 steps without meaningful movement.
            collision_penalty_weight: float = 5.0, # penalty for colliding with terrain
            height_penalty_weight: float = 10.0, # penalty for falling/flipping over
            tilt_penalty_weight: float = 0.1, # penalty for tilting > ~45degrees to encourage smoother movement
            speed_reward_weight: float = 0.1, # weight for rewarding speedy finishers over slow finishers. Set equal to healthy_reward_weight to have no preference, i.e. to just cancel out the extra survival rewards for getting there slower.
    ):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.goal_reward_weight = goal_reward_weight
        self.healthy_reward_weight = healthy_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.standing_still_penalty_weight = standing_still_penalty_weight
        self.collision_penalty_weight = collision_penalty_weight
        self.height_penalty_weight = height_penalty_weight
        self.tilt_penalty_weight = tilt_penalty_weight
        self.speed_reward_weight = speed_reward_weight
        self.current_step = 0

        mj_model = env.unwrapped.model
        self.torso_id = mj_model.body("torso").id # note car xml had chassis name changed to "torso" as well

        self.prior_x = 0
        self.prior_y = 0
        self.standing_still_counter = 0

        """
        # get environment geom IDs for collision and upright detection
        # use sets to ensure only added once
        self.wall_ids = set()
        self.robot_ids = set()
        for i in range(mj_model.ngeom):
            geom_name = mj_model.geom(i).name
            if "block" in geom_name: # note: this is a little fragile, relies on Gymnasium Robotics' generated terrain using this term for walls
                self.wall_ids.add(i)
            elif "floor" not in geom_name: # note: this is a little fragile, relies on everything other than the floor/blocks being part of the robot (i.e. no other obstacles)
                self.robot_ids.add(i)
            #if (geom_name == "torso" or geom_name == "car_body"): # note: again fragile, relies on specific naming
                #self.torso_id = mj_model.geom(i).bodyid

        # tracking movement
        self.start_x = 0
        self.start_y = 0 
        self.prior_max_dist_x = 0
        self.prior_max_dist_y = 0
        self.prior_x = 0
        self.prior_y = 0
        self.standing_still_counter = 0

        # track collision geom ids to not count them many times over multiple time steps
        self.prior_collision_geom1 = "none"
        self.prior_collision_geom2 = "none"
        """



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
        angular_z_axis_velocity = mujoco_data.qvel[5]
        
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
        info["angular_z_axis_velocity"] = angular_z_axis_velocity
        
        return obs, reward, terminated, truncated, info

        """
        ### SURVIVAL REWARD
        # still alive - add small baseline healthy reward
        reward += self.healthy_reward_weight
        """

        """
        ### CONTROL COST PENALTY
        # penalise high-torque movements
        control_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        reward -= control_cost
        """

        """
        ### MOVEMENT REWARDS AND PENALTIES
        # reward movement and penalise standing still
        if(self.current_step == 1): 
            self.start_x = info["x_position"]
            self.prior_x = info["x_position"]
            self.start_y = info["y_position"]
            self.prior_y = info["y_position"]
        else:
            current_x = info["x_position"]
            current_y = info["y_position"]
            dist_x = abs(current_x - self.start_x)
            dist_y = abs(current_y - self.start_y)

            # reward achieving new higher distances from start position
            if ((dist_x - self.prior_max_dist_x) > 0.2):
                # a flat reward gives disproportionate rewards to robots which "inch" along just fast enough to trigger this
                # so we scale the reward by the total amount of distance from the start, 
                # such that further distances are rewarded more than close ones
                reward += dist_x
                self.prior_max_dist_x = dist_x
            if ((dist_y - self.prior_max_dist_y) > 0.2):
                reward += dist_y
                self.prior_max_dist_y = dist_y

            
            # penalise standing still; hopefully helps encourage behaviour other than just stopping when near a wall (e.g. learning to turn better)
            # by making standing still more expensive than the survival reward
            # TODO: however, this can crush performance of early iterations and destroy the fitness landscape if too high, but doesn't sufficiently penalise later if too low,
            # so taking out for now and just truncating if stalled to reduce survival rewards for unmoving robots
            if (math.sqrt((current_x - self.prior_x)**2 + (current_y - self.prior_y)**2) < 0.02):
                self.standing_still_counter += 1
                
                if (self.standing_still_counter > 10):
                    reward -= self.standing_still_penalty_weight
                
                
                if (self.standing_still_counter > 300):
                    terminated = True
            
                
            else:
                self.standing_still_counter = 0
                # note - only update the "priors" once there's been some movement. 
                # Effectively acting as a "waypoint" or "landmark" we occasionally update.
                # Otherwise we would be effectively demanding a constant *rate* of change (at least so much *per timestep*) rather than just trying to identify that it's stalled.
                self.prior_x = current_x
                self.prior_y = current_y
            """

        """
        ### STABILITY PENALTIES
        # penalise falling over entirely or leaping upwards too high (e.g. onto the barriers)
        torso_z = mujoco_data.qpos[2]
        if torso_z < 0.3 or torso_z > 1.5:
            reward -= self.height_penalty_weight
            terminated = True
        else:
            # penalise tilting > ~45 degrees
            upright_alignment = self.env.unwrapped.data.xmat[self.torso_id][8]
            if upright_alignment < 0.7:
                tilt_penalty = self.tilt_penalty_weight * (1.0 - upright_alignment)
                reward -= tilt_penalty
        """

        """
        ### COLLISION PENALTY
        # detect and penalise wall collisions
        for i in range(mujoco_data.ncon):
            contact = mujoco_data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if ((geom1 in self.robot_ids and geom2 in self.wall_ids) or
                    (geom1 in self.wall_ids and geom2 in self.robot_ids)):
                # count collisions once, not many times across multiple time steps
                if (geom1 != self.prior_collision_geom1 or geom2 != self.prior_collision_geom2):
                    #print("collision between geom1: ", mj_model.geom(geom1).name, " and geom2: ", mj_model.geom(geom2).name)
                    reward -= self.collision_penalty_weight
                    self.prior_collision_geom1 = geom1
                    self.prior_collision_geom2 = geom2
                break
        """

