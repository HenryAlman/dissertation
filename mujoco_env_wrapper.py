import gymnasium as gym
import numpy as np
import math
import mujoco

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
        
        self.prior_x = [0,0,0,0,0]
        self.prior_y = [0,0,0,0,0]
        self.standing_still_counter = 0

        self.start_x = 0
        self.start_y = 0

        self.prev_vels = []


    # reset in case of bleed-over - shouldn't be possible but better safe than sorry!
    def reset(self, seed, options):
        self.current_step = 0
        self.standing_still_counter = 0
        self.prior_x = [0,0,0]
        self.prior_y = [0,0,0]
        self.start_x = 0
        self.start_y = 0
        return self.env.reset(seed=seed, options=options)


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # note we entirely ignore the default reward as it's defined relative to the goal

        mujoco_data = self.env.unwrapped.data
        mujoco_model = self.env.unwrapped.model

        # don't end on default condition of maze goal reached!
        if (terminated):
            terminated = False

        # measures
        x_pos = mujoco_data.qpos[0]
        y_pos = mujoco_data.qpos[1]

        if (self.current_step == 0):
            self.prior_x = [x_pos, x_pos, x_pos]
            self.prior_y = [y_pos, y_pos, y_pos]
            for idx, geom_id in enumerate(self.robot_ids):
                body_id = mujoco_model.geom_bodyid[geom_id]
                self.prev_vels.append(np.copy(mujoco_data.cvel[body_id]))

        dist_since_checkpoint_0 = math.sqrt((x_pos - self.prior_x[0])**2 + (y_pos - self.prior_y[0])**2)
        dist_since_checkpoint_1 = math.sqrt((x_pos - self.prior_x[1])**2 + (y_pos - self.prior_y[1])**2)
        dist_since_checkpoint_2 = math.sqrt((x_pos - self.prior_x[2])**2 + (y_pos - self.prior_y[2])**2)
        dist_since_checkpoints = [dist_since_checkpoint_0, dist_since_checkpoint_1, dist_since_checkpoint_2]

        # only count path distance after approximately travelling ~1 cell length (4m) from prior checkpoints
        # so we don't reward "bouncing" back and forth between cells or off walls, we want continuous navigation,
        # or end up giving distance rewards for "bobbing/weaving" during gait stride
        # so we ensure it's a new position relative to the last three checkpoints
        # more than 3 checkpoints means e.g. successfully turning around in a dead end etc. is quite harshly "punished"
        if (dist_since_checkpoint_0 > 4
            and dist_since_checkpoint_1 > 4
            and dist_since_checkpoint_2 > 4):
            # update with new checkpoint based on whichever is largest, i.e. now furthest away
            check_idx = np.argmax(dist_since_checkpoints)
            self.prior_x[check_idx] = x_pos
            self.prior_y[check_idx] = y_pos

            # but reward based on the minimum distance between new pos and checkpoint pos
            # in most scenarios this is "correct", there are some cases (e.g. u-turn in dead-end)
            # where it will slightly undercount a "fair" representation of the travel distance
            reward_idx = np.argmin(dist_since_checkpoints)
            info["path_dist"] = dist_since_checkpoints[reward_idx]
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

        # detect collisions for leg damage
        # we use both force and velocity changes to try to detect this - it's not perfect but reasonable on an "eye-test"
        broken_geoms=[]
        for idx, geom_id in enumerate(self.robot_ids):
            body_id = mujoco_model.geom_bodyid[geom_id]
            cur_vel = np.copy(mujoco_data.cvel[body_id])
            prev_vel = self.prev_vels[idx]
            vel_diff = cur_vel[3:6] - prev_vel[3:6]
            impact = np.linalg.norm(vel_diff)
            if (impact > 22):
                #print(mujoco_model.geom(geom_id).name, "broken! Impact = ", impact)
                broken_geoms.append(mujoco_model.geom(geom_id).name)
            self.prev_vels[idx] = cur_vel
            
        for i in range(mujoco_data.ncon):
            contact = mujoco_data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            if ((geom1 in self.robot_ids and geom2 in self.wall_ids) or
                    (geom1 in self.wall_ids and geom2 in self.robot_ids)):
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(mujoco_model, mujoco_data, i, force)
                normal_force = force[0]
                tangent_forces = force[1:3]
                total_force = sum([normal_force, sum(tangent_forces)])

                #print("collision between geom1: ", mujoco_model.geom(geom1).name, " and geom2: ", mujoco_model.geom(geom2).name, "with normal force: ", normal_force, "and tangential forces: ", tangent_forces)
                if (total_force > 100):
                    #print(mujoco_model.geom(geom1).name, "broken! Total force = ", total_force)
                    broken_geoms.append(mujoco_model.geom(geom1).name)

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
        info["broken_geoms"] = broken_geoms
        
        return obs, reward, terminated, truncated, info