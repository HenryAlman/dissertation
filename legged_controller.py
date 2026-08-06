import numpy as np
import math

# Ant/Hexapod uses CPGRBF network with num_rbfs RBFs, equally spaced around unit circle
# the learned weight w_cpg determines the frequency with which the CPG rotates around the unit circle.
# each RBF then drives hips and ankles uniformly (based on its learned w_hip and w_ankle weight) - indirect encoding, see https://mathiasthor.github.io/assets/pdf/generic_neural_locomotion_control_framework.pdf
# we then hardcode phases into the legs via gait_phases (i.e. some legs are at state cpg_state, some are at cpg_state * rotation matrix i.e. opposite side of unit circle)
# this produces symmetric, patterned movement. For regularised/symmetric robots (as our Ant and Hexapod are), their research showed this can be highly effective.


# rbf_sigma defines the "range" of each RBF, i.e. how much of an effect it has when the cpg_state is a certain distance away.
# lowerer rbf_sigma means that range is smaller, i.e. it only drives actions when the cpg_state is close, creating sharper/higher-frequency actions

# sign_mask translates the "virtual torques" output by the model (+ve = forward/down, -ve = backward/up) to the actual torque directions based on the XML,
# as the XML is defined such that a +ve torque produces a forward/backward/down/up motion in different legs/joints depending on physical orientation!

# Sensors are incorporated in one of two sensor_mode modes, each with a different number of weights needed at the end.
## 1. "unilateral": hips and ankles are all each dampened/amplified the same amount by the front sensor and the left_sensor - right_sensor differential. 
## So, 4 weights: frontsensor+hip, frontsensor+ankle, diffsensor+hip, diffsensor+ankle.
## 2. "per_joint": each individual joint has its own weight for the front sensor and left_sensor - right_sensor differential. This is closest to the
## original implementation with shunting inhibition neurons as "behaviour submodules". So, num_joints * 2 weights.

class LeggedController():
    # note: defaults are for rangefinder_ant model with 10 RBFs, a wide sigma RBF, 
    # and in unilateral sensor mode, so expecting a model of 25 weights (w_cpg, 20*w_locomotion, 4*w_sensor)
    def __init__(
            self,
            model: np.ndarray,
            num_legs: int = 4,
            gait_phases: np.ndarray =  np.array([
                                            0.0, # frontright
                                            np.pi, # frontleft
                                            np.pi, # backright
                                            0.0 # backleft
                                        ]),
            sign_mask: np.ndarray = np.array([
                                            1.0, # frontright hip, +ve torque is forward
                                            1.0, # frontright ankle, +ve torque is down
                                            -1.0, # leftright hip, -ve torque is forward
                                            1.0, # leftright ankle, +ve torque is down
                                            1.0, # backright hip, +ve torque is forward
                                            -1.0, # backright ankle, -ve torque is down
                                            -1.0, # backleft hip, -ve torque is forward
                                            -1.0 # backleft hip, -ve torque is down
                                        ]),
            num_rbfs: int = 10,
            rbf_sigma: float = ((math.sqrt(5)-1)/2) * (0.75),
            sensor_mode: str = "unilateral",
        ) -> None:
        self.num_legs = num_legs
        self.gait_phases = gait_phases
        self.sign_mask = sign_mask
        self.num_rbfs = num_rbfs
        self.rbf_sigma = rbf_sigma

        if(sensor_mode not in ["unilateral", "per_joint"]):
                raise ValueError("Unrecognised sensor_mode. Only unilateral or per_joint supported.")
        self.sensor_mode = sensor_mode

        # initialise CPG start state, RBFs placed equilaterally around circle, phase gaits, and torque sign masks
        self.cpg_state = np.array([0.1, 0.0]) 

        # rbfs equally spaced around unit circle
        angles = np.linspace(0, 2 * np.pi, self.num_rbfs, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)
        self.rbf_centers = np.column_stack((xs, ys))

        # cpg_state rotation matrices for each leg, depending on gait_phases
        # a gait phase of pi means that leg is being driven by the RBF on the *opposite side* than the RBF driving a leg with phase 0.0
        self.rotation_matrices = np.array([
                [[np.cos(phi), -np.sin(phi)], 
                [np.sin(phi),  np.cos(phi)]] 
                for phi in self.gait_phases
        ])

        self.unpack_model(model)


    def unpack_model(self, model: np.ndarray):
        # unpack weights and set up sensor reflex map
        self.w_cpg = model[0]
        locomotion_end_idx = int(1 + (self.num_rbfs*2)) # w_cpg, w_locomotion
        w_locomotion = model[1:locomotion_end_idx]
        w_locomotion = w_locomotion.reshape(self.num_rbfs, 2)
        self.w_locomotion_hip = w_locomotion[:,0]
        self.w_locomotion_knee = w_locomotion[:,1]
        
        if(self.sensor_mode == "unilateral"):
            # note: assumes order of righthip/rightank/lefthip/leftank/righthip/rightank/lefthip/leftank... in XML/action space!
            # note: assumes sensordiff is LEFT - RIGHT differential
            self.w_hip_centresensor, self.w_ankle_centresensor, self.w_hip_sensordiff, self.w_ankle_sensordiff = model[-4:]
            self.sensor_reflex_map = np.empty((0,2))
            for i in range(int(self.num_legs/2)):
                    self.sensor_reflex_map = np.append(self.sensor_reflex_map, [
                            [self.w_hip_centresensor, -self.w_hip_sensordiff], # right hip
                            [self.w_ankle_centresensor, -self.w_ankle_sensordiff], # right ankle
                            [self.w_hip_centresensor, self.w_hip_sensordiff], # left hip
                            [self.w_ankle_centresensor, self.w_ankle_sensordiff], # left ankle
                    ], axis = 0)
            # explanation: right hip gets hip signal and inverts sign of sensordiff signal
            # (as sensor diff is calculated as left - right, with closer being higher, so if positive then left wall is closer
            # and we want to turn right, i.e. dampen right motion and exaggerate left motion).
            # same logic for the others.
        elif(self.sensor_mode == "per_joint"):
              print("TODO!") #TODO



    def get_action(self,  rangefinders):
        # SO(2) oscillator for discrete time steps, with weight w_cpg taking place
        # of phi parameter. Original paper set it to exactly .01*pi, here we learn it.
        x, y = self.cpg_state
        next_x = x * np.cos(self.w_cpg) - y * np.sin(self.w_cpg)
        next_y = x * np.sin(self.w_cpg) + y * np.cos(self.w_cpg)
        # note: original paper multiplies above by 1.01 and takes tanh, rather than dividing by norm.
        # The *purpose* is to keep things in a stable loop, which dividing by the norm does just as well
        # by holding it to the unit circle (around which our rbf centers are evenly spaced).
        # the alpha/tanh formulation from the paper allows for regaining cpg stability if it is disrupted
        # by injected e.g. sensor data, but we inject our sensor data directly to the torques for simplicity
        # so it isn't needed here.
        # note: 1e-8 there as safety buffer to avoid div by zero error if state is exactly 0
        self.cpg_state = np.array([next_x, next_y]) / (np.linalg.norm(self.cpg_state) + 1e-8)

        # for each leg, get rotated cpg state per its phase in gait_phases
        leg_gait_torques = []
        for i in range(len(self.gait_phases)):
            rotated_cpg_hip = np.dot(self.rotation_matrices[i], self.cpg_state)
            # calc RBF activations based on distance b/w them and rotated cpg
            distances_hip = np.linalg.norm(self.rbf_centers - rotated_cpg_hip, axis=1)
            rbf_activations_hip = np.exp(- (distances_hip**2) / (2 * (self.rbf_sigma**2)))
            hip_torque = np.dot(rbf_activations_hip, self.w_locomotion_hip)

            #force knee to lag 1/4 cycle
            #90 degree counterclockwise rotation is same as x = y, y = -x
            rotated_cpg_knee = np.array([rotated_cpg_hip[1], -rotated_cpg_hip[0]])
            distances_knee = np.linalg.norm(self.rbf_centers - rotated_cpg_knee, axis=1)
            rbf_activations_knee = np.exp(- (distances_knee**2) / (2 * (self.rbf_sigma**2)))
            knee_torque = np.dot(rbf_activations_knee, self.w_locomotion_knee)

            leg_gait_torques.extend([hip_torque, knee_torque])

        virtual_gait_torques = np.array(leg_gait_torques)

        if(self.sensor_mode == "unilateral"):
            # steering signal swings positive/negative, and size of signal changes, depending on whether 
            # left or right is closer (and by how much)
            left_right_differential = rangefinders[0] - rangefinders[2]
            # combine with front sensor, and map to reflex
            sensors = np.array([rangefinders[1], left_right_differential])
            sensor_reflex = np.dot(self.sensor_reflex_map, sensors)
            # shift upwards so ranges around 1.0 (i.e. don't change anything from normal cpg) 
            # rather than 0.0 (don't *do* anything b/c 0 * cpg = 0)
            # clip to 0.2 to 1.8 so we amplify/dampen by 80% max
            sensor_multipliers = np.clip(1.0 + sensor_reflex, 0.2, 1.8)

            virtual_torques = virtual_gait_torques * sensor_multipliers
            xml_torques = virtual_torques * self.sign_mask

            # pass through tanh activation function
            # note: ant/hex control ranges in XML are already -1 to 1 so can use these as is
            unique_actions = np.tanh(xml_torques)

            return unique_actions
        
        elif(self.sensor_mode == "per_joint"):
            print("TODO!") #TODO