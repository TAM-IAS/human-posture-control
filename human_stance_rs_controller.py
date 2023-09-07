import mujoco
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from mushroom_rl.core import Serializable
from mushroom_rl.utils.mujoco import ObservationHelper
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from functools import partial
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def get_dim_obs_type(obs_type):
    if obs_type == ObservationType.BODY_POS:
        return 3
    elif obs_type == ObservationType.BODY_ROT:
        return 4
    elif obs_type == ObservationType.BODY_VEL:
        return 6
    elif obs_type == ObservationType.JOINT_POS:
        return 1
    elif obs_type == ObservationType.JOINT_VEL:
        return 1
    elif obs_type == ObservationType.SITE_POS:
        return 3
    elif obs_type == ObservationType.SITE_ROT:
        return 9
    else:
        raise ValueError("Unknown observation_type % s" % obs_type)


class ObservationMemory(Serializable):
    """
   This class implements the observation memory needed to model delayed observations
   """

    def __init__(self, initial_size, max_size):
        """
        Constructor.

        Args:
            initial_size (int): initial number of elements in the observation memory
                                --> this should depend on the maximum delay! initial_size = timestep * max_delay_steps;
            max_size (int): maximum number of elements that the observation memory
                            can contain --> this should be greater than the maximum number of delay_steps

        """
        self._initial_size = initial_size
        self._max_size = max_size
        self._full = False
        self._obs_mem = list()
        self._delayed_params_mem = list()  #

        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _idx='primitive!',
            _full='primitive!',
            _obs_mem='pickle!',
        )

    def add(self, obs):
        """
        Add elements to the observation memory.

        Args:
            obs (array): array of observations to add to the observation memory;
            n_steps_return (int, 1): number of steps to consider for computing n-step return;
            gamma (float, 1.): discount factor for n-step return.

        """
        if self.size_obs_mem == self._max_size:
            # remove oldest obs
            self._obs_mem.pop(0)
            self._obs_mem.append(obs)
            self._full = True
        else:
            self._obs_mem.append(obs)

    def add_delayed_parameters(self, delayed_params):  #
        """
        Add delayed parameters to the memory. These include: [head acc, jac head position, jac com, jac trunk orientation, inertia matrix].

        Args:
            jac (array): array of jacobians to add to the observation memory;
        """
        if self.size__delayed_params_mem == self._max_size:
            # remove oldest values
            self._delayed_params_mem.pop(0)
            self._delayed_params_mem.append(delayed_params)
        else:
            self._delayed_params_mem.append(delayed_params)

    def get(self, n_delay_steps):
        """
        Returns the provided the observation from n_delay_steps ago.
        Args:
            n_delay_steps (int): the delay steps.
        Returns:
            The requested observation delayed by n_delay_steps wrt. to current timestep.
        """
        # assert n_delay_steps > 0, f'{n_delay_steps}'
        assert self.size_obs_mem >= n_delay_steps
        return self._obs_mem[-1-n_delay_steps]

    def reset(self):
        """
        Reset the replay memory.

        """
        self._full = False
        self._obs_mem = list()
        self._delayed_params_mem = list()  #

    @property
    def initialized_obs_mem(self):
        """
        Returns:
            Whether the observation memory has reached the number of elements that
            allows it to be used.

        """
        return self.size_obs_mem >= self._initial_size

    @property
    def initialized_delayed_params_mem(self):
        """
        Returns:
            Whether the delayed params memory has reached the number of elements that
            allows it to be used.

        """
        return self.size__delayed_params_mem >= self._initial_size

    @property
    def size_obs_mem(self):
        """
        Returns:
            The number of elements contained in the observation memory.

        """
        return len(self._obs_mem)
    
    @property
    def size__delayed_params_mem(self):
        """
        Returns:
            The number of elements contained in the delayed params memory.

        """
        return len(self._delayed_params_mem)


class HumanStance(MuJoCo):
    """
    Mujoco simulation of Human Stance task.
    "A multi-joint model of quiet, upright stance accounts for the “uncontrolled manifold” structure of joint variance",
    Hendrik Reimann, Gregor Schöner (2017)

    """

    def __init__(self, gamma=0.99, horizon=1000, n_intermediate_steps=1, n_substeps=1, timestep=0.002,
                 n_proprioceptive_delay_steps=15, n_vestibular_delay_steps=60, observation_noise_std=0.0,
                 rs_controller=False, noisy=False, traj_params=None):
        """
        Constructor.

        """
        xml_path = (Path(__file__).resolve().parent / "models" / "human_stance_rs_model.xml").as_posix()
        action_spec = ["act/ankle_joint", "act/knee_joint", "act/hip_joint"]

        observation_spec = [("torso_rot", "torso", ObservationType.BODY_ROT),  # 4d quaternions
                            ("head_pos", "head", ObservationType.BODY_POS),  # 3d coordinates (x, y, z)
                            # ("head_vel", "head", ObservationType.BODY_VEL),
                            # 6d vector first angular velocity around x, y, z. Then linear velocity for x, y, z
                            ("hip_joint_pos", "hip_joint", ObservationType.JOINT_POS),  # 1d scalar
                            ("knee_joint_pos", "knee_joint", ObservationType.JOINT_POS),  # 1d scalar
                            ("ankle_joint_pos", "ankle_joint", ObservationType.JOINT_POS),  # 1d scalar
                            ("hip_joint_vel", "hip_joint", ObservationType.JOINT_VEL),  # 1d scalar
                            ("knee_joint_vel", "knee_joint", ObservationType.JOINT_VEL),  # 1d scalar
                            ("ankle_joint_vel", "ankle_joint", ObservationType.JOINT_VEL)]  # 1d scalar
        
        additional_data_spec = []
        collision_groups = []
        
        super().__init__(xml_path, action_spec, observation_spec, gamma, horizon, timestep, n_substeps, n_intermediate_steps,
                         additional_data_spec=additional_data_spec, collision_groups=collision_groups)
        
        self.use_rs_controller = rs_controller
        self.gear_values = self.get_gear_values(xml_path)
        self.torso_rot_obs_index = [1, 2, 3, 0]
        self.joint_angles_obs_index = [9, 8, 7]
        self.joint_velocities_obs_index = [12, 11, 10]
        # self.head_vel_obs_index = [10, 11, 12]


        """Reimann & Schöner Controller Parameters"""
        "Values from MATLAB code"
        self._n_proprioceptive_delay_steps = n_proprioceptive_delay_steps
        self._n_vestibular_delay_steps = n_vestibular_delay_steps
        self.timestep = timestep
        self.A = np.array([[10.94, 1.1, 0],
                           [0, 7.43, 1.2],
                           [0, 0.94, 9.1]])  #Muscle Setup Matrix
        self.tau_c = 0.015 #Calcium kinetics temporal constant aka tau_m
        self.alpha_E = 12 #Stretch reflex form parameter in rad^(-1)
        self.mu = 0.06 #Stretch reflex velocity gain, time constant in seconds
        self.rho = np.array([[0.0], [0.0], [0.0]]) #Co-contraction command aka muscle_cc
        self.rho_ref = 0.01 #muscle_cc_ref
        self.B = np.array([[25, 2.5137, 0],
                [0, 16.9790, 2.7422],
                [0, 2.1481, 20.7952]])
        sigma_m = 0.02      #Signal dependent motor noise
        sigma_n = 0.006     #Neural processing noise aka sigma_lambda_dot
        sigma_r = 0
        sigma_jp = 0.002    #Muscle spindle activation noise (position)
        sigma_jv = 0.002    #Muscle spindle activation noise strength (velocity)
        sigma_ja = 0.02     #Muscle spindle activation noise strength (acceleration)
        sigma_pp = 0        #Head position estimation noise strength (EO/EC) 
        sigma_pv = 0.015    #Head velocity estimation noise strength (EO/EC)
        sigma_pa = 0.015    #Head acceleration estimation noise strength (EO/EC)
        sigma_op = 0.02     #Trunk orientation estimation noise strength (EO/EC)
        sigma_ov = 0        #Trunk velocity estimation noise strength (EO/EC)
        sigma_oa = 0        #Trunk acceleration estimation noise strength (EO/EC)
        sigma_bp = 0        #Tracking position sensor noise
        sigma_bv = 0        #Tracking velocity sensor noise
        sigma_ba = 0        #Tracking acceleration sensor noise
        sigma_cp = 0        #CoM sensor noise for position
        sigma_cv = 0.02     #CoM sensor noise for velocity
        sigma_ca = 0.02     #CoM sensor noise for acceleration
        sigma_tp = 0        #Joint torque sensor noise
        sigma_cop = 0       #Center of pressure sensor noise
        sigma_jacm = 0      #CoM jacobian representation noise
        sigma_jacv = 0      #Head jacobian representation noise
        sigma_jaci = 0      #Inertia representation noise

        # head position forcelet parameters
        self.alpha_pp = 0
        self.alpha_pv = 12  # rad-1s-2
        zeta = 0.5 #damping factor
        self.alpha_pddot = 2 * zeta * np.sqrt(self.alpha_pv)

        # head orientation forcelet parameters
        self.alpha_op = 40
        self.alpha_ov = 0
        self.alpha_oa = 0 

        self.use_bounded_activation = False
        self.E_max = 10        

        """Initial and Desired Pose"""
        # self.init_robot_pos = np.array([0.25, 1.3, -1.0])
        # self.init_robot_pos = np.array([-0.3375, -0.3325, 0.67])
        # self.init_robot_pos = np.array([-0.3273, -0.3086, 0.7359])
        self.init_robot_pos = np.array([-0.1,0.2,-0.2])
        self.init_torso_rot = np.sum(self.init_robot_pos)
        self.desired_torso_rot = -0.1

        """Reimann & Schöner Controller Variables"""
        self.d_activation_d_lambda = np.eye(3)  # "R" in Reimann & Schöner paper (eq. 22)
        self.d_agonistActivation_d_lambda = np.zeros((3, 3))
        self.d_antagonistActivation_d_lambda = np.zeros((3, 3))
        self.T_act = np.zeros((3, 1))
        self.T_act_dot = np.zeros((3, 1))
        self.T_act_two_dot = np.zeros((3, 1))
        self.head_vel = np.zeros((3, 1))
        self.head_pos = np.zeros((3,1))
        np.random.seed(2)
        
        """Reimann & Schöner Controller Noise Computation"""
        self.eta_lamda_dot = np.zeros((horizon, 3))
        self.eta_p_dot = np.zeros((horizon, 3))
        self.eta_p_ddot = np.zeros((horizon, 3))
        self.eta_o = np.zeros((horizon, 1))
        self.eta_theta = np.zeros((horizon, 3))
        self.eta_theta_dot = np.zeros((horizon, 3))
        self.eta_m = np.zeros((horizon, 3))
        if noisy:
            self.eta_lamda_dot = self.get_ornstein_uhlenbeck_noise(horizon, alpha_eta=5,
                                                                   sigma_star=np.array([sigma_n, sigma_n, sigma_n]))
            self.eta_p_dot = self.get_ornstein_uhlenbeck_noise(horizon, alpha_eta=5,
                                                               sigma_star=np.array([sigma_pv, 0.0, sigma_pv]))
            self.eta_p_ddot = self.get_ornstein_uhlenbeck_noise(horizon, alpha_eta=5,
                                                                sigma_star=np.array([sigma_pa, 0.0, sigma_pa]))
            self.eta_o = self.get_ornstein_uhlenbeck_noise(horizon, alpha_eta=5, sigma_star=np.array([sigma_op]))
            self.eta_theta = self.get_ornstein_uhlenbeck_noise(horizon, alpha_eta=5, sigma_star=np.array(
                [sigma_jp, sigma_jp, sigma_jp]))
            self.eta_theta_dot = self.get_ornstein_uhlenbeck_noise(horizon, alpha_eta=5, sigma_star=np.array(
                [sigma_jv, sigma_jv, sigma_jv]))
            self.eta_m = self.get_ornstein_uhlenbeck_noise(horizon, alpha_eta=5,
                                                           sigma_star=np.array([sigma_m, sigma_m, sigma_m]))

        """Observation Memory"""
        # we use an observation memory to model delayed observations
        greatest_delay = max(self._n_proprioceptive_delay_steps, self._n_vestibular_delay_steps)
        self.current_step = 0
        self._obs_mem = ObservationMemory(initial_size=greatest_delay+1, max_size=10 * greatest_delay+1)

        # we want to store the first observation
        self._first_obs = None
        self._first_delayed_params = None

        # to know where to apply which delay, we want a mask to distinguish between proprioceptive and vestibular senses
        self._obs_vestibular_mask = np.concatenate([np.ones(7), np.zeros(6)]).astype(bool)
        self.obs_prioception_mask = ~self._obs_vestibular_mask

        # std for normal noise added to observation
        self._observation_noise_std = observation_noise_std

        """Plotter Fields"""
        self.data = {"Joint Angles": [], "Joint Velocities": [], "Head Position": [], "Head Velocity": [], "Head Acceleration": [],
                     "Torso RotEuler": [], "Head Forcelet Position": [], "Head Forcelet Orientation": [], "Lambda Dot": [], "Lambda": [], "E": [], "T": [], "T_act":[], "T_ela":[], "T_vis":[], "T_gravity": [], "True Joint Angles": [], "True Joint Velocities": [], "Joint Accelerations": []}

    def setup(self, obs=None):
        self._data.qpos = deepcopy(self.init_robot_pos)
        self._data.qvel = np.zeros_like(self.init_robot_pos)
        initial_torso_quat = Rotation.from_euler("y", self.init_torso_rot, degrees=False).as_quat()
        self._data.body("torso").xquat =  np.array([initial_torso_quat[3], initial_torso_quat[0], initial_torso_quat[1], initial_torso_quat[2]])
        # print(self._data.body("torso").xquat)
        "Run forward kinematics pass to set initial body position"
        mujoco.mj_fwdPosition(self._model, self._data)
        mujoco.mj_fwdVelocity(self._model, self._data)
        self.head_pos = deepcopy(self._data.site_xpos)
        self._first_obs = deepcopy(self.obs_helper._build_obs(self._data))
        self._first_delayed_params = []
        self._first_delayed_params.append(self.get_head_vel())
        self._first_delayed_params.append(self.get_head_acc())
        self._first_delayed_params.append(self.get_jac_head_position())
        self._first_delayed_params.append(self.get_jac_com())
        self._first_delayed_params.append(self.get_jac_trunk_orientation())
        self._first_delayed_params.append(self.get_inertia_matrix())
        self._obs_mem.reset()
        self._obs_mem.add(self._first_obs)
        self._obs_mem.add_delayed_parameters(self._first_delayed_params)

        if self.use_rs_controller:
            # Value from R&S Matlab code
            # self.lamda = deepcopy(np.array([-0.026698027271998, 0.106314335775726, -0.124050563479848]).reshape(-1,1))
            self.lamda = self.get_init_lamda()
            self.set_init_torques()

    def play_trajectory_demo(self, trajectory):
        """
        Plays a demo of the trajectory by forcing the model
        *positions* to the ones in the reference trajectory at every step

        """

        for sample in tqdm(trajectory, desc="Samples Played"):
            self.set_qpos_qvel(sample)

            mujoco.mj_forward(self._model, self._data)

            self.render()
    
    def play_trajectory_demo_from_velocity(self, trajectory):
        """
        Plays a demo of the loaded trajectory by using numerical integration using only the velocities of the data,
        calculating therewith the positions and setting the model positions to the calculated ones at every steps.
        """

        obs_spec = deepcopy(self.obs_helper.observation_spec)

        joint_pos_mask, _ = self.get_obs_mask(obs_spec, ObservationType.JOINT_POS)
        joint_vel_mask, _ = self.get_obs_mask(obs_spec, ObservationType.JOINT_VEL)

        # get the first joint position
        sample = trajectory[0]
        self.set_qpos_qvel(sample)
        curr_qpos = sample[joint_pos_mask]

        for sample in tqdm(trajectory[1:], desc="Samples Played"):
            # use velocity and integration to calculate the new pos
            qvel = sample[joint_vel_mask]
            print(qvel)
            curr_qpos = curr_qpos + self.dt * qvel
            sample[joint_pos_mask] = curr_qpos

            self.set_qpos_qvel(sample)

            mujoco.mj_forward(self._model, self._data)

            self.render()

    def set_qpos_qvel(self, sample):
        """
        Takes a sample including joint positions and joint velocities in the order of observation spec and sets
        them in the model.

        """

        obs_spec = deepcopy(self.obs_helper.observation_spec)

        # get the relevant masks
        joint_pos_mask, rel_obs_spec_pos = self.get_obs_mask(obs_spec, ObservationType.JOINT_POS)
        joint_vel_mask, rel_vel_spec_vel = self.get_obs_mask(obs_spec, ObservationType.JOINT_VEL)
        joint_pos_vel_mask = joint_pos_mask + joint_vel_mask

        rel_obs_spec = rel_obs_spec_pos + rel_vel_spec_vel
        sample = sample[joint_pos_vel_mask]

        # set the joint position and joint velocities
        for key_name_ot, value in zip(rel_obs_spec, sample):
            key, name, ot = key_name_ot
            if ot == ObservationType.JOINT_POS:
                self._data.joint(name).qpos = value
            elif ot == ObservationType.JOINT_VEL:
                self._data.joint(name).qvel = value

    def _simulation_post_step(self):
        """ We store all observation (using the simulators frequency!) to compute delayed observations at our
            control frequency """
        # get current observation from simulation
        obs = self.obs_helper._build_obs(self._data)
        head_vel = self.get_head_vel()
        head_acc = self.get_head_acc()
        self.head_vel = head_vel
        self.head_pos = deepcopy(self._data.site_xpos)
        jac_head = self.get_jac_head_position()
        jac_com = self.get_jac_com() 
        jac_trunk = self.get_jac_trunk_orientation()
        inertia_matrix = self.get_inertia_matrix()
        # add it to the observation memory
        self._obs_mem.add(obs)
        self._obs_mem.add_delayed_parameters([head_vel, head_acc, jac_head, jac_com, jac_trunk, inertia_matrix])

    def get_obs_mask(self, obs_spec, requested_obs_type):
        """
        Returns a mask given the requested observation type and a list of the filtered observation specification.
        """
        rel_mask = []
        rel_obs_spec = []
        for os in obs_spec:
            obs_type = os[2]
            if obs_type == requested_obs_type:
                rel_mask.append(np.ones(get_dim_obs_type(obs_type), dtype=bool))
                rel_obs_spec.append(os)
            else:
                rel_mask.append(np.zeros(get_dim_obs_type(obs_type), dtype=bool))
        return np.concatenate(rel_mask), rel_obs_spec

    def _create_observation(self, obs):
        """ Instead of using the current observation, we use a delayed one from the observation memory. This function
            is called at our control frequency. """
        # if the observation memory is initialized, we can used the delayed observation, else we use the first one
        if self._obs_mem.initialized_obs_mem:
            # get obs at reflex delay
            obs_vestibular = self._obs_mem.get(self._n_vestibular_delay_steps)[self._obs_vestibular_mask]
            # get obs at  transcortical delay
            obs_proprioceptive = self._obs_mem.get(self._n_proprioceptive_delay_steps)[self.obs_prioception_mask]
            # concatenate both
        elif self._obs_mem.size_obs_mem > self._n_vestibular_delay_steps and self._obs_mem.size_obs_mem <= self._n_proprioceptive_delay_steps:
            obs_vestibular = self._obs_mem.get(self._n_vestibular_delay_steps)[self._obs_vestibular_mask]
            obs_proprioceptive = self._first_obs[self.obs_prioception_mask]
        elif self._obs_mem.size_obs_mem <= self._n_vestibular_delay_steps and self._obs_mem.size_obs_mem > self._n_proprioceptive_delay_steps:
            obs_vestibular = self._first_obs[self._obs_vestibular_mask]
            obs_proprioceptive = self._obs_mem.get(self._n_proprioceptive_delay_steps)[self.obs_prioception_mask]
        else:
            obs_vestibular = self._first_obs[self._obs_vestibular_mask]
            obs_proprioceptive = self._first_obs[self.obs_prioception_mask]
        obs = np.concatenate([obs_vestibular, obs_proprioceptive])

        # add noise to observation
        if self.use_rs_controller:
            "Equation 2 if we're using Reimann and Schoener controller"
            # add torso rotation noise, torso rotation given by MuJoCo is in [rw,rx,ry,rz] format
            torso_rot_noise = Rotation.from_euler('y', self.eta_o[self.current_step, :], degrees=False)
            torso_rot_original = Rotation.from_quat(obs[self.torso_rot_obs_index])
            torso_rot_noisy = torso_rot_noise * torso_rot_original
            torso_rot_noisy_quat = torso_rot_noisy.as_quat()
            obs[0:4] = [torso_rot_noisy_quat[0,3], torso_rot_noisy_quat[0,0], torso_rot_noisy_quat[0,1], torso_rot_noisy_quat[0,2]]
            # add rest of the noises
            noise = np.concatenate((np.zeros((1, 3)),  # head position is not sensed by the human, therefore no noise
                                    self.eta_theta[self.current_step:self.current_step + 1, :],
                                    self.eta_theta_dot[self.current_step:self.current_step + 1, :]), axis=1)
            noise = np.squeeze(noise)
            obs[4:] += noise
        else:
            noise = np.random.normal(0.0, self._observation_noise_std, len(obs))
            obs += noise
        # print("Obervations: ", obs)
        return obs

    def get_delayed_params(self, n_delay_steps):
        """ Instead of using the current jacobians and inertia matrix, we use the delayed ones from the jac memory. This function
        is called from the _compute_action function. """
        delayed_params = {}
        if self._obs_mem.size__delayed_params_mem > self._n_vestibular_delay_steps:
            delayed_params["Head Velocity"] = self._obs_mem._delayed_params_mem[-1-n_delay_steps][0]
            delayed_params["Head Acceleration"] = self._obs_mem._delayed_params_mem[-1-n_delay_steps][1]
            delayed_params["Jac Head Position"] = self._obs_mem._delayed_params_mem[-1-n_delay_steps][2]
            delayed_params["Jac COM"] = self._obs_mem._delayed_params_mem[-1-n_delay_steps][3]
            delayed_params["Jac Trunk Orientation"] = self._obs_mem._delayed_params_mem[-1-n_delay_steps][4]
            delayed_params["Inertia Matrix"] = self._obs_mem._delayed_params_mem[-1-n_delay_steps][5]
        else:
            delayed_params["Head Velocity"] = self._first_delayed_params[0]
            delayed_params["Head Acceleration"] = self._first_delayed_params[1]
            delayed_params["Jac Head Position"] = self._first_delayed_params[2]
            delayed_params["Jac COM"] = self._first_delayed_params[3]
            delayed_params["Jac Trunk Orientation"] = self._first_delayed_params[4]
            delayed_params["Inertia Matrix"] = self._first_delayed_params[5]
        return delayed_params

    def is_absorbing(self, obs):
        """Stop simulation if head is close to the ground"""
        head_vector = self._data.body("head").xpos
        if np.linalg.norm(head_vector[2] - 0.0) < 0.1:
            return True
        else:
            return False
    
    def reward(self, state, action, next_state, absorbing):
        """ This function is not needed for R&S controller simulation"""
        return 1.

    # def get_head_acc(self):
    #     """Get acceleration of head"""
    #     head_id = self._model.body("head").id  # ID of the root body
    #     acc_array = np.zeros(6)
    #     mujoco.mj_objectAcceleration(self._model, self._data, mujoco.mjtObj.mjOBJ_BODY, head_id, acc_array, 0)
    #     # print("Head acceleration=", acc_array)
    #     return acc_array[3:6].reshape(-1, 1)

    def get_head_acc(self):
        head_acc = (self.get_head_vel() - self.head_vel)/self.timestep
        return head_acc

    def get_head_vel(self):
        """Get velocity of head"""
        head_id = self._model.body("head").id  # ID of the root body
        vel_array = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data, mujoco.mjtObj.mjOBJ_BODY, head_id, vel_array, 0)
        # print("Head velocity=", vel_array[3:6])
        return vel_array[3:6].reshape(-1, 1)

    # def get_head_vel(self):
    #     head_vel = (self._data.site_xpos - self.head_pos)/self.timestep
    #     # print("Head velocity=", head_vel)

    #     return head_vel.reshape(-1,1)

    def get_jac_com(self):
        """Get jacobian of center of mass from MuJoCo"""
        root_id = 0 # ID of the root of the whole body
        nv = self._model.nv
        jac_com = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(self._model, self._data, jac_com, root_id)
        # print("Jacobian COM full=", jac_com)
        return jac_com

    def get_jac_head_position(self):
        """Get jacobian of head from MuJoCo"""
        head_id = self._model.body("head").id  # ID of the root body
        nv = self._model.nv
        jac_head = np.zeros((3, nv))
        head_com = self._data.geom_xpos[head_id]
        mujoco.mj_jac(self._model, self._data, jac_head, None, head_com, head_id)
        # print(jac_head)
        return jac_head

    def get_jac_trunk_orientation(self):
        """Get jacobian of trunk orientation from MuJoCo"""
        trunk_id = self._model.body("torso").id  # ID of the root body
        nv = self._model.nv
        jac_trunk_rot = np.zeros((3, nv))
        mujoco.mj_jacBody(self._model, self._data, None, jac_trunk_rot, trunk_id)
        # print("Jacobian trunk full=", jac_trunk_rot)
        return jac_trunk_rot

    def get_inertia_matrix(self):
        """Get inertial matrix from MuJoCo"""
        nv = self._model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self._model, M, self._data.qM)
        # print(M.shape)
        return M

    def get_gear_values(self, xml_path):
        """Get gear values for motors from xml"""
        xml_model = ET.parse(xml_path)
        motors = xml_model.getroot().find("actuator").findall("motor")
        gear_values = [float(motor.get("gear")) for motor in motors]
        return gear_values

    def get_init_lamda(self):
        """Get initial values of lamda Equation 26"""
        joint_angles = deepcopy(self._data.qpos).reshape(-1, 1)
        joint_angles_passive = deepcopy(joint_angles)
        joint_velocities = np.zeros((3, 1))
        M = self.get_inertia_matrix()
        gravitational_torques = self._data.qfrc_bias.reshape(-1, 1)
        passive_elastic_torques = self.get_passive_elastic_torque(joint_angles_passive)
        desired_active_torques = gravitational_torques - passive_elastic_torques
        desired_activation = np.matmul(np.linalg.inv(self.A), desired_active_torques)

        def obj_fun(lamda):
            [_, E_AG, _, E_AN] = self.get_activations(joint_angles=joint_angles, joint_velocities=joint_velocities,
                                                      lamda=lamda, lambda_dot=np.zeros((3, 1)), rho=np.zeros((3, 1)),
                                                      mu=0)
            return np.linalg.norm(desired_activation + E_AG - E_AN)

        init_lambda = minimize(obj_fun, np.squeeze(joint_angles), tol=1e-6, method='Nelder-Mead').x
        # print("Init lambda = ", init_lambda)

        # init_lambda = np.zeros((3,1))
        # init_lambda = np.array([[-0.0267], [0.1063], [-0.1241]])

        """Check"""
        [_, computed_E_AG, _, computed_E_AN] = self.get_activations(joint_angles, joint_velocities, init_lambda,
                                                                    lambda_dot=np.zeros((3, 1)), rho=np.zeros((3, 1)),
                                                                    mu=0)
        computed_activations = -computed_E_AG + computed_E_AN
        computed_torques = np.matmul(self.A, computed_activations)
        self.T_act = computed_torques
        difference = np.linalg.norm(computed_torques + passive_elastic_torques - gravitational_torques)
        assert difference < 1e-4, f'{difference}'
        return init_lambda.reshape(-1, 1)

    def set_init_torques(self):
        joint_angles = deepcopy(self._data.qpos).reshape(-1, 1)
        joint_velocities = np.zeros((3, 1))
        init_lambda = deepcopy(self.lamda)
        [_, computed_E_AG, _, computed_E_AN] = self.get_activations(joint_angles, joint_velocities, init_lambda,
                                                                    lambda_dot=np.zeros((3, 1)), rho=np.zeros((3, 1)),
                                                                    mu=0)
        computed_activations = -computed_E_AG + computed_E_AN
        computed_torques = np.matmul(self.A, computed_activations)
        self.T_act = computed_torques
        
    def get_ornstein_uhlenbeck_noise(self, n, alpha_eta, sigma_star):
        """Method to get Ornstein Uhlenbeck Noise using Equation 24"""
        dim = sigma_star.shape[0]
        eta_t = np.zeros((n + 1, dim))
        for i in range(0, n):
            for j in range(0, dim):
                eta_dot_t = -alpha_eta * eta_t[i, j] + sigma_star[j] * np.random.randn() / (self.timestep ** 0.5)
                eta_t[i + 1, j] = eta_t[i, j] + eta_dot_t * self.timestep
        assert eta_t.shape[0] == n + 1
        return eta_t[1:, :]

    def _compute_action(self, obs, action):
        """ Method to compute joint torques"""
        if self.use_rs_controller:
            delayed_params = self.get_delayed_params(self._n_vestibular_delay_steps)
            modified_action = self.rs_controller(obs, delayed_params, self.alpha_pv, self.alpha_pddot, self.alpha_op, strategy=4)
            modified_action = [i / j for i, j in
                               zip(modified_action, self.gear_values)]  # To account for gear ratios in MuJoCo model
            # print("Joint torques", modified_action)
        else:
            modified_action = action

        return modified_action
    
    def rs_controller(self, obs, delayed_params, alpha_pdot, alpha_pddot, alpha_o, strategy):
        """
            Method to compute the joint torques from Reimann & Schoner 2017 controller

                strategy:
                   -1 - All feedback removed
                    0 - no outer loop i.e. lamda_dot = 0
                    1 - Ankle strategy with co-contraction at proximal joints (Section 2.2.2 A)
                    2 - Ankle strategy with local feedback control of proximal joints (Section 2.2.2 B)
                    3 - Ankle strategy with multi-joint coordination (Section 2.2.2 C)
                    4 - Distributed strategy with multi-joint coordination (Section 2.2.2 D)
        """
        joint_angles = deepcopy(obs[self.joint_angles_obs_index].reshape(-1, 1))
        # print("Joint Angles:", joint_angles.T)
        joint_velocities = deepcopy(obs[self.joint_velocities_obs_index].reshape(-1, 1))

        "Equation 13"
        head_acc =  delayed_params["Head Acceleration"] + np.transpose(self.eta_p_ddot[self.current_step:self.current_step + 1, :])
        head_vel = delayed_params["Head Velocity"] + np.transpose(self.eta_p_dot[self.current_step:self.current_step + 1, :]) #np.array(obs[self.head_vel_obs_index]).reshape(-1, 1)
        # print("Head Velocity", head_vel)
        assert head_vel.shape == (3, 1), f'{head_vel.shape}'
        # head_acc = (head_vel - self.head_vel)/self.timestep + np.transpose(self.eta_p_ddot[self.current_step:self.current_step+1,:])
        assert head_acc.shape == (3, 1), f'{head_acc.shape}'
        torso_rot_quaternions = obs[self.torso_rot_obs_index]
        rot = Rotation.from_quat(torso_rot_quaternions)
        rot_euler = rot.as_euler('xyz', degrees=False)
        "Equation 14"
        f_p = -alpha_pdot * head_vel[1] - alpha_pddot * head_acc[1]
        f_o = -alpha_o * (rot_euler[0]-self.desired_torso_rot)

        head_forcelet_position = np.zeros((3,1))
        head_forcelet_orientation = np.zeros((3,1))
        if strategy == -1:
            " Section 4.5, Figure 9 Right, Purely feedforward control, i.e., all feedback removed"
            joint_angles = self.init_robot_pos.reshape(-1,1)
            joint_velocities = np.zeros_like(joint_angles)
            lambda_dot = np.zeros((3, 1))
        elif strategy == 0:
            " Section 4.5, Figure 9 Left, Only the reflexive feedback left intact"
            lambda_dot = np.zeros((3, 1))
        elif strategy == 1:
            " Section 2.2.2 A : Ankle strategy with co-contraction at proximal joints"
            "TODO: Check"
            self.alpha_pv = 5
            self.alpha_pddot = 0.1
            self.alpha_o = 0
            lambda_dot = np.zeros((3, 1))
            lambda_dot[0] = f_p
            lambda_dot[1] = 0
            lambda_dot[2] = 0
            self.rho = np.array([[0.01], [0.15], [0.15]])
        elif strategy == 2:
            "Section 2.2.2 B : Ankle strategy with local feedback control of proximal joints"
            raise NotImplementedError
        elif strategy == 3 or strategy == 4:
            if strategy == 3:
                "Section 2.2.2 C: Ankle strategy with multi-joint coordination"
                "Equation 17"
                J_p = np.zeros((3, 3))
                J_p[1, 0] = delayed_params["Jac Head Position"][1, 0]
                J_p_penv = np.linalg.pinv(J_p)
                # F_p = np.matmul(J_p_penv, f_p)
            elif strategy == 4:
                "Section 2.2.2 D: Distributed strategy with multi-joint coordination"
                "Equation 23"
                J_p = np.zeros((3, 3))
                J_p[1, :] = delayed_params["Jac Head Position"][1, :]
                # F_p = np.matmul(J_p_penv, f_p)

            "Equation 18, 19"
            "Convert jac_com 3x3 to J_c 1x3 by using just the horizontal position"
            J_com = delayed_params["Jac COM"][1, :].reshape(-1, 1).T  # np.ones((1,3))
            # print("Jacobian COM reduced = ", J_com)
            "Convert jac_trunk_rot 3x3 to J_o 1x3 by using just angle wrt Y-axis"
            J_o = delayed_params["Jac Trunk Orientation"][0, :].reshape(-1, 1).T  # np.ones((1,3))
            # print("Jacobian trunk reduced = ", J_o)
            J_aug = np.concatenate([J_o, J_com])
            assert J_aug.shape == (2, 3), print(J_aug.shape)
            J_aug_penv = np.linalg.pinv(J_aug)
            # F_o = np.matmul(J_aug_penv, np.array([[f_o], [0]]))

            "Equation 20"
            # F = F_p + F_o

            "Equation 21"
            M = delayed_params["Inertia Matrix"]  # np.ones((3,3))
            # print("Inertia matrix", M)
            transformationMatrix_position = np.linalg.inv(self.d_activation_d_lambda) @ np.linalg.inv(self.A) @ M @ np.linalg.pinv(J_p)
            transformationMatrix_orientation = np.linalg.inv(self.d_activation_d_lambda) @ np.linalg.inv(self.A) @ M @ J_aug_penv

            head_forcelet_position = (transformationMatrix_position[:,1] * f_p).reshape(-1,1)
            head_forcelet_orientation = (transformationMatrix_orientation[:,0] * f_o).reshape(-1,1)
            neural_noise = np.transpose(self.eta_lamda_dot[self.current_step:self.current_step + 1, :])
            lambda_dot =  head_forcelet_position + head_forcelet_orientation + neural_noise
            # print("lambda_dot",lambda_dot)

        self.lamda = self.lamda + lambda_dot * self.timestep
        E = self.reflex_controller(joint_angles, joint_velocities, self.lamda, lambda_dot, self.rho, self.mu)
        [T_act, T_passive, T_viscous] = self.muscle_model(self._obs_mem._obs_mem[-1], E) #Very Important: Muscles do not get delayed observations
        
        "Equation 11 Total instantaneous torque"
        T = T_act + T_passive + T_viscous
        assert len(T) == len(joint_angles)
        assert T.shape[1] == 1
        torques = T.squeeze()

        # Data Logger
        self.data["Joint Angles"] = joint_angles
        self.data["True Joint Angles"] = self._data.qpos.reshape(-1,1)
        self.data["Joint Velocities"] = joint_velocities
        self.data["True Joint Velocities"] = self._data.qvel.reshape(-1,1)
        self.data["Joint Accelerations"] = self._data.qacc.reshape(-1,1)
        self.data["Head Position"] = self._data.site_xpos.reshape(-1,1)
        self.data["Head Velocity"] = head_vel
        self.data["Head Acceleration"] = head_acc
        self.data["Torso RotEuler"] = rot_euler.reshape(-1,1)
        self.data["Head Forcelet Position"] = head_forcelet_position
        self.data["Head Forcelet Orientation"] = head_forcelet_orientation
        self.data["Lambda Dot"] = lambda_dot
        self.data["Lambda"] = self.lamda
        self.data["E"] = E
        self.data["T"] = T
        self.data["T_act"] = T_act
        self.data["T_ela"] = T_passive
        self.data["T_vis"] = T_viscous
        self.data["T_gravity"] = self._data.qfrc_bias.reshape(-1, 1)
        
        return torques

    def get_activations(self, joint_angles, joint_velocities, lamda, lambda_dot, rho, mu):
        lamda = deepcopy(lamda).reshape(-1, 1)
        assert lamda.shape[1] == 1
        assert lambda_dot.shape[1] == 1
        # agonistActivation in stableLambda_EPH.m
        base_AG = self.alpha_E * (joint_angles - lamda + rho + mu * (joint_velocities - lambda_dot))
        E_AG = np.exp(np.maximum(np.zeros((3, 1)), base_AG)) - 1
        # anatagonistActivation in stableLambda_EPH.m
        base_AN = -self.alpha_E * (joint_angles - lamda - rho + mu * (joint_velocities - lambda_dot))
        E_AN = np.exp(np.maximum(np.zeros((3, 1)), base_AN)) - 1
        return [base_AG, E_AG, base_AN, E_AN]

    def reflex_controller(self, joint_angles, joint_velocities, lamda, lambda_dot, rho, mu):
        """
        Reimann & Schoner 2017 Equation 1
        """
        lamda = deepcopy(lamda).reshape(-1, 1)
        assert lamda.shape[1] == 1
        assert lambda_dot.shape[1] == 1
        assert joint_angles.shape[1] == 1
        assert joint_velocities.shape[1] == 1

        [base_AG, f_AG, base_AN, f_AN] = self.get_activations(joint_angles, joint_velocities, lamda, lambda_dot, rho, mu)
        E_AG = deepcopy(f_AG)
        E_AN = deepcopy(f_AN)

        "Equation 4"
        if self.use_bounded_activation:
            E_max = self.E_max
            E0 = 3 / 4 * E_max
            b0 = np.log(E0 + 1)
            c2 = -b0 + E_max / E0 - 1
            c3 = E_max
            c1 = (b0 + c2) * (E0 - c3)
            for joint_number in range(3):
                if f_AG[joint_number] > E0:
                    E_AG[joint_number] = c1/(E_AG[joint_number] + c2) + c3

                if f_AN[joint_number] > E0:
                    E_AN[joint_number] = c1/(E_AN[joint_number] + c2) + c3

        reflex_action_noise = np.transpose(self.eta_m[self.current_step:self.current_step + 1, :])  # PostureNoise.m line 148
        
        "Equation 3"
        E = (-E_AG + E_AN) * (1 + reflex_action_noise)
        assert len(E) == len(self.joint_angles_obs_index)

        "Equation 22"
        self.d_agonistActivation_d_lambda = (-self.alpha_E * (f_AG + 1)) * (base_AG > 0)
        self.d_antagonistActivation_d_lambda = self.alpha_E * (f_AN + 1) * (base_AN > 0)
        self.d_activation_d_lambda = np.diag(
            ((-self.d_agonistActivation_d_lambda + self.d_antagonistActivation_d_lambda)).squeeze())
        # print("f_AG", f_AG.T)
        # print("f_AN", f_AN.T)
        # print("E_AG", E_AG.T)
        # print("E_AN", E_AN.T)
        # print("d_activation_d_lambda=", self.d_activation_d_lambda)
        # print("Muscle activation =", E.T)
        return E

    def get_passive_elastic_torque(self, joint_angles):
        "Equation 8 Torque due to passive-elastic properties"
        theta_passive_deg = deepcopy(joint_angles) * 180 / np.pi
        theta_passive_deg[2] = -1 * theta_passive_deg[2] #TODO: Why? Similar to R&S simulation.m line 594

        stiffnessTorque_1 = np.exp(2.1016 - 0.0843 * theta_passive_deg[0] - 0.0176 * theta_passive_deg[1]) - np.exp(
            - 7.9763 + 0.1949 * theta_passive_deg[0] + 0.0008 * theta_passive_deg[1]) - 1.792
        stiffnessTorque_2 = np.exp(
            1.800 - 0.0460 * theta_passive_deg[0] - 0.0352 * theta_passive_deg[1] + 0.0217 * theta_passive_deg[
                2]) - np.exp(
            -3.971 - 0.0004 * theta_passive_deg[0] + 0.0495 * theta_passive_deg[1] - 0.0128 * theta_passive_deg[
                2]) - 4.820 + np.exp(2.220 - 0.150 * theta_passive_deg[1])
        stiffnessTorque_3 = np.exp(1.4655 - 0.0034 * theta_passive_deg[1] - 0.075 * theta_passive_deg[2]) - np.exp(
            1.3403 - 0.0226 * theta_passive_deg[1] + 0.0305 * theta_passive_deg[2]) + 8.072

        return np.array([stiffnessTorque_1, stiffnessTorque_2, stiffnessTorque_3])

    def muscle_model(self, obs, E):
        joint_angles = deepcopy(obs[self.joint_angles_obs_index].reshape(-1, 1))
        joint_velocities = deepcopy(obs[self.joint_velocities_obs_index].reshape(-1, 1))
        "Equation 5 Torque due to muscle activation"
        T_act_tilde = np.matmul(self.A, E)

        "Equation 7 (Calcium Kinetics, delay)"
        self.T_act_two_dot = (T_act_tilde - self.T_act - 2 * self.tau_c * self.T_act_dot) / self.tau_c ** 2
        self.T_act_dot = self.T_act_dot + self.T_act_two_dot * self.timestep
        self.T_act = self.T_act + self.T_act_dot * self.timestep

        "Equation 8 Torque due to passive-elastic properties"
        T_passive = self.get_passive_elastic_torque(joint_angles)

        "Equation 9 Torque due to passive-viscous properties"
        T_viscous = -np.matmul(self.B, joint_velocities)
        assert len(T_viscous) == len(joint_angles), f'Incorrect T_viscous length {len(T_viscous)}'

        return [self.T_act, T_passive, T_viscous]    


class Plotter:
    def __init__(self, step, data):
        self.row = len(data[0].keys())
        self.col = max([len(data[0][a]) for a in data[0].keys()])
        self.data = data
        self.axis = {}
        self.x = np.arange(step)
        self.lines = {}
        self.labels = list(data[0].keys())
        self.fig, self.axis = plt.subplots(self.row, self.col)
        self.make_lines()

    def __call__(self, step, data):
        self.data = data
        self.x = np.arange(step)
        if step == 1:
            self.make_lines()

        self.do_plot()
        # plt.savefig("plots")

    def make_lines(self):
        for cat_idx, category in enumerate(self.axis):
            for dim_idx, dim in enumerate(category):
                line, = dim.plot([], [])
                self.lines[(cat_idx, dim_idx)] = line

    def do_plot(self):
        for index, label in enumerate(self.labels):
            value = self.data[0][label]
            for idx, dim in enumerate(value):
                self.lines[(index, idx)].set_label(label + str(idx))
                self.lines[(index, idx)].set_xdata(self.x)
                self.lines[(index, idx)].set_ydata(dim)
                ax = self.axis[index, idx]
                ax.set_xlim(min(self.x), max(self.x))
                ax.set_ylim(np.min(dim), np.max(dim))
                ax.legend()

        plt.draw()
        plt.pause(0.0001)

if __name__ == '__main__':
    # Dictionary of all categories to be plotted corresponding to a key_nr
    categories = {
        "1": "Joint Angles",
        "2": "True Joint Angles",
        "3": "Joint Velocities",
        "4": "True Joint Velocities",
        "5": "Joint Accelerations",
        "6": "Head Position",
        "7": "Head Velocity",
        "8": "Head Acceleration",
        "9": "Torso RotEuler",
        "10": "Head Forcelet Position",
        "11": "Head Forcelet Orientation",
        "12": "Lambda Dot",
        "13": "Lambda",
        "14": "E",
        "15": "T",
        "16": "T_act",
        "17": "T_ela",
        "18": "T_vis",
        "19": "T_gravity"
    }

    # Choose the categories that you want to plot. If too many, the plots might be too small.
    choose = [str(1),str(3), str(6), str(6), str(10), str(11), str(15)]
    data = {}
    data_all = []

    RENDER_ENVIRONMENT = True  # Set to True to render the environment
    PLOT_RESULTS = 2  # 0 : No plot, 1 : Plot during simulation, 2 : Plot at the end of simulation

    horizon = 15000
    n_vestibular_delay_steps = 60
    n_proprioceptive_delay_steps = 15
    timestep = 0.002

    env = HumanStance(horizon=horizon, n_substeps=1, n_intermediate_steps=1, timestep=timestep, rs_controller=True, noisy=True, n_proprioceptive_delay_steps=n_proprioceptive_delay_steps, n_vestibular_delay_steps=n_vestibular_delay_steps, observation_noise_std=0.0)
    policy = lambda x: np.zeros(3) #We're using a dummy policy here. Action gets computed in the _compute_action() method

    state = env.reset()
    absorbing = False
    if RENDER_ENVIRONMENT:
        env.render()
    env.current_step = 0
    while True:
        # reset if needed
        if env.current_step == horizon or absorbing:
            # env.current_step = 0
            # state = env.reset()
            break

        # predict an action based on our policy.
        action = policy(state)

        # increment simulation
        next_state, reward, absorbing, _ = env.step(action)
        if env.current_step == 0:
            for cat in choose:
                data[categories[cat]] = env.data[categories[cat]]
            if PLOT_RESULTS == 1:
                plotter = Plotter(env.current_step + 1, [data])
        else:
            for key in data.keys():
                    data[key] = np.concatenate((data[key], env.data[key]), 1)
            data_all.append(data)
            if PLOT_RESULTS == 1:
                plotter(env.current_step + 1, [data])
        if RENDER_ENVIRONMENT:
            env.render()
        env.current_step += 1

        state = next_state
if PLOT_RESULTS == 2:
    plotter = Plotter(env.current_step, data_all)
    plotter(env.current_step, data_all)
plt.show()