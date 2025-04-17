import gym
import numpy as np
from env.MRACExosystemenv import MRACExosystem
from MRAC.MRAC_model import MRACReferenceModel, MRACPlainModel, MRACController, MRACErrorModel, MRACObserver
def generate_reference_models():
    models = []

    # Vertical landing: strengthen y and vy control, weaken x
    A_vertical = np.eye(4)
    B_vertical = np.array([[0.0], [0.15], [0.0], [0.2]])
    models.append((A_vertical, B_vertical))

    # Left-biased strategy: increase the negative x response
    A_tilt_left = np.eye(4)
    B_tilt_left = np.array([[-0.1], [0.1], [-0.05], [0.2]])
    models.append((A_tilt_left, B_tilt_left))

    # Right-biased strategy: increase the positive x response
    A_tilt_right = np.eye(4)
    B_tilt_right = np.array([[0.1], [0.1], [0.05], [0.2]])
    models.append((A_tilt_right, B_tilt_right))

    # Hovering strategy: try to suppress speed and keep the height unchanged
    A_hover = np.diag([1.0, 1.0, 0.95, 0.95])
    B_hover = np.array([[0.0], [0.05], [0.0], [0.05]])
    models.append((A_hover, B_hover))

    return models
class LunarLanderDQNMRACWrapper(gym.Wrapper):
    def __init__(self, env_name='LunarLander-v2', exo_mode="sin", render=False):
        #env = gym.make(env_name)
        mode = "human" if render else None
        env = gym.make(env_name, render_mode=mode)
        super().__init__(env)
        self.env = env

        # MRAC control toggle
        self.use_mrac = True


        # Exosystem for disturbance
        self.exosystem = MRACExosystem(mode=exo_mode, amplitude=0.05, frequency=0.1, dim=2)

        # System dimensions
        state_dim = 4  # [x, y, vx, vy] for MRAC tracking
        self.check = False
        phi_dim = state_dim * 3  # assuming phi = [x, x^2, tanh(x)]

        # Placeholder model matrices (to be replaced with DQN output or manual setting)
        A_m = np.eye(state_dim)
        B_m = np.ones((state_dim, 1)) * 0.1
        A = np.eye(state_dim)
        B = np.ones((state_dim, 1)) * 0.05
        E = np.eye(state_dim, 2) * 0.01  # disturbance injected into x, y
        C = np.eye(state_dim)
        L = np.eye(state_dim) * 0.1
        Pi = np.zeros((state_dim, 2))

        # Core MRAC modules
        self.reference_model = MRACReferenceModel(A_m, B_m)
        self.controller = MRACController(state_dim=state_dim, phi_dim=phi_dim)
        self.controller_main  = MRACController(state_dim=state_dim, phi_dim=phi_dim)
        self.controller_left  = MRACController(state_dim=state_dim, phi_dim=phi_dim)
        self.controller_right = MRACController(state_dim=state_dim, phi_dim=phi_dim)
        # Error model for tracking
        self.error_model = MRACErrorModel(use_exosystem=True, Pi=Pi)
        self.observer = MRACObserver(A, B, C, L)
        self.plant_model = MRACPlainModel(A, B, E)
        # DQN-indexed reference model library
        # self.reference_models = [
        #     (np.eye(4), np.ones((4,1)) * 0.1),                     # model 0
        #     (0.95*np.eye(4), np.eye(4)[:, [0]] * 0.2),             # model 1
        #     (np.diag([0.9, 0.9, 1.0, 1.0]), np.ones((4,1)) * 0.05) # model 2
        # ]
        self.reference_models = generate_reference_models()



    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self.exosystem.reset()
        self.reference_model.reset()
        self.observer.reset()
        return obs

    def step(self, action):
        xi = self.exosystem.step()

        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.check:
            control_input, x_m, x, xi = self.mrac(obs)
            info['mrac_control'] = control_input
            info["x_m"] = x_m
            info["x"] = x
            info["xi"] = xi
        # --- reward shaping ---
            shaped_reward = 0.0

            # MRAC tracking error: encourage tracking reference model
            tracking_error = np.linalg.norm(x - x_m)
            shaped_reward -= tracking_error / 10.0  # scale down to ~[0, 1]

            # low altitude + high speed => punishment
            if obs[1] < 0.1 and abs(obs[3]) > 0.5:
                shaped_reward -= 0.3

            # low altitude + low speed + low angle => encourage stable landing
            if obs[1] < 0.2 and abs(obs[3]) < 0.1 and abs(obs[5]) < 0.1:
                shaped_reward += 0.5
                info["landed"] = True

            # angle penalty
            if abs(obs[4]) > 0.4:
                shaped_reward -= 0.1

            # horizontal drift penalty
            if abs(obs[0]) > 0.5:
                shaped_reward -= 0.1

            # clip final reward to stable range
            reward = np.clip(shaped_reward, -1.0, 1.0)

        return obs, reward, terminated, truncated, info
    def map_mrac_to_action(self, u, obs=None):
        u = np.array(u).flatten()
        u_main, u_left, u_right, _ = u 
        threshold = 0.001
        if obs is not None:
            angle = obs[4]       
            angle_dot = obs[5]  
            if angle > 0.3 and u_right > u_left:
                u_right = -np.inf  
            elif angle < -0.3 and u_left > u_right:
                u_left = -np.inf  
        if np.max(np.abs(u)) < threshold:
            return 0  
        if u_main > max(u_left, u_right):
            return 1
        elif u_left > u_right:
            return 2
        else:
            return 3

    def set_reference_model(self, index):
        A_m, B_m = self.reference_models[index]
        self.reference_model = MRACReferenceModel(A_m, B_m)
    def apply_mrac_(self, obs):
        xi = self.exosystem.step()
        x = np.array([obs[0], obs[1], obs[2], obs[3]])

        # MRAC reference trajectory (fixed reference input r)
        #r = np.zeros((1,))  # desired constant position
        r = np.array([[1.0]]) 

        x_m = self.reference_model.step(r)

        # Tracking error
        e = self.error_model.compute_error(x, xi)

        # Observer update
        u = np.zeros((1,))  # 单一控制量（main engine）
        x_hat = self.observer.step(u=u, e=e)

        # MRAC control
        #u, phi_x = self.controller.get_control(x)

        # Update adaptive parameters
        #self.controller.update(phi_x, e.mean())
        # --- Control for each engine ---
        u_main,  phi_x_main  = self.controller_main.get_control(x)
        u_left,  phi_x_left  = self.controller_left.get_control(x)
        u_right, phi_x_right = self.controller_right.get_control(x)

        # Update each controller
        self.controller_main.update(phi_x_main, e.mean())
        self.controller_left.update(phi_x_left, e.mean())
        self.controller_right.update(phi_x_right, e.mean())

        # Output combined u vector
        u = [u_left, u_main, u_right]

        return u, x_m, x, xi
    def mrac(self, obs):
        xi = self.exosystem.step()
        x = np.array([obs[0], obs[1], obs[2], obs[3]])

        # MRAC reference trajectory (fixed reference input r)
        r = np.zeros((1,))  # desired constant position
        x_m = self.reference_model.step(r)

        # Tracking error
        e = self.error_model.compute_error(x, xi)

        # Observer update
        u = np.zeros((1,)) 
        x_hat = self.observer.step(u=u, e=e)

        # MRAC control
        u, phi_x = self.controller.get_control(x)

        # Update adaptive parameters
        self.controller.update(phi_x, e.mean())

        return u, x_m, x, xi

    def compute_state_error(self, obs, xi):
        x_ref = np.array([0.0, 0.0, 0.0, 0.0])
        x_ref[0] += xi[0]
        x_ref[1] += xi[1]
        current = np.array([obs[0], obs[1], obs[2], obs[3]])
        return current - x_ref


    
'''
添加exosystem后的 LunarLander 环境.
env = LunarLanderDQNMRACWrapper(exo_mode='sin')
env.use_mrac = True

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("MRAC control input:", info.get("mrac_control"))

'''
