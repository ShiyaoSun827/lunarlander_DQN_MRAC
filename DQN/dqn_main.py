import gym
from DQN.dqn_agent import DQNagent
from env.DQN_MRACenv import LunarLanderDQNMRACWrapper

train_mode = 'test'  # 'traindqn', 'exosystem', or 'test'
# 'traindqn' for training DQN agent
# 'exosystem' for training with MRAC
# 'test' for testing the trained agent
if(train_mode == 'traindqn'):#It does not work anymore.Use the exosystem env mode instead.
    # Create environment
    env = gym.make("LunarLander-v2")  # Discrete action space version
    #env = gym.make("LunarLander-v2", render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize DQN agent
    agent = DQNagent(state_dim=state_dim, action_dim=action_dim)

    # Train agent with model saving every 100 episodes
    agent.train(env, num_episodes=1000, max_timesteps=1000, save_path="./models")

    # Plot training reward curve
    agent.plot_training_curve()
    test_env = gym.make("LunarLander-v2", render_mode="human")
    # Test the trained policy
    #agent.test(env, num_episodes=5, render=True)
    agent.test(num_episodes=5)
elif train_mode == 'exosystem':
    # 1. create a env with MRAC 
    env = LunarLanderDQNMRACWrapper(exo_mode='sin')
    

    # 2. initialize DQN agent
    state_dim = env.observation_space.shape[0]
    action_dim = len(env.reference_models)  # every index -> A_m, B_m

    agent = DQNagent(state_dim=state_dim, action_dim=action_dim)

    # 3. training
    agent.train(env, num_episodes=1000, max_timesteps=1000, save_path="./models")

    # 4. plot training curve
    agent.plot_training_curve()

    # 5. test the trained policy
    agent.test(num_episodes=5)
else:
    env = LunarLanderDQNMRACWrapper(exo_mode='sin')
    # Test the trained policy
    #agent.test(env, num_episodes=5, render=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNagent(state_dim=state_dim, action_dim=action_dim)
    agent.load("./models/best.pth") 
    agent.test(num_episodes=5)
