from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
from Agents.dqn_agent import *
import pickle

env = UnityEnvironment(file_name="C:/Users/gabyc/Desktop/Reinforcment_TP/Value-based-methods/p1_navigation/Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

agent = Agent(state_size=len(env_info.vector_observations[0]), action_size=brain.vector_action_space_size, seed=0)
def DQN(n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    Score_All = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0        # initialize the score
        while True:
            action = agent.act(state, eps=eps) # select an action
            env_info = env.step(int(action))[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done) # UPDATE
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                   # exit loop if episode finished
                break
        scores_window.append(score)
        Score_All.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    agent.export_network()
    return Score_All

scores = DQN()

with open('DQN_scores.pkl', 'wb') as f:
    pickle.dump(scores, f)