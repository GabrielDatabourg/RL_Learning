from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np
from Agents.dqn_agent import *
from Models.model import QNetwork

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = UnityEnvironment(file_name="C:/Users/gabyc/Desktop/Reinforcment_TP/RL_Learning/Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

state_size=len(env_info.vector_observations[0])
action_size=brain.vector_action_space_size

# Import and load the model
lettre=input("Which model you would like to use? DQN press A, DoubleDQN press B, DuelingDQN press C !")
if lettre == "A" or lettre == "a":
  PATH = "Results/dqn_agent.pth"
elif lettre == "B" or lettre == "B":
  PATH = "Results/doubledqn_agent.pth"
else:
  PATH = "Results/dueling_dqn_agent.pth"
model = QNetwork(state_size, action_size, seed=0).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

agent = Agent(state_size, action_size, seed=0)
score = 0
state = env_info.vector_observations[0] 
while True:
  ### Act with the trained model ###
  state = torch.from_numpy(state).float().unsqueeze(0).to(device)
  model.eval()
  with torch.no_grad():
    action_values = model(state)
  action = np.argmax(action_values.cpu().data.numpy())
  ## Interact and grab data from the environment ##
  env_info = env.step(int(action))[brain_name]   # send the action to the environment
  next_state = env_info.vector_observations[0]   # get the next state
  reward = env_info.rewards[0]                   # get the reward
  done = env_info.local_done[0]                  # see if episode has finished
  score += reward                                # update the score
  state = next_state                             # roll over the state to next time step
  if done:
    break