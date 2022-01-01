import numpy as np
import random
from collections import namedtuple, deque

from Models.model_categorical import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-2             # learning rate
UPDATE_EVERY = 4        # how often to update the network
Vmin = -50
Vmax = 50
N = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.N = N
        self.Vmin = Vmin
        self.Vmax = Vmax

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, N, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, N, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.range_batch = torch.arange(BATCH_SIZE).long().to(device)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Z
        self.deltaT = (Vmax - Vmin)/ (N - 1)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            # WE MAKE 1 ET 2
            Z_dist = torch.from_numpy(np.asarray([[Vmin + i*self.deltaT for i in range(N)]])).to(device)
            Z_dist = torch.unsqueeze(Z_dist, 2).float()
            softmax,logsoftmax = self.qnetwork_local(state)
            softmax = softmax.detach()
            Q = torch.matmul(softmax,Z_dist)
            #action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(Q.cpu().data.numpy(),axis=1)[0]
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        m = torch.zeros(BATCH_SIZE,N).to(device)

        Z_dist_init = torch.from_numpy(np.asarray([[Vmin + i*self.deltaT for i in range(N)]])).to(device)
        Z_dist_init = torch.unsqueeze(Z_dist_init, 2).float()

        # Q TARGET
        M,_ = self.qnetwork_target(next_states)
        M = M.detach()


        Q_target = torch.matmul(M , Z_dist_init)

        Q = torch.argmax(Q_target,dim=1) # Q est la ligne qui correspond à a*
        Q_dist_star = M[self.range_batch, Q.squeeze(1), :]

        proba, log_proba = self.qnetwork_local(states)
        #print("LOG")
        #print(log_proba.shape)
        # Je comprends pas cette ligne
        log_proba = log_proba[self.range_batch, actions.squeeze(1),:]  # .reshape(-1, self.action_size, self.N)[self.range_batch, actions.squeeze(1), :]
        #print(log_proba.shape)

        # FOR ONE SAMPLE ONLY -> Make for a Batch

        # Enumerate value on atoms. Each atom is translate
        for n in range(self.N-1):
            Non_aligned_batch_zj = rewards + gamma *(1-dones)*(self.Vmin + n*self.deltaT)#Z_dist_init[0][n].repeat(BATCH_SIZE,1)
            torch.clamp_(Non_aligned_batch_zj, min=self.Vmin, max=self.Vmax)
            diff = (Non_aligned_batch_zj - self.Vmin) / self.deltaT
            l = torch.floor(diff).long()
            u = torch.ceil(diff).long()

            # Calculate the distance weight
            weight1 = u - diff
            weight2 = diff - l
            """
            Init1 = torch.zeros(BATCH_SIZE).to(device)
            Init2 = torch.zeros(BATCH_SIZE).to(device)
            compteur =0
            for BatchM,BatchQ in zip(M,Q):
                Init1[compteur] += BatchM[BatchQ, n][0] * weight1[compteur][0]
                Init2[compteur] += BatchM[BatchQ, n][0] * weight2[compteur][0]

            m[:,n] += Init1
            m[:, n+1] += Init2
            """

            # IMPLEMENTATION BEAUCOUP PLUS RAPIDE QUE LA BOUCLE #
            mask_Q_l = torch.zeros(m.size()).to(device)
            mask_Q_l.scatter_(1, l, Q_dist_star[:,n].unsqueeze(1))
            mask_Q_u = torch.zeros(m.size()).to(device)
            mask_Q_u.scatter_(1, u, Q_dist_star[:,n].unsqueeze(1))
            m += mask_Q_l*(u.float() + (l == u).float()-diff.float())
            m += mask_Q_u*(-l.float()+diff.float())


        # Compute loss
        loss = - torch.sum(torch.sum(torch.mul(log_proba, m),-1),-1) / BATCH_SIZE
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def export_network(self):
        torch.save(self.qnetwork_local.state_dict(), 'dqn_agent.pth')


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)