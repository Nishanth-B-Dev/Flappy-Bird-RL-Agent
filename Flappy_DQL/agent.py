import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import DQN


class Agent:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main model
        self.model = DQN().to(self.device)

        self.target_model = DQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=10000)

        self.gamma = 0.9
        self.epsilon = 1.0

        self.update_counter = 0

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, 1)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        q_values = self.model(state)

        return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        # Current Q
        q_values = self.model(states)

        #  Use Target Network
        next_q_values = self.target_model(next_states)

        target = q_values.clone()

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i])

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # To Update Target Network Every Few Steps 
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())