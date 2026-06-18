import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Initialize environment
env = gym.make("CartPole-v1")

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

def choose_action(state, epsilon, policy_net, device):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_t)
            return q_values.argmax(dim=1).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() 

memory = deque(maxlen=10000)
num_episodes = 800
gamma = 0.99
batch_size = 64

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9975      
target_update_episodes = 10  

optimizer = optim.Adam(policy_net.parameters(), lr=5e-4) 
loss_fnc = nn.MSELoss()

for i in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = choose_action(state, epsilon, policy_net, device)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        
        memory.append((state, action, reward, next_state, done))
        state = next_state
        
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            
            states = torch.FloatTensor(np.array([t[0] for t in batch])).to(device)
            actions = torch.LongTensor([t[1] for t in batch]).to(device)
            rewards = torch.FloatTensor([t[2] for t in batch]).to(device)
            next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(device)
            dones = torch.FloatTensor([t[4] for t in batch]).to(device)
            
            current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(dim=1)[0]
                target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
            
            loss = loss_fnc(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if i % target_update_episodes == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if i % 10 == 0:
        print(f"Episode {i}: Reward = {episode_reward:.1f}, Epsilon = {epsilon:.3f}")

env.close()