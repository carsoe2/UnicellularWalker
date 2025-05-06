import animate_walker as animator
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import re
import noisy_walker_v6 as v6

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Define Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Flatten observation dict to vector
def preprocess_obs(obs):
    leg_state = obs['leg_state']
    theta = obs['theta']
    return np.concatenate([leg_state, theta])

def train_dqn(env, episodes=100, max_steps=200, gamma=0.99, lr=1e-3, epsilon=1.0, min_epsilon=0.05, batch_size=64, buffer_size=10000, target_update_freq=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_size = len(preprocess_obs(env.reset()[0]))
    num_actions = env.action_space.n

    policy_net = QNetwork(obs_size, num_actions).to(device)
    target_net = QNetwork(obs_size, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = deque(maxlen=buffer_size)
    writer = SummaryWriter()
    max_eps = epsilon
    reward_vals = []

    step_count = 0
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            state = preprocess_obs(obs)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_obs(next_obs)

            replay_buffer.append((state, action, reward, next_state, terminated))
            obs = next_obs
            total_reward += reward
            reward_vals.append(reward_vals)
            step_count += 1

            # Sample minibatch
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if terminated or truncated:
                break

        epsilon = epsilon - (max_eps-min_epsilon)/episodes
        writer.add_scalar("Episode Reward", total_reward, episode)
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    writer.close()
    return policy_net

class LegWalkerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.num_legs = 14
        self.actions = [i for i in range(14)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.initial_cell_state, self.initial_leg_attach, self.initial_leg_state = get_initial_position()
        self.observation_space = spaces.Dict({
            "leg_state": spaces.MultiBinary(self.num_legs),
            "theta": spaces.Box(low=0.0, high=2*np.pi, shape=(1,), dtype=np.float32)
        })
        self.x_goal = self.initial_cell_state[0]+30
        self.max_steps = 200
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cell_state = self.initial_cell_state
        self.leg_attach = self.initial_leg_attach
        self.leg_state = self.initial_leg_state.copy()
        self.truncated = False
        self.terminated = False
        self.current_steps = 0
        self.prev_action = None
        self.info = {}
        self.obs = {
            "leg_state": np.array(self.leg_state, dtype=np.int8),
            "theta": np.array([self.cell_state[2]-self.initial_cell_state[2]], dtype=np.float32)
        }
        return self.obs, self.info

    def step(self, action):
        prev_cell_state = self.cell_state

        self.leg_attach, self.leg_state, self.cell_state = \
            v6.noisy_walker_v6(self.leg_attach, self.leg_state, action, self.cell_state)
        dx = self.cell_state[0] - prev_cell_state[0]

        self.obs = {
            "leg_state": np.array(self.leg_state, dtype=np.int8),
            "theta": np.array([self.cell_state[2]-self.initial_cell_state[2]], dtype=np.float32)
        }
        # reward = x distance traveled since the previous step
        reward = self.cell_state[0] - prev_cell_state[0]
        self.current_steps += 1
        self.terminated = False
        self.truncated = False

        # Constraint: At least 3 legs on the ground
        # Constraint: Leg "cooldown"
        if self.leg_state.count(0) < 3: # or self.prev_action == action:
            reward = -5
        
        self.prev_action = action

        return self.obs, reward, self.terminated, self.truncated, self.info

# used to separate file reading from the evironment reset to speed up the runtime
def get_initial_position(infile='../input_files/cirrus_testfile_full_rotation.txt'):
    with open(infile, 'r') as fid:
        numlines = 0
        big_string = ""

        # Read the whole file as one huge string
        for tline in fid:
            tline = tline.strip()  # Remove newline characters and extra spaces
            if not tline.startswith('%'):  # Skip any comment lines
                numlines += 1
                big_string += ' ' + tline
    # Pad any non-space delimiters to ensure proper splitting
    big_string = re.sub(r'{', ' { ', big_string)
    big_string = re.sub(r'}', ' } ', big_string)
    big_string = re.sub(r'\[', ' [ ', big_string)
    big_string = re.sub(r'\]', ' ] ', big_string)
    big_string = re.sub(r',', ' , ', big_string)

    # Trim out multiple white spaces to prevent empty tokens
    big_string = re.sub(r'\s+', ' ', big_string).strip()

    # Split the string into tokens based on spaces
    input_words = big_string.split(' ')

    # Count the number of words
    num_words = len(input_words)

    current_cirrus_ID = 0
    cirrus_name = []
    xpos = []
    ypos = []
    leg_attach = []

    for i in range(num_words):
        word = input_words[i]
        
        if word.startswith('{'):
            current_cirrus_ID += 1  # Create a new ID for each new cirrus when defined
            cirrus_name.append(input_words[i + 1])
        elif word.startswith('['):
            number_string1 = input_words[i + 1]  # Extract the next word
            number_string2 = input_words[i + 3]  # Extract the word after the comma
            
            xpos.append(float(number_string1))  # Convert to float
            ypos.append(float(number_string2))  # Convert to float
            leg_attach.append((float(number_string1), float(number_string2)))  # Convert to float

    num_cirri = current_cirrus_ID

    # Calculate the longest linear dimension of the cell based on the largest
    # distance between pairs of cirri
    max_distance = 0
    for i in range(num_cirri):
        for j in range(num_cirri):
            distance = np.sqrt((xpos[i] - xpos[j]) ** 2 + (ypos[i] - ypos[j]) ** 2)
            if distance > max_distance:
                max_distance = distance

    # Calculate the centroid of the cell
    x_centroid = sum(xpos) / num_cirri
    y_centroid = sum(ypos) / num_cirri
    cell_state = (x_centroid, y_centroid, 0)
    leg_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return cell_state, leg_attach, leg_state

# runs the model for max_steps steps and shows the animation based on the results
def run_and_animate_model(model, env, max_steps=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    obs, _ = env.reset()
    leg_matrix = []
    x_coords = []
    y_coords = []
    actions = []
    rewards = []


    for _ in range(max_steps):
        leg_matrix.append(obs["leg_state"].tolist().copy())  # Save leg state
        x_coords.append(env.cell_state[0])
        y_coords.append(env.cell_state[1])

        if model:
            state = preprocess_obs(obs)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(action)

        if terminated or truncated:
            break

    print("leg matrix:", leg_matrix)
    print("x coords:", x_coords)
    print("y coords:", y_coords)
    print("actions:", actions)
    print("rewards:", rewards)

    animator.animate_spider(leg_matrix, x_coords, y_coords)

if __name__ == '__main__':
    env = LegWalkerEnv()
    obs_dim = len(preprocess_obs(env.reset()[0]))
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random_steps = False
    if random_steps:
        run_and_animate_model(None, env)
    else:
        ### Set to false if model should be loaded
        train = False
        if train:
            # Train model
            model = train_dqn(env)

            # Save trained model
            torch.save(model.state_dict(), "dqn_model.pt")
        else:
            # Re-initialize model architecture
            model = QNetwork(input_dim=obs_dim, output_dim=action_dim).to(device)

            # Load the trained weights
            model.load_state_dict(torch.load("dqn_model.pt", map_location=device))

        # Run and animate
        run_and_animate_model(model, env)