# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions,env_name,task, hidden_channels=512):
        super(DQN, self).__init__()

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.task = task
        if env_name == "CartPole-v1":
            # CartPole
            print("DQN for CartPole")
            self.feature_layer = nn.Sequential(
                nn.Linear(4, 128),  
                nn.ReLU(),
                nn.Linear(128, 128),  
                nn.ReLU(),
                nn.Linear(128, num_actions)  
        )
        else:
            # # Pong
            print("DQN model for Pong")
            self.feature_layer = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(64, hidden_channels, kernel_size=7, stride=1, padding=3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_channels * 7 * 7, 512),
                nn.ReLU(),
            )
        if task == 3 : 
            # Dueling DQN 的 Value 和 Advantage Stream
            self.value_stream = nn.Linear(128 if env_name == "CartPole-v1" else 512, 1)
            self.advantage_stream = nn.Linear(128 if env_name == "CartPole-v1" else 512, num_actions)
        else:
            # 標準 DQN 的輸出層
            self.output_layer = nn.Linear(128 if env_name == "CartPole-v1" else 512, num_actions)
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        # print(f'Input shape: {x.shape}')
        # k = input()
        #Input shape: torch.Size([1, 4])
        # print(x)
      
        if x.ndim == 4 and x.shape[1:] == (4, 84, 84):
            x = x / 255.0
        features = self.feature_layer(x)

        if self.task ==3:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))  # 合併 Value 和 Advantage
        else:
            q_values = self.output_layer(features)

        return q_values


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if (obs.shape==(4,)):
           return obs
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        if (obs.shape==(4,)):
           return obs
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        if (obs.shape==(4,)):
           return obs
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
    def __len__(self):
        return len(self.buffer)
    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        state, action = self.n_step_buffer[0][:2]
        return (state, action, reward, next_state, done)
    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # 插入新經驗與其 priority
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) == self.n_step:
            multi_step_transition = self._get_n_step_info()
            p = (abs(error) + 1e-6) ** self.alpha
            if len(self.buffer) < self.capacity:
                self.buffer.append(multi_step_transition)
            else:
                self.buffer[self.pos] = multi_step_transition
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # 根據 priority 抽樣
        scaled_p = self.priorities[:len(self.buffer)] ** self.alpha
        probs = scaled_p / scaled_p.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        # 計算重要性取樣權重
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normalize
        return indices, samples, torch.tensor(weights, dtype=torch.float32, device=self.device)
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # 更新被抽樣 transitions 的 priority
        for idx, err in zip(indices, errors.detach().cpu().numpy()):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        if args.task == 3:
            env_name = "Pong"
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        env_name = args.wandb_run_name
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.lr = args.lr
        self.memory_size = args.memory_size
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.num_actions,env_name,args.task).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions,env_name,args.task).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir

        if args.task != 3:
            self.memory = deque(maxlen=args.memory_size) #用deque才會把最舊的經驗丟掉
        elif args.task == 3:
            self.memory = PrioritizedReplayBuffer(args.memory_size, n_step=3, gamma=self.gamma)
        self.args = args

        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            if (obs.shape==(4,)):
                state = obs
            else:
                state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                if self.args.task == 3:
                    # Add multi-step transition to PER buffer
                    transition = (state, action, reward, next_state, done)
                    state_t = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                    next_state_t = torch.from_numpy(np.array(next_state, dtype=np.float32)).unsqueeze(0).to(self.device)
                    action_t = torch.tensor([action], dtype=torch.int64).to(self.device)
                    reward_t = torch.tensor([reward], dtype=torch.float32).to(self.device)
                    done_t = torch.tensor([done], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        q_val = self.q_net(state_t).gather(1, action_t.unsqueeze(1)).squeeze(1)
                        next_q = self.target_net(next_state_t).max(1)[0]
                        target = reward_t + (1 - done_t) * self.gamma * next_q
                        init_error = (target - q_val).item()
                    self.memory.add(transition, init_error)
                else:
                    # 非 PER 任務就維持原本 append
                    self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            wandb.config.update({
            "batch_size": self.batch_size,
            "memory_size": self.memory_size,
            "learning_rate": self.lr,
            "discount_factor": self.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "target_update_frequency": self.target_update_frequency,
            "replay_start_size": self.replay_start_size,
            "max_episode_steps": self.max_episode_steps,
            "train_per_step": self.train_per_step,
            "save_dir": self.save_dir
            })
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                # print(state_tensor.shape)
                # input("Press Enter to continue...")
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
    
        ########## YOUR CODE HERE (<5 lines) ##########
        if self.args.task != 3:
            indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
            batch = [self.memory[idx] for idx in indices]
        elif self.args.task == 3:
            indices, batch, is_weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        ########## END OF YOUR CODE ##########

        # Convert to torch tensors
        states = torch.from_numpy(states.astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(next_states.astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if args.task != 3 :
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            elif args.task == 3: # DDQN
                # 1) 用線上網路選出最優動作
                next_actions = self.q_net(next_states).argmax(dim=1)
                # 2) 用 target 網路評估該動作的價值
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        if self.args.task == 3:
            # TD error & 加權 MSE loss
            td_errors = target_q_values - q_values
            loss = (is_weights * td_errors.pow(2)).mean()
            # 更新 priority
            self.memory.update_priorities(indices, td_errors.abs())

        elif self.args.task != 3:
            loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./trash")
    # parser.add_argument("--wandb-run-name", type=str, default="CartPole-v1")
    parser.add_argument("--wandb-run-name", type=str, default="Pong")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=10000)
    parser.add_argument("--replay-start-size", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    print(args)
    # wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(args.episodes)


# python dqn.py --wandb-run-name Pong\
#   --memory-size 100000 \
#   --replay-start-size 1000 \
#   --batch-size 128 \
#   --lr 1e-4 \
#   --epsilon-decay 0.995\
#   --target-update-frequency 7500 \
#   --train-per-step 4 



# python rainbow_myself.py \
#   --wandb-run-name Pong-DuelingDQN-PER \
#   --task 3 \
#   --memory-size 10000 \
#   --replay-start-size 10000 \
#   --batch-size 32 \
#   --lr 1e-4 \
#   --discount-factor 0.99 \
#   --epsilon-start 1.0 \
#   --epsilon-decay 0.999985 \
#   --epsilon-min 0.01 \
#   --target-update-frequency 1000 \
#   --train-per-step 1
#   --max-episode-steps 800


#   --wandb-run-name Pong \
#   --memory-size 10000 \
#   --replay-start-size 10000 \
#   --batch-size 32 \
#   --lr 1e-4 \
#   --discount-factor 0.99 \
#   --epsilon-start 1.0 \
#   --epsilon-decay 0.999985 \
#   --epsilon-min 0.01 \
#   --target-update-frequency 1000 \
#   --train-per-step 1 \
#   --max-episode-steps 1500



# python rainbow_myself.py   --wandb-run-name
#  Pong-DuelingDQN-PER --max-episode-steps 100000 --task 3  --episodes 10000 --memory-size 100000   --replay-start-siz
# e 10000   --batch-size 32   --lr 1e-4   --discount-factor 0.99   --epsilon-start 1.0   --epsilon-decay 0.999985   --epsilon-min 0.01   --target-update-frequency 1000   --train-per-step 4

# python rainbow_myself.py   --
# wandb-run-name Pong-DuelingDQN-PER --max-episode-steps 100000 --task 3  --episodes 10000 --memory-size 100000   --replay-start-size 10000   --batch-size 32   --lr 1e-4   --discount-factor 0.99   --epsilon-start 1.0   --epsilon-decay 0.999985   --epsilon-min 0.01   --target-update-frequency 1000   --train-per-step 4