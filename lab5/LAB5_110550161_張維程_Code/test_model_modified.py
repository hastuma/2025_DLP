import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions,env_name):
        super(DQN, self).__init__()
        if env_name == "CartPole-v1":
            # print("Using CartPole-v1")
            self.network = nn.Sequential(
                nn.Linear(4, 128),  
                    nn.ReLU(),
                    nn.Linear(128, 128),  
                    nn.ReLU(),
                    nn.Linear(128, num_actions)  
            )
        elif env_name == "ALE/Pong-v5":
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )

    def forward(self, x):
        # print(x.shape)
        # input("Press Enter to continue...")
        if x.ndim == 4 and x.shape[1:] == (4, 84, 84):
            x = x / 255.0
        return self.network(x)
        
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
        
def evaluate(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions,args.env_name).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        # state = preprocessor.reset(obs)
        if args.env_name == "CartPole-v1":
            state = obs  # CartPole-v1 的輸入是 1D 向量
        else:
            state = preprocessor.reset(obs)  
        # print(obs.shape)
        # print(state.shape)
        # input("Press Enter to continue...")
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            if args.env_name == "CartPole-v1":
                # print("Using CartPole-v1")
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            else:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device) / 255.0  # Pong 的輸入需要歸一化
            # state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # print(state_tensor.shape)
            # input("Press Enter to continue...")
            with torch.no_grad():
                # print("ep: ",ep,"state_tensor",state_tensor)
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            # state = preprocessor.step(next_obs)
            if args.env_name == "CartPole-v1":
                state = next_obs  # CartPole-v1 的輸入是 1D 向量
            else:
                state = preprocessor.reset(next_obs)  
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        # print(frames.shape)
        # print(frames[0].shape)
        with imageio.get_writer(out_path, fps=30, format='FFMPEG') as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./task1_eval_videos")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="ALE/Pong-v5 or CartPole-v1")
    args = parser.parse_args()
    evaluate(args)
