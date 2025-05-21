#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=Mish):
        super().__init__() 
        ############TODO#############
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions)
        )
        
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
        #############################
    
    def forward(self, X):
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        
        return torch.distributions.Normal(means, stds)

class Critic(nn.Module):
    def __init__(self, state_dim, activation=Mish):
        """Initialize."""
        super().__init__()
        ############TODO#############
        # Remeber to initialize the layer weight   s
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, 1),
        )
        #############################
    def forward(self, X):
         """Forward method implementation."""
        ############TODO#############
         return self.model(X)
        #############################
class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.max_grad_norm = 0.5  # Add gradient clipping threshold
        
        # device: cpu / gpu
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim, activation=Mish).to(self.device)
        self.critic = Critic(obs_dim, activation=Mish).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # memory for batch processing
        self.memory = []
        self.steps_on_memory = args.steps_on_memory  # Number of steps to collect before updating

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        dist = self.actor(state)
        selected_action = dist.mean if self.is_test else dist.sample()

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])
            self.memory.append(self.transition)
            self.transition = []

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        if len(self.memory) < self.steps_on_memory:
            return 0.0, 0.0

        # Process batch
        states = []
        log_probs = []
        next_states = []
        rewards = []
        dones = []

        for transition in self.memory:
            state, log_prob, next_state, reward, done = transition
            states.append(state)
            log_probs.append(log_prob)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

        # Convert to tensors
        states = torch.stack(states)
        log_probs = torch.stack(log_probs)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)

        # Compute target values
        mask = 1 - dones
        target_values = rewards + self.gamma * self.critic(next_states) * mask
        values = self.critic(states)

        # Value loss
        value_loss = F.mse_loss(values, target_values.detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Compute advantages
        advantages = (target_values - values).detach()

        # Policy loss
        dists = self.actor(states)
        entropy = dists.entropy().mean()
        policy_loss = -(log_probs * advantages).mean() - self.entropy_weight * entropy

        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Clear memory
        self.memory = []

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        reach_goal = False
        for ep in tqdm(range(1, self.num_episodes)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset(seed=self.seed)
            score = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action) # Take an action and return the response of the env.

                
                if len(self.memory) >= self.steps_on_memory: # Update model if enough steps > default 16 
                    actor_loss, critic_loss = self.update_model()
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1

                # W&B logging
                if len(actor_losses) > 0:
                    wandb.log({
                        "step": step_count,
                        "actor loss": actor_losses[-1],
                        "critic loss": critic_losses[-1],
                    })

                if done:
                    scores.append(score)
                    print(f"Episode {ep}: Total Reward = {score}")
                    wandb.log({
                        "episode": ep,
                        "return": score
                    })
                    # if score > -150.0:
                    #     print(f"Early stopping at episode {ep} with score {score}")
                    #     reach_goal = True
                    #     break
        
        # Save the trained models in a single dictionary
        model_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }
        torch.save(model_dict, './task1_results/ LAB7_110550161_task1_a2c_pendulum.pt')
        print("Model saved successfully!")

    def test(self, video_folder: str):
        """Test the agent over 20 episodes and calculate average score."""
        self.is_test = True
        num_eval_episodes = 20
        scores = []
        seeds = [0,14,15,23,24,37,42,63,78,80,84,100,103,114,124,135,141,160, 178,180]
        # Create video recording environment
        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        for episode in range(num_eval_episodes):
            seed = seeds[episode]
            state, _ = self.env.reset(seed=seed )  # Different seed for each episode + episode
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward

            scores.append(score)
            # print(f"Evaluation Episode {episode + 1}: random seed : {rand_seed } Score = {score}")
            print(f"seed : {seed } Score = {score}")

        # Calculate and print average score
        avg_score = sum(scores) / len(scores)
        print(f"\nAverage Score over {num_eval_episodes} episodes: {avg_score}")
        print(f"All scores: {scores}")
        print(seeds)
        # Log results to wandb
        wandb.log({
            "test_avg_score": avg_score,
            "test_scores": scores
        })

        self.env.close()
        self.env = tmp_env

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--steps_on_memory", type=int, default=16) # Number of steps to collect before updating

    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(
        project="DLP-Lab7-A2C-Pendulum", 
        name=args.wandb_run_name, 
        save_code=True,   
        config={
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "discount_factor": args.discount_factor,
            "num_episodes": args.num_episodes,
            "entropy_weight": args.entropy_weight,
            "mem step ":args.steps_on_memory
        }
    )
    
    agent = A2CAgent(env, args)
    agent.train()
    agent.test("task1_results")
    # python a2c_pendulum.py --actor-lr 1e-3 --critic-lr 1e-3 --discount-factor 0.9 --num-episodes 1000 --steps_on_memory 64