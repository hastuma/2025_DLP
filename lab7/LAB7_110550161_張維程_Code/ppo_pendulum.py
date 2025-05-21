#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh
import math
import random
from collections import deque
from typing import Deque, List, Tuple

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

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

def orthogonal_init(layer: nn.Linear, gain: float = math.sqrt(2), bias_const: float = 0.0):
    """Orthogonal weight init with configurable gain (PPO detail #2)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(in_dim, 64)), nn.Tanh(),
            orthogonal_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        # Mean head (very small init, centred around 0)
        self.mu_head = orthogonal_init(nn.Linear(64, out_dim), gain=0.01)
        # State-independent log-std (initially 0)
        self.log_std = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().clamp(1e-3, 2.0)
        dist = Normal(mu, std)
        raw_action = dist.rsample()             # before tanh
        tanh_action = torch.tanh(raw_action) * 2.0  # env range [−2,2]
        return tanh_action, raw_action, dist

class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            orthogonal_init(nn.Linear(in_dim, 64)), nn.Tanh(),
            orthogonal_init(nn.Linear(64, 64)), nn.Tanh(),
            orthogonal_init(nn.Linear(64, 1), gain=1.0),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    #############################
    return returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    indices = np.arange(batch_size)
    
    for _ in range(update_epoch):
        # Shuffle indices for each epoch
        np.random.shuffle(indices)
        
        # Split into mini-batches
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                states[batch_indices],
                actions[batch_indices],
                values[batch_indices],
                log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices]
            )

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.max_steps = args.max_steps if hasattr(args, 'max_steps') else 200000
        
        # device: cpu / gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # Store initial learning rates for decay
        self.actor_lr0 = args.actor_lr
        self.critic_lr0 = args.critic_lr
        
        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        tanh_action, raw_action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else tanh_action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(raw_action)  # Store raw action
            self.values.append(value)
            self.log_probs.append(dist.log_prob(raw_action))
        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""

        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        with torch.no_grad():
            next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        # stack tensors and detach all tensors that don't need gradients
        states = torch.cat(self.states).view(-1, self.obs_dim)
        raw_actions = torch.cat(self.actions).detach()  # These are raw actions
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()

        # Linear learning rate decay
        frac = 1.0 - (self.total_step / self.max_steps)
        for pg in self.actor_optimizer.param_groups:
            pg['lr'] = self.actor_lr0 * frac
        for pg in self.critic_optimizer.param_groups:
            pg['lr'] = self.critic_lr0 * frac

        actor_losses, critic_losses = [], []

        # Reshape advantages to match the batch size
        advantages = advantages.view(-1)

        for state, raw_action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=raw_actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # Actor update
            self.actor_optimizer.zero_grad()
            _, _, dist = self.actor(state)
            log_prob = dist.log_prob(raw_action).sum(-1)
            
            # Calculate ratio without additional clipping
            ratio = (log_prob - old_log_prob).exp()

            # Ensure advantages have the same shape as ratio
            adv = adv.view(-1)

            # PPO surrogate objective with epsilon clipping
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Add entropy bonus
            entropy = dist.entropy().mean()
            actor_loss = actor_loss - self.entropy_weight * entropy

            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            actor_losses.append(actor_loss.item())

            # Critic update
            self.critic_optimizer.zero_grad()
            value = self.critic(state.detach())
            critic_loss = F.smooth_l1_loss(value, return_)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

        # Clear memory
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss


    def test(self, video_folder: str):
        """Test the agent over 20 episodes and calculate average score."""
        self.is_test = True
        num_eval_episodes = 20
        scores = []
        seeds = [0,14,15,23,24,37,42,63,78,80,84,100,103,114,124,135,141,160,178,180]
        
        # Create video recording environment
        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        for episode in range(num_eval_episodes):
            seed = seeds[episode]
            state, _ = self.env.reset(seed=seed)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state
                score += reward

            scores.append(score)
            print(f"Evaluation Episode {episode + 1}: seed {seed}, Score = {score}")

        # Calculate and print average score
        avg_score = sum(scores) / len(scores)
        print(f"\nAverage Score over {num_eval_episodes} episodes: {avg_score}")
        print(f"All scores: {scores}")
        
        # Log results to wandb
        wandb.log({
            "test_avg_score": avg_score,
            "test_scores": scores
        })

        self.env.close()
        self.env = tmp_env
        return avg_score

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        total_steps = 0
        
        # Log hyperparameters
        wandb.config.update({
            "actor_lr": self.actor_optimizer.param_groups[0]['lr'],
            "critic_lr": self.critic_optimizer.param_groups[0]['lr'],
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "entropy_weight": self.entropy_weight,
            "rollout_len": self.rollout_len,
            "update_epoch": self.update_epoch
        })
        
        for ep in tqdm(range(1, self.num_episodes)):
            score = 0
            print("\n")
            for _ in range(self.rollout_len):
                self.total_step += 1
                total_steps += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score}")
                    # Log episode metrics
                    wandb.log({
                        "episode": episode_count,
                        "episode_reward": score,
                        "average_reward": np.mean(scores[-100:]) if len(scores) > 0 else score,
                        "total_steps": total_steps
                    })
                    score = 0

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            # Print and log losses
            print(f"  Actor Loss: {actor_loss:.4f}  Critic Loss: {critic_loss:.4f}")
            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "total_steps": total_steps
            })

            # Save model periodically
            if episode_count % 10 == 0:
                model_dict = {
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict()
                }
                torch.save(model_dict, './task2_results/LAB7_110550161_task2_ppo_pendulum.pt')

        # termination
        self.env.close()

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.1)
    parser.add_argument("--rollout-len", type=int, default=1024)  
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200000)
    args = parser.parse_args() 
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    # Initialize wandb
    wandb.init(
        project="DLP-Lab7-PPO-Pendulum", 
        name=args.wandb_run_name, 
        save_code=True,
        config={
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "discount_factor": args.discount_factor,
            "num_episodes": args.num_episodes,
            "entropy_weight": args.entropy_weight,
            "tau": args.tau,
            "batch_size": args.batch_size,
            "epsilon": args.epsilon,
            "rollout_len": args.rollout_len,
            "update_epoch": args.update_epoch
        }
    )
    
    # Create and train agent
    agent = PPOAgent(env, args)
    agent.train()
    
    # Test the trained agent
    print("\nStarting evaluation...")
    avg_score = agent.test("task2_results")
    print(f"\nFinal evaluation average score: {avg_score}")
    
    # Close wandb
    wandb.finish()