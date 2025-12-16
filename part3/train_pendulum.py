'''
Q-Learning Training Script for Pendulum Agent
Implements the training loop, agent logic, and evaluation procedures.
'''
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from abc import ABC, abstractmethod

# Import the custom environment
import pendulum_env

# Action Selection Strategies ----- Polymorphism
class ExplorationStrategy(ABC): # Template
    """
    Abstract interface for action selection.
    Subclasses define specific behaviors for training (exploration) vs evaluation (exploitation).
    """
    @abstractmethod
    def select_action(self, q_values, action_space):
        pass

class EpsilonGreedy(ExplorationStrategy):
    """
    Implements epsilon-greedy strategy: random action with probability epsilon,
    otherwise selects the action with the highest Q-value.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon # Stores the probability (e.g., 0.1 or 10%)

    def select_action(self, q_values, action_space):
        # Generate a random number between 0.0 and 1.0. 
        # If it is less than epsilon (e.g., 0.1), we explore.
        if np.random.random() < self.epsilon:
            return action_space.sample() # Return a random action (0-4)
        return np.argmax(q_values) # Otherwise find the index of the highest value in the Q-table row.

class Greedy(ExplorationStrategy): # Eval
    """
    Implements pure greedy strategy: always selects the action with the highest Q-value.
    Used primarily for evaluating the trained agent.
    """
    def select_action(self, q_values, action_space):
        return np.argmax(q_values) # Always pick best action

# Agent Interface ------- Abstraction
class BaseAgent(ABC):
    """
    Abstract base class defining the required methods for a Reinforcement Learning agent.
    Enforces a consistent interface for acting, updating, and persistence.
    """
    @abstractmethod
    def act(self, state, strategy: ExplorationStrategy):
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state):
        pass
    
    @abstractmethod
    def save(self, filepath):
        pass
    
    @abstractmethod
    def load(self, filepath):
        pass

# Q-Learning Implementation --------------------------------
class QLearningAgent(BaseAgent): # Inherits from base agent
    """
    Tabular Q-Learning agent implementation.
    Manages the Q-table and applies the Bellman update rule.
    """
    def __init__(self, state_shape, num_actions, alpha=0.9, gamma=0.95):
        # Initialize Q-table with zeros
        # Shape: (angle_bins, velocity_bins, num_actions), 20 x 20 x 5
        self._q_table = np.zeros(state_shape + (num_actions,)) # Encapsulation
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.num_actions = num_actions
        
    def act(self, state, strategy: ExplorationStrategy):
        """
        Delegates action selection to the provided strategy.
        """
        # state is expected to be [angle_bin, velocity_bin], fetch values
        angle_bin, velocity_bin = state
        q_values = self._q_table[angle_bin, velocity_bin, :]
        
        # Simple wrapper to provide a .sample() interface for the strategy
        class ActionSpace:
            def __init__(self, n): self.n = n
            def sample(self): return np.random.randint(0, self.n)
            
        return strategy.select_action(q_values, ActionSpace(self.num_actions)) # Choose exploration strategy

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value for a given state-action pair using the Bellman equation.
        """
        angle, vel = state
        next_angle, next_vel = next_state
        
        current_q = self._q_table[angle, vel, action] # Get the current guess (Old Q)
        max_next_q = np.max(self._q_table[next_angle, next_vel, :]) # Find the best possible score from the NEXT state (Max Q')
        
        # Calculate the Target (Reward + discounted future)
        target = reward + self.gamma * max_next_q
        
        # Update the table (move Old Q slightly towards Target)
        self._q_table[angle, vel, action] += self.alpha * (target - current_q)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self._q_table, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
        with open(filepath, 'rb') as f:
            self._q_table = pickle.load(f)
        print(f"Model loaded from {filepath}")

# Experiment Management ----------------------------------
class ExperimentRunner:
    """
    Orchestrates the training and evaluation loops.
    Handles environment interaction, metrics tracking, and visualization.
    """
    def __init__(self, env_id, episodes, is_training, render_mode=None):
        self.env_id = env_id
        self.episodes = episodes
        self.is_training = is_training
        self.render_mode = render_mode
        self.rewards_history = []
        
        # Training hyperparameters
        self.epsilon_start = 1.0 # Start at 1.0 (100% random), go down to 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay = 2.0 / episodes

    def run(self):
        # Initialize environment
        env = gym.make(self.env_id, render_mode=self.render_mode)
        
        # Initialize Agent (20x20 bins are hardcoded to match the Env)
        state_shape = (20, 20) 
        num_actions = env.action_space.n
        agent = QLearningAgent(state_shape, num_actions)
        
        # Load existing model if we are running in evaluation mode
        model_path = 'pendulum_q_table.pkl'
        if not self.is_training:
            try:
                agent.load(model_path)
            except FileNotFoundError:
                print("Error: Model file not found. Please train the agent first.")
                return

        current_epsilon = self.epsilon_start
        
        mode = "Training" if self.is_training else "Evaluation"
        print(f"Starting {mode} for {self.episodes} episodes...")

        # Main Episode Loop
        for episode in range(self.episodes):
            state, _ = env.reset() # Restart Pendulum
            state = tuple(state.astype(int)) # Ensure state is a tuple of integers for array indexing
            
            terminated = False
            truncated = False
            total_reward = 0
            
            # Select appropriate strategy for the current mode
            if self.is_training:
                strategy = EpsilonGreedy(current_epsilon)
            else:
                strategy = Greedy()

            while not (terminated or truncated):
                # Select action
                action = agent.act(state, strategy)
                
                # Execute step
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = tuple(next_state.astype(int))
                
                # Update agent if in training mode
                if self.is_training:
                    agent.update(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward

            # Track metrics
            self.rewards_history.append(total_reward)
            
            # Decay epsilon
            if self.is_training:
                current_epsilon = max(self.epsilon_min, current_epsilon - self.epsilon_decay)
            
            # Progress logging
            if (episode + 1) % 100 == 0:
                avg_r = np.mean(self.rewards_history[-100:])
                print(f"Episode {episode+1}: Avg Reward = {avg_r:.2f} | Epsilon = {current_epsilon:.3f}")

        env.close()
        
        # Persist model
        if self.is_training:
            agent.save(model_path)
        
        self._plot_results()

    def _plot_results(self):
        """Generates and saves performance plots."""
        plt.figure(figsize=(12, 5))
        
        # Reward Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_history, alpha=0.3, color='blue', label='Episode Reward')
        
        # Moving Average
        window = 100
        if len(self.rewards_history) >= window:
            moving_avg = np.convolve(self.rewards_history, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.rewards_history)), moving_avg, color='red', label='100-Ep Avg')
            
        plt.title('Training Progress' if self.is_training else 'Evaluation Results')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        # Histogram
        plt.subplot(1, 2, 2)
        plt.hist(self.rewards_history, bins=50, color='green', alpha=0.7)
        plt.title('Reward Distribution')
        
        plot_name = 'pendulum_training.png' if self.is_training else 'pendulum_evaluation.png'
        plt.savefig(plot_name)
        print(f"Plot saved to {plot_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pendulum Q-Learning Agent")
    parser.add_argument('--train', action='store_true', help='Run in training mode')
    parser.add_argument('--episodes', type=int, default=10, help='Total number of episodes')
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    
    args = parser.parse_args()
    
    # Environment ID must match registration in pendulum_env.py
    ENV_ID = 'pendulum-discrete-v0' 
    
    runner = ExperimentRunner(
        env_id=ENV_ID, 
        episodes=args.episodes, 
        is_training=args.train, 
        render_mode='human' if args.render else None
    )
    runner.run()