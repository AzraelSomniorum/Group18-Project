"""
This module wraps Gymnasium's Pendulum-v1 environment and provides:
1. State space discretization (continuous -> discrete bins)
2. Action space discretization (5 discrete actions -> continuous torque)
3. Pygame visualization
"""

import gymnasium as gym
import numpy as np
import pygame
from enum import Enum


class PendulumAction(Enum):
    """
    Discrete action enum for Pendulum control.
    Maps 5 discrete actions to continuous torque values.
    """
    STRONG_LEFT = 0   # Torque: -2.0 (push hard left)
    WEAK_LEFT = 1     # Torque: -1.0 (push gently left)
    NONE = 2          # Torque:  0.0 (no action)
    WEAK_RIGHT = 3    # Torque:  1.0 (push gently right)
    STRONG_RIGHT = 4  # Torque:  2.0 (push hard right)


class PendulumAgent:
    """
    Problem model for the Pendulum balancing task.
    
    This class wraps the continuous Pendulum-v1 environment and provides
    discretization for both state and action spaces, making it suitable
    for tabular Q-Learning.
    
    State Space (Discretized):
        - Angle bins: 20 bins covering [-π, π]
        - Velocity bins: 20 bins covering [-8, 8] rad/s
        - Total states: 20 * 20 = 400
    
    Action Space (Discretized):
        - 5 discrete actions
    """
    
    def __init__(self, fps=30):
        """
        Initialize the Pendulum agent.
        
        Args:
            fps (int): Frames per second for pygame rendering
        """
        self.env = gym.make('Pendulum-v1')
        
        # Discretization parameters
        self.angle_bins = 20
        self.velocity_bins = 20
        
        # Define bin edges for discretization
        self.angle_edges = np.linspace(-np.pi, np.pi, self.angle_bins + 1)
        self.velocity_edges = np.linspace(-8.0, 8.0, self.velocity_bins + 1)
        
        # Action to torque mapping
        self.action_to_torque_map = {
            PendulumAction.STRONG_LEFT: -2.0,
            PendulumAction.WEAK_LEFT: -1.0,
            PendulumAction.NONE: 0.0,
            PendulumAction.WEAK_RIGHT: 1.0,
            PendulumAction.STRONG_RIGHT: 2.0
        }
        
        # Pygame setup for visualization
        self.fps = fps
        self.screen = None
        self.clock = None
        
    def reset(self, seed=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            np.ndarray: Discretized state [angle_bin, velocity_bin]
        """
        if seed is not None:
            self.env.reset(seed=seed)
        else:
            self.env.reset()
        
        # Get continuous state from environment
        continuous_state, _ = self.env.reset(seed=seed) if seed else self.env.reset()
        
        # Convert to discrete state
        discrete_state = self._discretize_state(continuous_state)
        
        return discrete_state
    
    def perform_action(self, action: PendulumAction):
        """
        Execute a discrete action in the environment.
        
        Args:
            action (PendulumAction): Discrete action to perform
            
        Returns:
            tuple: (discrete_state, reward, terminated, truncated)
                - discrete_state: np.ndarray of shape (2,)
                - reward: float
                - terminated: bool
                - truncated: bool
        """
        # Convert discrete action to continuous torque
        torque = self._action_to_torque(action)
        
        # Execute action in base environment (expects array)
        continuous_state, reward, terminated, truncated, _ = self.env.step([torque])
        
        # Discretize the resulting state
        discrete_state = self._discretize_state(continuous_state)
        
        return discrete_state, reward, terminated, truncated
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
        if self.screen is not None:
            pygame.quit()
    
    def _discretize_state(self, continuous_state):
        """
       Convert continuous state to discrete bins.
        
        Args:
            continuous_state: np.ndarray [cos(θ), sin(θ), θ̇]
            
        Returns:
            np.ndarray: [angle_bin, velocity_bin] where each is in [0, 19]
        """
        # Extract cos(θ), sin(θ), and angular velocity
        cos_theta, sin_theta, thetadot = continuous_state
        
        # Reconstruct angle from cos and sin
        theta = np.arctan2(sin_theta, cos_theta)  # Result in [-π, π]
        
        # Discretize angle
        angle_bin = np.digitize(theta, self.angle_edges) - 1
        angle_bin = np.clip(angle_bin, 0, self.angle_bins - 1)
        
        # Discretize velocity
        velocity_bin = np.digitize(thetadot, self.velocity_edges) - 1
        velocity_bin = np.clip(velocity_bin, 0, self.velocity_bins - 1)
        
        return np.array([angle_bin, velocity_bin], dtype=np.int32)
    
    def _action_to_torque(self, action: PendulumAction):
        """
        Convert discrete action to continuous torque value.
        
        Args:
            action (PendulumAction): Discrete action enum
            
        Returns:
            float: Continuous torque value in [-2.0, 2.0]
        """
        return self.action_to_torque_map[action]


# Test the agent if run directly
if __name__ == "__main__":
    print("Testing PendulumAgent...")
    
    agent = PendulumAgent()
    
    # Test reset
    state = agent.reset(seed=42)
    print(f"Initial state: {state}")
    print(f"State shape: {state.shape}")
    print(f"State dtype: {state.dtype}")
    
    # Test actions
    print("\nTesting all actions:")
    for action in PendulumAction:
        state, reward, term, trunc = agent.perform_action(action)
        print(f"{action.name:15} -> State: {state}, Reward: {reward:.2f}")
    
    # Test episode
    print("\nRunning 10-step episode:")
    state = agent.reset()
    for step in range(10):
        action = PendulumAction(step % 5)  # Cycle through actions
        state, reward, term, trunc = agent.perform_action(action)
        print(f"Step {step+1}: State={state}, Reward={reward:.2f}")
        if term or trunc:
            break
    
    agent.close()
    print("\nPendulumAgent test complete!")
