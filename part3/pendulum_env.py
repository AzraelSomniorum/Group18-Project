"""

====================================================
This module creates a Gymnasium-compatible environment that wraps PendulumAgent.
It provides a standard Gym interface with discrete spaces.

OOP Principles Demonstrated:
- Inheritance: Extends gym.Env base class
- Abstraction: Hides PendulumAgent implementation details
- Polymorphism: Can be used anywhere a gym.Env is expected
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pendulum_agent import PendulumAgent, PendulumAction


class PendulumEnv(gym.Env):
    """
    Custom Gymnasium environment for discretized Pendulum.
    
    This environment wraps the PendulumAgent and provides a standard
    Gym interface with discrete observation and action spaces.
    
    Observation Space:
        Box(low=[0, 0], high=[19, 19], dtype=int32)
        - [0]: Angle bin (0-19)
        - [1]: Angular velocity bin (0-19)
    
    Action Space:
        Discrete(5)
        - 0: STRONG_LEFT
        - 1: WEAK_LEFT
        - 2: NONE
        - 3: WEAK_RIGHT
        - 4: STRONG_RIGHT
    
    Reward:
        reward = -(theta^2 + 0.1*theta^2 + 0.001*torque²)
        Range: roughly [-16, 0]
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }
    
    def __init__(self, render_mode=None):
        """
        Initialize the Pendulum environment.
        
        Args:
            render_mode (str, optional): "human" for visualization, None for no rendering
        """
        super().__init__()
        
        # Create the problem model (Layer 1)
        self.agent = PendulumAgent(fps=30)
        
        # Define discrete action space (5 actions)
        # Note for implementation: (in Pendulum Agent): see action space in the beginning of this class
        self.action_space = spaces.Discrete(5)
        
        # discrete observation space (20x20 bins)
        # according to the warning in space.py, the action spaces should use predefined space classes. Here we use spaces.Box
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([19, 19], dtype=np.int32),
            dtype=np.int32
        )
        
        # Rendering mode
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options (unused)
            
        Returns:
            tuple: (observation, info)
                - observation: np.ndarray of shape (2,)
                - info: dict (empty)
        """
        # Set seed if provided
        super().reset(seed=seed)
        
        # Reset the agent
        observation = self.agent.reset(seed=seed)
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        return observation, {}
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): Discrete action ID (0-4)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: np.ndarray of shape (2,)
                - reward: float
                - terminated: bool (always False for Pendulum)
                - truncated: bool (True after 200 steps)
                - info: dict (empty)
        """
        # Convert action ID to PendulumAction enum
        pendulum_action = PendulumAction(action)
        
        # Execute action through the agent
        observation, reward, terminated, truncated = self.agent.perform_action(pendulum_action)
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, {}
    
    def render(self):
        """
        Render the current state.
        
        Returns:
            np.ndarray or None: RGB array if render_mode="rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None
        
        return self.agent.render()
    
    def close(self):
        """Close the environment and cleanup resources."""
        self.agent.close()


# Register the environment with Gymnasium
from gymnasium.envs.registration import register

register(
    id='pendulum-discrete-v0',
    entry_point='pendulum_env:PendulumEnv',
    max_episode_steps=200,  # standard
)


# Test the environment 
if __name__ == "__main__":
    print("Test of PendulumEnv...")
    
    # Test environment creation via gym.make()
    env = gym.make('pendulum-discrete-v0')
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max episode steps: {env.spec.max_episode_steps}")
    
    # reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation: {obs}")
    print(f"Observation in space: {env.observation_space.contains(obs)}")
    
    # episode
    print("\nRunning 10-step episode:")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: Action={action}, Obs={obs}, Reward={reward:.2f}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    
    env.close()
    print("\nPendulumEnv test complete!")
