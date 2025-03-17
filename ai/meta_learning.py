import gym
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Callable, Tuple
import threading
import time
import random
from collections import deque

class MetaLearningAgent:
    """An agent that can improve its own learning algorithms."""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 meta_learning_rate: float = 0.001,
                 memory_size: int = 10000):
        """Initialize meta-learning agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            meta_learning_rate: Learning rate for meta-model
            memory_size: Size of experience replay buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.meta_learning_rate = meta_learning_rate
        
        # Create primary model (policy network)
        self.primary_model = self._build_model()
        
        # Create meta-model (for optimizing hyperparameters)
        self.meta_model = self._build_meta_model()
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Hyperparameters that will be dynamically adjusted
        self.hyperparams = {
            "learning_rate": 0.001,
            "gamma": 0.95,  # discount factor
            "epsilon": 1.0,  # exploration rate
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "batch_size": 32
        }
        
        # Meta-learning memory (stores performance with different hyperparameters)
        self.meta_memory = []
        
        # Performance tracking
        self.performance_history = []
        
    def _build_model(self):
        """Build neural network model for Q-learning."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparams["learning_rate"])
        )
        return model
    
    def _build_meta_model(self):
        """Build meta-model for hyperparameter optimization."""
        # Input: current hyperparameters and recent performance metrics
        meta_input = tf.keras.layers.Input(shape=(len(self.hyperparams) + 3,))
        
        # Hidden layers
        x = tf.keras.layers.Dense(32, activation='relu')(meta_input)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        
        # Output: adjustments to hyperparameters
        meta_output = tf.keras.layers.Dense(len(self.hyperparams), activation='tanh')(x)
        
        # Create model
        model = tf.keras.Model(inputs=meta_input, outputs=meta_output)
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.meta_learning_rate)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        if np.random.rand() <= self.hyperparams["epsilon"]:
            return random.randrange(self.action_size)
        
        act_values = self.primary_model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train model on batch of experiences."""
        if len(self.memory) < self.hyperparams["batch_size"]:
            return 0
        
        # Sample batch from memory
        batch_size = min(self.hyperparams["batch_size"], len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare training data
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.hyperparams["gamma"] * np.amax(
                    self.primary_model.predict(next_state)[0]
                )
            
            target_f = self.primary_model.predict(state)
            target_f[0][action] = target
            
            states[i] = state
            targets[i] = target_f[0]
        
        # Train model
        history = self.primary_model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decay epsilon
        if self.hyperparams["epsilon"] > self.hyperparams["epsilon_min"]:
            self.hyperparams["epsilon"] *= self.hyperparams["epsilon_decay"]
            
        return loss
    
    def optimize_hyperparams(self, episode_reward):
        """Use meta-learning to optimize hyperparameters."""
        # Store performance with current hyperparameters
        self.meta_memory.append({
            "hyperparams": self.hyperparams.copy(),
            "reward": episode_reward
        })
        
        # Need enough data to optimize
        if len(self.meta_memory) < 5:
            return
        
        # Prepare input for meta-model
        # [current hyperparams, avg_reward, reward_trend, reward_variance]
        recent_rewards = [entry["reward"] for entry in self.meta_memory[-10:]]
        avg_reward = np.mean(recent_rewards)
        reward_trend = np.mean(np.diff(recent_rewards)) if len(recent_rewards) > 1 else 0
        reward_variance = np.var(recent_rewards)
        
        meta_input = np.array([[
            *list(self.hyperparams.values()),
            avg_reward,
            reward_trend,
            reward_variance
        ]])
        
        # Get hyperparameter adjustments from meta-model
        adjustments = self.meta_model.predict(meta_input)[0]
