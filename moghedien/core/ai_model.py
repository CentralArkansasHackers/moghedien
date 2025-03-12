import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class AttackPathEnvironment:
    """
    Reinforcement learning environment for attack path selection.
    Represents the AD environment as a state and defines actions for pathfinding.
    """

    def __init__(self, graph):
        """
        Initialize the environment with an AD graph.

        Args:
            graph: ADGraph instance representing the Active Directory
        """
        self.graph = graph
        self.nx_graph = graph.graph

        # Define action space based on relationship types
        self.actions = self._extract_action_space()
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.action_space_size = len(self.actions)

        # State tracking
        self.current_node = None
        self.target_node = None
        self.visited_nodes = []
        self.path = []

        # For tracking steps and rewards
        self.max_path_length = 10  # Maximum path length
        self.step_count = 0

        logger.info(f"Environment initialized with {self.action_space_size} actions")

    def _extract_action_space(self) -> List[str]:
        """
        Extract possible actions (relationship types) from the graph.

        Returns:
            List of relationship types that can be used as actions
        """
        # Collect all unique relationship types from edges
        actions = set()
        for _, _, data in self.nx_graph.edges(data=True):
            if 'relationship' in data:
                actions.add(data['relationship'])

        # Convert to list and sort for deterministic ordering
        return sorted(list(actions))

    def reset(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """
        Reset the environment to start a new episode.

        Args:
            source_id: Starting node ID
            target_id: Target node ID

        Returns:
            The initial state observation
        """
        if not (self.nx_graph.has_node(source_id) and self.nx_graph.has_node(target_id)):
            raise ValueError(f"Invalid source or target node")

        self.current_node = source_id
        self.target_node = target_id
        self.visited_nodes = [source_id]
        self.path = [source_id]
        self.step_count = 0

        return self._get_observation()

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by performing an action.

        Args:
            action_idx: Index of the action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Check if the action is valid
        if action_idx < 0 or action_idx >= self.action_space_size:
            logger.warning(f"Invalid action index: {action_idx}")
            return self._get_observation(), -1.0, True, {"error": "Invalid action"}

        self.step_count += 1
        action = self.actions[action_idx]

        # Find neighbors connected by the chosen relationship
        neighbors = []
        for _, neighbor, data in self.nx_graph.out_edges(self.current_node, data=True):
            if data.get('relationship') == action:
                neighbors.append((neighbor, data))

        # If no valid neighbors for this action, penalize
        if not neighbors:
            return self._get_observation(), -0.5, False, {"valid_move": False}

        # Choose the best neighbor (could be random or heuristic-based)
        # For simplicity, choose the first one for now
        next_node, edge_data = neighbors[0]

        # Check for cycles
        if next_node in self.visited_nodes:
            return self._get_observation(), -0.3, False, {"cycle_detected": True}

        # Update state
        self.current_node = next_node
        self.visited_nodes.append(next_node)
        self.path.append(next_node)

        # Check if reached target
        reached_target = (next_node == self.target_node)

        # Check if max steps reached
        max_steps_reached = (self.step_count >= self.max_path_length)

        # Determine if episode is done
        done = reached_target or max_steps_reached

        # Calculate reward
        reward = self._calculate_reward(next_node, edge_data, reached_target, max_steps_reached)

        # Return observation, reward, done flag, and additional info
        return self._get_observation(), reward, done, {
            "reached_target": reached_target,
            "max_steps_reached": max_steps_reached,
            "path_length": len(self.path) - 1
        }

    def _get_observation(self) -> Dict[str, Any]:
        """
        Get the current state observation.

        Returns:
            Dictionary representing the state
        """
        # Basic observation with current position and target
        observation = {
            "current_node": self.current_node,
            "target_node": self.target_node,
            "current_node_type": self.nx_graph.nodes[self.current_node].get('type', ''),
            "target_node_type": self.nx_graph.nodes[self.target_node].get('type', ''),
            "path_length": len(self.path) - 1,
            "visited_nodes": self.visited_nodes,
        }

        # Add information about available actions from current node
        available_actions = []
        for action_idx, action in enumerate(self.actions):
            count = 0
            for _, _, data in self.nx_graph.out_edges(self.current_node, data=True):
                if data.get('relationship') == action:
                    count += 1

            if count > 0:
                available_actions.append({
                    "action_idx": action_idx,
                    "action": action,
                    "count": count
                })

        observation["available_actions"] = available_actions

        return observation

    def _calculate_reward(self, node: str, edge_data: Dict[str, Any],
                          reached_target: bool, max_steps_reached: bool) -> float:
        """
        Calculate the reward for the current step.

        Args:
            node: The new node reached
            edge_data: Edge data for the transition
            reached_target: Whether the target was reached
            max_steps_reached: Whether max steps were reached

        Returns:
            The reward value
        """
        if reached_target:
            # Reward for reaching target is inverse to path length
            base_reward = 10.0  # Base reward for success
            length_factor = 1.0 / len(self.path)  # Shorter paths get higher rewards
            return base_reward + (length_factor * 5.0)

        if max_steps_reached:
            return -1.0  # Penalty for not reaching target within max steps

        # Calculate step reward based on edge properties
        step_reward = 0.0

        # Reward for approaching target (could use a heuristic or graph distance)
        # For now, use a simple approximation
        if node not in self.visited_nodes[:-1]:  # If not revisiting
            step_reward += 0.1  # Small reward for exploration

        # Weight based on edge properties
        if 'weight' in edge_data:
            # Lower weight edges are more valuable (easier/less risky paths)
            step_reward += (1.0 - edge_data['weight']) * 0.3

        return step_reward


class PathFinderModel(nn.Module):
    """
    Neural network model for the path finding agent.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the model.

        Args:
            input_size: Size of the input features
            hidden_size: Size of the hidden layer
            output_size: Size of the output (action space)
        """
        super(PathFinderModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class PathFinderAgent:
    """
    Reinforcement learning agent for finding attack paths.
    """

    def __init__(self, environment, input_size=100, hidden_size=128, lr=0.001, gamma=0.99):
        """
        Initialize the agent.

        Args:
            environment: The AttackPathEnvironment instance
            input_size: Size of the state representation
            hidden_size: Size of hidden layers in the model
            lr: Learning rate
            gamma: Discount factor
        """
        self.env = environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = PathFinderModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=self.env.action_space_size
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training parameters
        self.gamma = gamma
        self.epsilon = 1.0  # For epsilon-greedy exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64

        # Experience replay buffer
        self.memory = []
        self.max_memory_size = 10000

        logger.info(f"Agent initialized with model: {self.model}")
        logger.info(f"Using device: {self.device}")

    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert a state dictionary to a tensor representation.

        Args:
            state: The state observation from the environment

        Returns:
            Tensor representation of the state
        """
        # This is a simplified representation and should be enhanced
        # to better capture the AD graph structure
        features = []

        # One-hot encode node types
        node_types = ['User', 'Computer', 'Group', 'Domain', 'OU', 'GPO', 'Container']
        current_type = state["current_node_type"]
        target_type = state["target_node_type"]

        for nt in node_types:
            features.append(1.0 if current_type == nt else 0.0)
            features.append(1.0 if target_type == nt else 0.0)

        # Path length (normalized)
        features.append(min(state["path_length"] / 10.0, 1.0))

        # Available actions (simplified)
        action_counts = [0.0] * self.env.action_space_size
        for action_info in state["available_actions"]:
            action_counts[action_info["action_idx"]] = min(action_info["count"] / 5.0, 1.0)

        features.extend(action_counts)

        # Pad to fixed size if needed
        while len(features) < 100:
            features.append(0.0)

        # Truncate if too large
        features = features[:100]

        return torch.FloatTensor(features).unsqueeze(0).to(self.device)

    def get_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select an action based on the current state.

        Args:
            state: The state observation
            training: Whether to use epsilon-greedy policy

        Returns:
            The selected action index
        """
        # Available actions
        available_actions = [a["action_idx"] for a in state["available_actions"]]

        if not available_actions:
            # If no actions available, return a random action
            # This should be handled by the environment
            return np.random.randint(0, self.env.action_space_size)

        # Epsilon-greedy during training
        if training and np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)

        # Convert state to tensor
        state_tensor = self._state_to_tensor(state)

        # Get action values from model
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # Filter for available actions only
        masked_q_values = np.ones_like(q_values) * float('-inf')
        masked_q_values[available_actions] = q_values[available_actions]

        return np.argmax(masked_q_values)

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Limit memory size
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Train the model on batches from replay memory.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in batch:
            states.append(self._state_to_tensor(self.memory[i][0]))
            actions.append(self.memory[i][1])
            rewards.append(self.memory[i][2])
            next_states.append(self._state_to_tensor(self.memory[i][3]))
            dones.append(self.memory[i][4])

        # Convert to tensors
        states = torch.cat(states, dim=0)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        curr_q = self.model(states).gather(1, actions).squeeze(1)

        # Next Q values
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(curr_q, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def train(self, num_episodes=1000, max_steps=10):
        """
        Train the agent.

        Args:
            num_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode

        Returns:
            Training history
        """
        history = []

        for episode in range(num_episodes):
            # Select random source and target
            nodes = list(self.env.nx_graph.nodes())
            source = np