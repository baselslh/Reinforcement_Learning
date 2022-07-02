import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(
        self,
        env,
        state_grid,
        alpha=0.02,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay_rate=0.9995,
        min_epsilon=0.01,
        seed=505,
    ):
        """Initialize variables, create grid for discretization."""
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)
        self.action_size = self.env.action_space.n
        self.seed = np.random.seed(seed)
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)

        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = (
            epsilon_decay_rate  # how quickly should we decrease epsilon
        )
        self.min_epsilon = min_epsilon  # minimum exploration rate

        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        print("Q table size:", self.q_table.shape)

    def discretize(self, sample):
        """Discretize a sample as per given grid."""
        return list(int(np.digitize(s, g)) for s, g in zip(sample, self.state_grid))

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        return tuple(self.discretize(state))

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def act(self, state, reward=None, done=None, mode="train"):
        """Pick next action and update internal Q table (when mode != 'test')."""
        state = self.preprocess_state(state)
        if mode == "test":
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            # update q table (Q(s,a) <- Q(s,a) + alpha[r + gamma max Q(s',a) - Q(s,a)]
            self.q_table[self.last_state + (self.last_action,)] += self.alpha * (
                reward
                + self.gamma * max(self.q_table[state])
                - self.q_table[self.last_state + (self.last_action,)]
            )

            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

    def run(self, num_episodes, mode="train"):
        """Run agent for given number of episodes."""
        if mode == "train":
            self.reset_exploration()

        scores = []
        for i_episode in range(1, num_episodes + 1):
            state = self.env.reset(seed=self.seed)
            action = self.reset_episode(state)
            score = 0
            while True:
                state, reward, done, _ = self.env.step(action)
                action = self.act(state, reward, done, mode)
                if mode == "test" and i_episode >= num_episodes - 1:
                    self.env.render()
                score += reward  # accumulate reward
                if done:
                    break
            scores.append(score)
            print(
                "\rEpisode {}/{} | Average Score: {:.2f}".format(
                    i_episode, num_episodes, np.mean(scores)
                ),
                end="",
            )
            sys.stdout.flush()
            if mode == "train" and i_episode % 100 == 0:
                print(
                    "\rEpisode {}/{} | Average Score: {:.2f}".format(
                        i_episode, num_episodes, np.mean(scores)
                    )
                )
                sys.stdout.flush()
        return scores

    def calculate_accuracy(self, num_episodes):
        """Calculate accuracy of agent's policy."""
        count_failed = 0
        for _ in range(1, num_episodes + 1):
            state = self.env.reset(seed=self.seed)
            action = self.reset_episode(state)
            score = 0
            info = {"TimeLimit.truncated": False}
            while True:
                state, reward, done, info = self.env.step(action)
                action = self.act(state, reward, done, mode="test")
                score += reward  # accumulate reward
                if info.get("TimeLimit.truncated"):
                    count_failed += 1
                if done:
                    break

        accuracy = (num_episodes - count_failed) / num_episodes
        return accuracy
