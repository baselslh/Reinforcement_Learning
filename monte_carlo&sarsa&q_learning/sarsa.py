import sys
import gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


class SARSAagent:
    """SARSA agent that can act on a continuous state space by discretizing it."""

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
        # Environment info
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
        self.epsilon_decay_rate = epsilon_decay_rate

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

    def choose_action(self, state, mode="train"):
        """Pick next action and update internal Q table (when mode != 'test')."""
        if mode == "test":
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
            return action
        # Exploration vs. exploitation
        do_exploration = np.random.uniform(0, 1) < self.epsilon
        if do_exploration:
            # Pick a random action
            action = np.random.randint(0, self.action_size)
        else:
            # Pick the best action from Q table
            action = np.argmax(self.q_table[state])
        return action

    def update_q_table(self, reward, next_state, next_action):
        """Update Q table with new knowledge from last action."""
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        self.q_table[self.last_state + (self.last_action,)] += (self.alpha) * (
            reward
            + self.gamma * self.q_table[next_state + (next_action,)]
            - self.q_table[self.last_state + (self.last_action,)]
        )

    def run(self, num_episodes, mode="train"):
        """Run agent for given number of episodes."""
        if mode == "train":
            self.reset_exploration()

        scores = []
        for i_episode in range(1, num_episodes + 1):
            self.last_state = self.env.reset(seed=self.seed)
            self.last_action = self.reset_episode(self.last_state)
            score = 0
            while True:
                if mode == "test" and i_episode >= num_episodes - 1:
                    self.env.render()
                state_new, reward, done, _ = self.env.step(self.last_action)
                state_new = self.preprocess_state(state_new)
                action_new = self.choose_action(state_new, mode)
                if mode == "train":
                    self.update_q_table(reward, state_new, action_new)

                # Roll over current state, action for next step
                self.last_state = state_new
                self.last_action = action_new

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
        """Calculate accuracy of agent for given number of episodes."""
        count_fail = 0
        for _ in range(1, num_episodes + 1):
            self.last_state = self.env.reset(seed=self.seed)
            self.last_action = self.reset_episode(self.last_state)
            score = 0
            info = {"TimeLimit.truncated": False}
            while True:
                state_new, reward, done, info = self.env.step(self.last_action)
                state_new = self.preprocess_state(state_new)
                action_new = self.choose_action(state_new, "test")
                # Roll over current state, action for next step
                self.last_state = state_new
                self.last_action = action_new

                score += reward  # accumulate reward
                if info.get("TimeLimit.truncated"):
                    count_fail += 1
                if done:
                    break
        accuracy = (num_episodes - count_fail) / num_episodes
        return accuracy
