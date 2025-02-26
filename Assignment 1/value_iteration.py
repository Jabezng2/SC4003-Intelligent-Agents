import numpy as np
import copy
import pandas as pd
import pprint
import os
from config import *

class Grid:
    def __init__(self, tile_reward=TILE_REWARD, map=None, size=None):
        self.tile_reward = tile_reward
        
        # Initialize the grid map if not provided, otherwise use the given map
        if map is None:
            self.map = [["Green", "Wall", "Green", "White", "White", "Green"],
                        ["White", "Brown", "White", "Green","Wall","Brown"],
                        ["White","White", "Brown", "White", "Green","White"],
                        ["White", "White", "White", "Brown", "White", "Green"],
                        ["White", "Wall","Wall","Wall","Brown","White"],
                        ["White","White","White","White","White","White"]]
        else:
            self.map = map
        
        # Initialize value map with zeros, defaulting to a 6x6 grid
        if size is None:
            self.values_map = np.zeros((6,6))
        else:
            self.values_map = np.zeros((size, size))

        # Store all non-wall states in the environment
        self.states = [(c, r) for r, row in enumerate(self.map) for c, tile in enumerate(row) if tile != "Wall"]
    
    def step(self, pos, action):
        x_, y_ = copy.deepcopy(pos)

        # If the action is invalid, return the current position and its reward
        if self.is_valid_action(x_, y_, action) is False:
            return pos, self.get_reward(pos)

        # Update position based on action
        if action == ACTIONS["UP"]:
            y_ -= 1
        elif action == ACTIONS["DOWN"]:
            y_ += 1
        elif action == ACTIONS["LEFT"]:
            x_ -= 1
        elif action == ACTIONS["RIGHT"]:
            x_ += 1

        reward = self.get_reward((x_, y_))
        next_pos = (x_, y_)

        return next_pos, reward
    
    def get_reward(self, pos: tuple):
        x, y = pos
        return self.tile_reward[self.map[y][x]]  # Return reward associated with the tile
    
    def is_valid_action(self, x, y, action):
        x_, y_ = x, y

        # Update position based on action
        if action == ACTIONS["UP"]:
            y_ -= 1
        elif action == ACTIONS["DOWN"]:
            y_ += 1
        elif action == ACTIONS["LEFT"]:
            x_ -= 1
        elif action == ACTIONS["RIGHT"]:
            x_ += 1

        # Check if movement is out of bounds or into a wall
        if len(self.map) <= y_ or y_ < 0:
            return False
        elif len(self.map[0]) <= x_ or x_ < 0:
            return False
        elif self.map[y_][x_] == "Wall":
            return False

        return True

class Agent():
    def __init__(self, env, gamma, policy={}, actions=ACTIONS):
        self.actions = actions
        self.gamma = gamma
        
        # Initialize policy with None values
        if not policy:
            for s in env.states:
                policy[s] = None
        self.policy = policy
        
        self.env = env
        self.value_history = {}
        
        # Store value history for each state
        for s in env.states:
            self.value_history[s] = []
        
        self.v = self.env.values_map  # Reference to value map

    def get_action_probs(self, action):
        # Define probability distribution for each action, considering unintended deviations
        if action == 0:
            return [0, 2, 3], [0.8, 0.1, 0.1]
        if action == 1:
            return [1, 2, 3], [0.8, 0.1, 0.1]
        if action == 2:
            return [2, 0, 1], [0.8, 0.1, 0.1]
        if action == 3:
            return [3, 0, 1], [0.8, 0.1, 0.1]

    def value_iteration(self):
        theta = 0.001  # Convergence threshold
        count = 0  # Iteration counter
        all_states = self.policy.keys()
        flag = True  # Control variable for convergence

        while flag:
            count += 1
            print(count)
            delta = 0
            v_tmp = copy.deepcopy(self.v)
            
            for s in all_states:
                x, y = s
                max_a_value = -1

                for action in self.actions.values():
                    a_value = 0
                    actionl, probs = self.get_action_probs(action)
                    for a, p in zip(actionl, probs):
                        s_, r = self.env.step(s, a)
                        x_, y_ = s_
                        a_value += p * (r + self.gamma * v_tmp[y_][x_])
                    
                    if a_value > max_a_value:
                        max_a_value = a_value
                        self.policy[s] = action
                        self.v[y][x] = max_a_value
                        delta1 = abs(v_tmp[y][x] - self.v[y][x])
                
                self.value_history[(x, y)].append(self.v[y][x])
                delta = max(delta, delta1)
                if delta < theta:
                    flag = False

if __name__ == '__main__':
    env = Grid()
    gamma = 0.99 # Discount Factor
    agent = Agent(env, gamma)
    agent.value_iteration()
    
    print("Values for each state:")
    print(agent.v)
    print()
    print("Agent policy:")
    direction_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    # Convert the policy dictionary
    formatted_policy = {key: direction_map[value] for key, value in agent.policy.items()}

    # Pretty-print the formatted policy
    pprint.pprint(formatted_policy)

    df_dict = {str(key): value for key, value in agent.value_history.items()}
    df = pd.DataFrame.from_dict(df_dict)
    df.loc[0] = 0
    
    # Get directory where script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the file name and save path
    csv_filename = "value_iteration.csv"
    csv_filepath = os.path.join(script_dir, csv_filename)
    
    df.to_csv(csv_filepath, index=False)