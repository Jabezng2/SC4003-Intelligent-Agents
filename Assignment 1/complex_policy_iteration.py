import numpy as np
import copy
import pandas as pd
import pprint
import os
from config import *

class Grid:
    def __init__(self, tile_reward= TILE_REWARD, map = None, size = None):
        self.tile_reward = tile_reward
        
        # Initialize the map if not provided, otherwise use the given map
        if map == None:
            self.map = [["Green", "Wall", "Green", "White", "White", "Green"],
                    ["White", "Brown", "White", "Green","Wall","Brown"],
                    ["White","White", "Brown", "White", "Green","White"],
                    ["White", "White", "White", "Brown", "White", "Green"],
                    ["White", "Wall","Wall","Wall","Brown","White"],
                    ["White","White","White","White","White","White"]]
        else:
            self.map = map
        
        # Initialize a value map with zeros, defaulting to 6x6 grid if no size is provided
        if size == None:
            self.values_map = np.zeros((6,6))
        else:
            self.values_map = np.zeros((size, size))

        # Store all valid states (non-wall cells) in the environment
        self.states = [(c, r) for r, row in enumerate(self.map) for c, tile in enumerate(row) if tile != "Wall"]
    
    def step(self, pos, action):
        x, y = copy.deepcopy(pos)  # Deep copy to avoid modifying the original position
        x_, y_ = x, y

        # Check if the action is valid, otherwise return the current position and reward
        if self.is_valid_action(x, y, action) is False:
            return pos, self.get_reward(pos)

        # Move according to the action taken
        if action == ACTIONS["UP"]:
            y_ = y-1
        elif action == ACTIONS["DOWN"]:
            y_ = y+1
        elif action == ACTIONS["LEFT"]:
            x_ = x-1
        elif action == ACTIONS["RIGHT"]:
            x_ = x+1

        # Get the reward for the new position
        reward = self.get_reward((x, y))
        next_pos = (x_, y_)

        return next_pos, reward
    
    def get_reward(self, pos: tuple):
        x, y = pos
        return self.tile_reward[self.map[y][x]]  # Retrieve reward from the tile reward dictionary
        
    def is_valid_action(self, x, y, action):
        x_, y_ = x, y

        # Determine the next position based on the action taken
        if action == ACTIONS["UP"]:
            y_ += -1
        elif action == ACTIONS["DOWN"]:
            y_ += 1
        elif action == ACTIONS["LEFT"]:
            x_ += -1
        elif action == ACTIONS["RIGHT"]:
            x_ += 1

        # Check if the move is within the grid bounds
        if len(self.map) <= y_  or y_ < 0:
            return False
        elif len(self.map[0]) <= x_ or x_ < 0:
            return False
        elif self.map[y_][x_] == "Wall":
            return False

        return True  # Return True if the move is valid

class Agent():
    def __init__(self, env, gamma, policy = {}, actions= ACTIONS):
        self.actions = actions
        self.gamma = gamma
        
        # Initialize policy randomly if not provided
        if not policy:
            for s in env.states:
                policy[s] = 0
        self.policy = policy
        
        # Store value history for each state
        self.value_history ={}
        for s in env.states:
            self.value_history[s]=[]

        self.env = env
        self.v = self.env.values_map  # Reference to the value map
   
    def get_action_probs(self, action):
        # Define probability distribution for each action, considering unintended deviations
        if action == 0:
            return [0,2,3], [0.8,0.1,0.1]
        if action == 1:
            return [1,2,3], [0.8,0.1,0.1]
        if action == 2:
            return [2,0,1], [0.8,0.1,0.1]
        if action == 3:
            return [3,0,1], [0.8,0.1,0.1]

    def evaluate_policy(self):
        theta = 0.001  # Convergence threshold
        count = 0  # Iteration counter
        all_states = self.policy.keys()
        delta = np.inf  # Initialize delta to a high value

        while delta >= theta:
            count+=1
            delta = 0
            v_tmp = copy.deepcopy(self.v)
            
            for s in all_states:
                x, y = s
                actions, probs = self.get_action_probs(self.policy[s])
                value = 0
                
                for a, p in zip(actions,probs):
                    s_, r = self.env.step(s, a)
                    x_, y_ = s_
                    value += p*(r + self.gamma * v_tmp[y_][x_])
                
                self.v[y][x] = value
                self.value_history[(x,y)].append(value)
                delta = max(delta, abs(v_tmp[y][x] - self.v[y][x]))

    def improve_policy(self):
        is_stable = True  # Assume policy is stable
        all_states = self.policy.keys()
        
        for s in all_states:
            old_pi = copy.deepcopy(self.policy[s])
            argmax_action = None
            max_a_value = -1

            for action in self.actions.values():
                a_value = 0
                actionl, probs = self.get_action_probs(action)
                for a, p in zip(actionl,probs):
                    s_, r = self.env.step(s, a)
                    x_, y_ = s_
                    a_value += p*(self.gamma * self.v[y_][x_])
                
                if a_value > max_a_value:
                    max_a_value = a_value
                    argmax_action = action
                    self.policy[s] = argmax_action
            
            if old_pi != argmax_action:
                is_stable = False

        return is_stable  # Return whether the policy has stabilized


if __name__ == '__main__':
    final_utilities = []
    def generate_map(n):
        # Define the colors and wall as strings
        elements = ["Brown", "White", "Wall", "Green"]

        # Generate a NxN 2D array with random selection of the elements
        map_nxn = np.random.choice(elements, size=(n, n), p=[0.45, 0.20, 0.25, 0.10])

        map_list = map_nxn.tolist()
        return map_list

    n = 10
    bonus = generate_map(n)
    env = Grid(map=bonus, size=n)
    gamma = 0.99 # Discount Factor
    is_stable = False
    count = 0

    agent = Agent(env, gamma)
    print("Iterations:")
    while is_stable == False:
          agent.evaluate_policy()
          tmp = copy.deepcopy(agent.v)
          final_utilities.append(tmp.flatten())
          is_stable = agent.improve_policy()
          count += 1
          print(count)

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

    df_final_utilities = pd.DataFrame(final_utilities)
    column_headers = [f"({x}, {y})" for y in range(n) for x in range(n)]

    df_final_utilities.columns = column_headers

    # Ensure DataFrame is not empty before modifying
    if df_final_utilities.empty:
        print("No utility data found! Skipping CSV writing.")
    else:
        df_final_utilities.loc[0] = 0

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define file names and paths correctly
    csv_filename = "complex_policy_iteration_each_step.csv"
    csv_filepath = os.path.join(script_dir, csv_filename)

    csv_filename2 = "complex_policy_iteration_final_utilities.csv"
    csv_filepath2 = os.path.join(script_dir, csv_filename2)

    df.to_csv(csv_filepath, index=False)
    df_final_utilities.to_csv(csv_filepath2, index=False)