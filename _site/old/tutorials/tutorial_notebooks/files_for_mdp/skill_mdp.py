import copy
import math
import pdb

import numpy as np
import pickle
import sys
import networkx as nx
from itertools import product
import itertools
import matplotlib.pyplot as plt



DIRTY_SHIRT = 'dirty_shirt'
PICKUP = 'pickup'
PLACE = 'place'


class Gridworld():
    def __init__(self, initial_config, reward_dict):
        self.reward_dict = reward_dict
        self.initial_config = initial_config

        self.target_object = reward_dict['target_object']
        self.target_goal = reward_dict['target_goal']

        self.dirty_shirt_positions = initial_config['dirty_shirt_positions']
        self.start_pos = initial_config['start_pos']
        self.g1_laundry_bin = initial_config['g1']
        self.g2_closet = initial_config['g2']
        self.objects_in_g1 = None
        self.objects_in_g2 = None

        self.set_env_limits()
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.possible_single_actions = self.make_actions_list()
        self.current_state = self.create_initial_state()
        self.num_features = 4
        self.correct_target_reward = 10

        # set value iteration components
        self.transitions, self.rewards, self.state_to_idx, self.idx_to_action, \
        self.idx_to_state, self.action_to_idx = None, None, None, None, None, None
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.001
        self.gamma = 0.99
        self.maxiter = 10000





    def make_actions_list(self):
        actions_list = []
        actions_list.extend(self.directions)
        actions_list.append(PICKUP)
        actions_list.append(PLACE)
        return actions_list


    def set_env_limits(self):
        self.x_min = 0
        self.x_max = 5
        self.y_min = 0
        self.y_max = 5

        self.all_coordinate_locations = list(product(range(self.x_min,self.x_max),
                                                     range(self.y_min, self.y_max)))



    def reset(self):
        self.current_state = self.create_initial_state()




    def create_initial_state(self):
        # create dictionary of object location to object type and picked up state
        state = {}
        state['pos'] = copy.deepcopy(self.start_pos)
        state['holding'] = None
        state['objects_in_g1'] = None
        state['objects_in_g2'] = None
        return state


    def is_done_given_state(self, current_state):
        # check if an object has been placed at either the laundry basket or the closet/dresser.
        if current_state['objects_in_g1'] != None or current_state['objects_in_g2'] != None:
            return True

        return False

    def is_valid_push(self, current_state, action):
        current_loc = current_state['pos']
        new_loc = tuple(np.array(current_loc) + np.array(action))
        if new_loc[0] < self.x_min or new_loc[0] >= self.x_max or new_loc[1] < self.y_min or new_loc[1] >= self.y_max:
            return False
        return True


    def step_given_state(self, input_state, action):
        step_cost = -0.1
        current_state = copy.deepcopy(input_state)

        if self.is_done_given_state(current_state):
            step_reward = 0
            return current_state, step_reward, True


        if action in self.directions:
            if self.is_valid_push(current_state, action) is False:
                step_reward = step_cost
                return current_state, step_reward, False

        if action == PICKUP:

            # if not holding anything
            if current_state['holding'] is None:
                # check if there is an object to pick up
                if current_state['pos'] in self.dirty_shirt_positions:
                    current_state['holding'] = 'dirty_shirt'


                step_reward = step_cost
                return current_state, step_reward, False

            else:
                step_reward = step_cost
                return current_state, step_reward, False

        if action == PLACE:
            if current_state['holding'] is not None:
                holding_object = current_state['holding']
                if current_state['pos'] == self.g1_laundry_bin:
                    current_state['objects_in_g1'] = current_state['holding']
                    current_state['holding'] = None
                    step_reward = step_cost
                    done = self.is_done_given_state(current_state)
                    if self.target_object == holding_object and self.target_goal == 'g1':
                        step_reward += self.correct_target_reward
                        return current_state, step_reward, done
                elif current_state['pos'] == self.g2_closet:
                    current_state['objects_in_g2'] = current_state['holding']
                    current_state['holding'] = None
                    step_reward = step_cost
                    done = self.is_done_given_state(current_state)
                    if self.target_object == holding_object and self.target_goal == 'g2':
                        step_reward += self.correct_target_reward
                        return current_state, step_reward, done

                step_reward = step_cost
                return current_state, step_reward, False
            else:
                step_reward = step_cost
                return current_state, step_reward, False


        current_loc = current_state['pos']
        new_loc = tuple(np.array(current_loc) + np.array(action))
        current_state['pos'] = new_loc
        step_reward = step_cost
        done = self.is_done_given_state(current_state)


        return current_state, step_reward, done



    def state_to_tuple(self, current_state):
        # convert current_state to tuple
        current_state_tup = []
        current_state_tup.append(('pos', current_state['pos']))
        current_state_tup.append(('holding', current_state['holding']))
        current_state_tup.append(('objects_in_g1', current_state['objects_in_g1']))
        current_state_tup.append(('objects_in_g2', current_state['objects_in_g2']))
        return tuple(current_state_tup)

    def tuple_to_state(self, current_state_tup):
        # convert current_state to tuple
        current_state_tup = list(current_state_tup)
        current_state = {}
        current_state['pos'] = current_state_tup[0][1]
        current_state['holding'] = current_state_tup[1][1]
        current_state['objects_in_g1'] = current_state_tup[2][1]
        current_state['objects_in_g2'] = current_state_tup[3][1]

        return current_state

    def enumerate_states(self):
        self.reset()

        actions = self.possible_single_actions
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.current_state)]

        while stack:
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)

            # if state has not been visited, add it to the set of visited states
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get the neighbors of this state by looping through possible actions
            for idx, action in enumerate(actions):
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True

                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                new_state_tup = self.state_to_tuple(next_state)

                if new_state_tup not in visited_states:
                    stack.append(copy.deepcopy(next_state))

                # add edge to graph from current state to new state with weight equal to reward
                G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)

        states = list(G.nodes)
        # print("NUMBER OF STATES", len(state
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        for i in range(len(states)):
            # get all outgoing edges from current state
            state = self.tuple_to_state(idx_to_state[i])
            for action_idx_i in range(len(actions)):
                action = idx_to_action[action_idx_i]
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True

                else:
                    next_state, team_reward, done = self.step_given_state(state, action)

                next_state_i = state_to_idx[self.state_to_tuple(next_state)]

                transition_mat[i, next_state_i, action_idx_i] = 1.0
                reward_mat[i, action_idx_i] = team_reward

        self.transitions, self.rewards, self.state_to_idx, \
        self.idx_to_action, self.idx_to_state, self.action_to_idx = transition_mat, reward_mat, state_to_idx, \
                                                                    idx_to_action, idx_to_state, action_to_idx


        return transition_mat, reward_mat, state_to_idx, idx_to_action, idx_to_state, action_to_idx

    def vectorized_vi(self):
        # get number of states and actions
        n_states = self.transitions.shape[0]
        n_actions = self.transitions.shape[2]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))
        policy = {}

        for i in range(self.maxiter):
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()
                # compute new value function
                Q[s] = np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0)
                vf[s] = np.max(np.sum((self.rewards[s] + self.gamma * vf) * self.transitions[s, :, :], 0))
                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))
            # check for convergence
            if delta < self.epsilson:
                break
        # compute optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(np.sum(vf * self.transitions[s, :, :], 0))
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        return vf, pi

    def rollout_full_game_joint_optimal(self):
        self.reset()
        done = False
        total_reward = 0

        iters = 0
        game_results = []
        sum_feature_vector = np.zeros(4)

        while not done:
            iters += 1

            current_state_tup = self.state_to_tuple(self.current_state)

            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]
            print("ACTION", action)
            print("state", self.current_state)

            game_results.append((self.current_state, action))

            next_state, team_rew, done = self.step_given_state(self.current_state, action)

            self.current_state = next_state

            total_reward += team_rew

            if iters > 40:
                break


        return total_reward, game_results


    def compute_optimal_performance(self):
        self.enumerate_states()

        self.vectorized_vi()

        optimal_rew, game_results = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results



if __name__ == '__main__':
    # reward_weights = [1, -1, -1, 1]  # [obj A placed in G1, object A in G2, B in G1, B in G2]

    state = {
        'start_pos': (0,0),
        'g1': (4,4),
        'g2': (4,0),
        'dirty_shirt_positions': [(1,3)],
    }
    reward = {
        'target_object': DIRTY_SHIRT,
        'target_goal': 'g1',
    }

    game = Gridworld(state, reward)
    optimal_rew, game_results = game.compute_optimal_performance()
    # pdb.set_trace()


