#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from array import array


# In[2]:



MIN = -np.inf


# In[3]:



class AlphaVector(object):
    """
    Simple wrapper for the alpha vector used for representing the value function for a POMDP as a piecewise-linear,
    convex function
    """
    def __init__(self, a, v):
        self.action = a
        self.v = v

    def copy(self):
        return AlphaVector(self.action, self.v)
    


# # Define POMDP

# In[4]:


discount = 0.9

human_prefers_pans, human_prefers_plates = 0,1
robot_takes_pan, robot_takes_plate = 0,1
human_takes_pan, human_takes_plate = 0,1

states = [human_prefers_pans, human_prefers_plates]
actions = [robot_takes_pan, robot_takes_plate]
observations = [human_takes_pan, human_takes_plate]

states_text = {human_prefers_pans: 'human_prefers_pans', human_prefers_plates: 'human_prefers_plates'}
actions_text = {robot_takes_pan: 'robot_takes_pan', robot_takes_plate: 'robot_takes_plate'}
observations_text = {human_takes_pan: 'human_takes_pan', human_takes_plate: 'human_takes_plate'}

num_states = 2
num_actions = 2
num_observations = 2

T = np.array([[[1.0,0.0] , [0.0,1.0] ],
    [[1.0,0.0] , [0.0,1.0] ]])

O = np.array([[[0.85, 0.15] , [0.15, 0.85] ],
    [[0.85, 0.15] , [0.15, 0.85] ]])


R = np.array([[-100.0, 10.0 ],
    [10.0, -100.0 ]])

C = [-1, -1]


# # setup params

# # Solve POMDP

# In[5]:


def solve_pomdp(max_timesteps, alpha_vecs, gamma_reward):

    for step in range(max_timesteps):

        # First compute a set of updated vectors for every action/observation pair
        gamma_intermediate = {}
        for a in actions:
            gamma_intermediate[a] = {}
            for o in observations:
    #             gamma_action_obs = compute_gamma_action_obs(a, o)

                gamma_action_obs = []
                for alpha in alpha_vecs:
                    v = np.zeros(num_states)  # initialize the update vector [0, ... 0]
                    for i, si in enumerate(states):
                        for j, sj in enumerate(states):
                            v[i] += T[a, si, sj] * O[a, sj, o] * alpha.v[j]
                        v[i] *= discount
                    gamma_action_obs.append(v)

                gamma_intermediate[a][o] = gamma_action_obs

        # Now compute the cross sum
        gamma_action_belief = {}
        for a in actions:

            gamma_action_belief[a] = {}
            for bidx, b in enumerate(belief_points):

                gamma_action_belief[a][bidx] = gamma_reward[a].copy()
    #             print("gamma_reward[a].copy()", gamma_reward[a].copy())

                for o in observations:
                    # only consider the best point
                    best_alpha_idx = np.argmax(np.dot(gamma_intermediate[a][o], b))
    #                 print("gamma_intermediate[a][o][best_alpha_idx]", gamma_intermediate[a][o][best_alpha_idx])
                    gamma_action_belief[a][bidx] += gamma_intermediate[a][o][best_alpha_idx]


        # Finally compute the new(best) alpha vector set
        alpha_vecs, max_val = [], MIN

        for bidx, b in enumerate(belief_points):
            best_av, best_aa = None, None

            for a in actions:
                val = np.dot(gamma_action_belief[a][bidx], b)
                if best_av is None or val > max_val:
                    max_val = val
                    best_av = gamma_action_belief[a][bidx].copy()
                    best_aa = a

            alpha_vecs.append(AlphaVector(a=best_aa, v=best_av))


    return alpha_vecs


# # Play Game

# In[6]:


# setup alpha vectors
alpha_vecs = [AlphaVector(a=-1, v=np.zeros(num_states))] 
stepsize = 0.1
belief_points = [[np.random.uniform() for s in states] for p in np.arange(0., 1. + stepsize, stepsize)]

# gamma_reward_og = {
#             a: np.frombuffer(array('d', [R[s,a] for s in states]))
#             for a in actions
#         }

gamma_reward = {
            a: np.array( [R[a,s] for s in states])
            for a in actions
        }

max_timesteps = 10


# In[7]:


# belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()
rand_nums = np.random.randint(0, 100, size=num_states)
base = sum(rand_nums)*1.0
belief = [x/base for x in rand_nums]
belief = [0.5, 0.5]
state = 1


# In[8]:




total_rewards = 0
budget = 20
max_play = 20
for i in range(max_play):
    # plan, take action and receive environment feedbacks
    alpha_vecs = solve_pomdp(max_timesteps, alpha_vecs, gamma_reward) 
    print("belief", belief)
#     action = pomdp.get_action(belief)
    
#     print("ALPHA VECS")
    for av in alpha_vecs:
        v = np.dot(av.v, belief)
#         print("action: ", av.action)
#         print("v: ", v)
    
    
    max_v = -np.inf
    best = None
    for av in alpha_vecs:
        v = np.dot(av.v, belief)
        if v > max_v:
            max_v = v
            best = av
    action = best.action
    
    print("state ", states_text[state])
    print("action ", actions_text[action])
    
    
    ai = action
    si = state
    # get new state
    s_probs = [T[ai, si, sj] for sj in states]
    next_state = states[np.random.choice(np.arange(num_states), p=s_probs)]

    # get new observation
    o_probs = [O[ai, next_state, oj] for oj in observations]
    next_obs = observations[np.random.choice(np.arange(num_observations), p=o_probs)]

    next_reward = R[ai, si]  
    next_cost = C[ai]
    
    print("reward ", next_reward)
    print("next_obs ", observations_text[next_obs])
    print("next_state ", states_text[next_state])
    print()
        
    new_state, obs, reward, cost = next_state, next_obs, next_reward, next_cost
#     state, observation, reward, cost = self.simulate_action(curr_state, action)
#     curr_state = state
        
#     new_state, obs, reward, cost = pomdp.take_action(action)

    # update states
#     belief = pomdp.update_belief(belief, action, obs)

    b_new = []
    for sj in states:
        p_o_prime = O[action, sj, obs]
        summation = 0.0
        for i, si in enumerate(states):
            p_s_prime = T[action, si, sj]
            summation += p_s_prime * float(belief[i])
        b_new.append(p_o_prime * summation)

    # normalize
    total = sum(b_new)
    belief = [x / total for x in b_new]
    
    
    
    total_rewards += reward
    budget -= cost

    if budget <= 0:
        print('Budget spent.')
        break
        
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




