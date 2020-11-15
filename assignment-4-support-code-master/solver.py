import random
from laser_tank import LaserTankMap, DotDict
import time
"""
Template file for you to implement your solution to Assignment 4. You should implement your solution by filling in the
following method stubs:
    train_q_learning()
    train_sarsa()
    get_policy()
    
You may add to the __init__ method if required, and can add additional helper methods and classes if you wish.

To ensure your code is handled correctly by the autograder, you should avoid using any try-except blocks in your
implementation of the above methods (as this can interfere with our time-out handling).

COMP3702 2020 Assignment 4 Support Code
"""

def get_max(q_values, state):
    max = None
    if len(q_values.keys()) == 0:
        return 0,'f'
    if state not in q_values.keys():
        return 0,'f'
    for i in q_values[state].keys():
        if max == None or q_values[state][i] > max:
            max = q_values[state][i]
        else:
            continue
    return max,i
def update_q(self,q_values,key,a,max_next_value):
    if key in q_values.keys():
        if a in q_values[key].keys():
            q_value = q_values[key][a]
            value = q_value + self.alpha*(self.a_reward[0]+self.gamma*max_next_value-q_value)
            q_values[key][a] = value
        else:
            q_value = 0
            value = q_value + self.alpha*(self.a_reward[0]+self.gamma*max_next_value-q_value)
            q_values[key][a] = value
    else:
        q_value = 0
        value = q_value + self.alpha*(self.a_reward[0]+self.gamma*max_next_value-q_value)
        q_values[key] = {a: value}
    return q_values
def next_action_porbility(self,a):
    next_action = None
    p = random.random()
    if  p <= 0.8:
        next_action = a
    elif random.random()<=0.2:
        next_action = self.actions[0]
    elif random.random()<=0.2:
        next_action = self.actions[1]
    elif random.random()<=0.2:
        next_action = self.actions[2]
    # elif random.random(0,1)<=0.2:
    #     next_action = self.actions[3]
    return next_action
class Solver:
    def __init__(self):
        """
        Initialise solver without a Q-value table.
        """
        self.alpha = 0.2
        self.actions = ['f', 'l', 'r']
        # 当前action的reward
        self.a_reward = []
        self.state = None
        self.next_reward_action = {}
        self.gamma = None
        self.timelimit = None
        self.q_values = None
        self.value = {'temp': None}

    def train_q_learning(self, simulator):
        """
        Train the agent using Q-learning, building up a table of Q-values.
        :param simulator: A simulator for collecting episode data (LaserTankMap instance)
        """
        #初始化
        self.gamma = simulator.gamma
        self.timelimit = simulator.time_limit
        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        q_values = {}
        #initialize start position
        self.init_simulator = simulator
        start_time = time.time()
        exec_time = 0
        n=0
        temp_x = simulator.player_x
        temp_y = simulator.player_y
        temp_heading = simulator.player_heading
        temp_grid_data = [row[:] for row in simulator.grid_data]
        while time.time() - start_time <= self.timelimit:
            self.a_reward = []
            key = None
            key = hash(simulator)

            a = random.sample(self.actions, 1)[0]
            self.a_reward= simulator.apply_move(a)

            #get max_reward to choose maxˆQ∗(s, a)
            new_key = hash(simulator)
            max_next_value,action = get_max(q_values, new_key)
            if self.a_reward[1]:
                q_values = update_q(self,q_values,key,a,max_next_value)
                simulator.player_x = temp_x
                simulator.player_y = temp_y
                simulator.player_heading = temp_heading
                simulator.grid_data = [row[:] for row in temp_grid_data]
                continue
            q_values = update_q(self,q_values,key,a,max_next_value)
        self.q_values = q_values
        # print (n)

    def train_sarsa(self, simulator):
        """
        Train the agent using SARSA, building up a table of Q-values.
        :param simulator: A simulator for collecting episode data (LaserTankMap instance)
        """

        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        #初始化
        self.gamma = simulator.gamma
        self.timelimit = simulator.time_limit
        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        q_values = {}
        #initialize start position
        start_time = time.time()

        temp_x = simulator.player_x
        temp_y = simulator.player_y
        temp_heading = simulator.player_heading
        temp_grid_data = [row[:] for row in simulator.grid_data]
        a = random.sample(self.actions, 1)[0]
        while time.time() - start_time <= self.timelimit:
            self.a_reward = []
            key = None
            key = hash(simulator)
            # a = random.sample(self.actions, 1)[0]
            self.a_reward = simulator.apply_move(a)

            #get max_reward to choose maxˆQ∗(s, a)
            new_key = hash(simulator)
            max_next_value,next_a= get_max(q_values, new_key)
            if self.a_reward[1]:
                q_values = update_q(self,q_values,key,a,max_next_value)
                #get next action
                a=next_action_porbility(self,next_a)
                #initialize
                simulator.player_x = temp_x
                simulator.player_y = temp_y
                simulator.player_heading = temp_heading
                simulator.grid_data = [row[:] for row in temp_grid_data]
                continue
            q_values = update_q(self,q_values,key,a,max_next_value)
            #get next action
            a = next_action_porbility(self,next_a)

        self.q_values = q_values

        #
        # TODO
        # Write your SARSA implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should
        # continue training until the time limit is reached.
        #

        # store the computed Q-values
        self.q_values = q_values

    def get_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """

        #
        # TODO
        # Write code to return the optimal action to be performed at this state based on the stored Q-values.
        #
        # You can assume that either train_q_learning( ) or train_sarsa( ) has been called before this
        # method is called.
        #
        # When this method is called, you are allowed up to 1 second of compute time.
        #
        key = hash(state)
        max = None
        a = None
        for i in self.q_values[key].keys():
            if max == None or self.q_values[key][i] > max:
                max = self.q_values[key][i]
                a = i
        # print ('action:', i)
        return a

        pass







