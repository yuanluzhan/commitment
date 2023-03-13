import math
import random

import delayed
import pandas as pd
import mesa.time
import mesa.datacollection
import matplotlib.pyplot as plt
import numpy as np
import random
from multiprocessing import Pool, cpu_count
import math
import os
import time
from joblib import Parallel,delayed
import json


class Agent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, param_agent):
        super().__init__(unique_id, model)

        self.type = random.randint(0, 2)  # type  type 0 cooperate  1 fake   0 no preference
        # strategy
        self.strategy_region()
        self.strategy = random.choice(self.strategies_region)  # strategy

        self.individual = 0 # abs(random.gauss(mu=0, sigma=param_agent["ind_distri"][0]))  # preference intense
        self.rationality = 5#float(param_agent["ind_rationality"][0])#random.gauss(mu=param_agent["ind_rationality"][0],
                                        #sigma=param_agent["ind_rationality"][1])  # updating parameter
        self.memory_length = random.randint(param_agent["ind_memory_length"][0],
                                            param_agent["ind_memory_length"][1])  # learning horizon
        self.opp_history_strategy = []  # interaction history strategy
        self.opp_history = []  # interaction history
        self.history = []  # history strategy
        self.history.append(self.strategy)
        self.interaction_time = []

    def strategy_region(self):
        if self.type == 0:
            self.strategies_region = [0, 1, 2,3, 4]
        if self.type == 2:
            self.strategies_region = [0, 1, 2,3, 4]
        if self.type == 1:
            self.strategies_region = [0,1,2, 3, 4]

    def interation(self, t, payoff_matrix):
        # Random Match a player
        other_agent = self.random.choice(self.model.schedule.agents)

        # Store basic information
        self.opp_history.append(other_agent.unique_id)
        other_agent.opp_history.append(self.unique_id)
        self.interaction_time.append(t)
        other_agent.interaction_time.append(t)
        # Strategy Choice
        tmp = random.uniform(0, 1)
        if tmp < 0.0001:
            self.strategy = random.choice(other_agent.strategies_region)
            other_agent.strategy = random.choice(other_agent.strategies_region)
        else:
            self.strategy, other_agent.strategy = best_response(self, other_agent, payoff_matrix)
        other_agent.opp_history_strategy.append(self.strategy)
        self.opp_history_strategy.append(other_agent.strategy)
        self.history.append(self.strategy)
        other_agent.history.append(other_agent.strategy)

    def calculate_payoff(self, payoff_matrix):

        self.payoff = []
        t = self.sample_memory()
        for s in range(0, 5):
            tmp = 0
            for x in t:
                tmp = tmp + payoff_matrix[s][x] + self.individual * int(self.type == (s // 2))
                ylz = self.type == (s // 2)
            self.payoff.append(tmp)

    def step(self, t, payoff_matrix):
        self.interation(t, payoff_matrix)


def sample_other_agent_memory(agent1, agent2):
    if len(agent1.history) < agent1.memory_length:
        t1 = agent1.history
    else:
        t1 = agent1.history[-agent1.memory_length:-1]
    if len(agent2.history) < agent2.memory_length:
        t2 = agent2.history
    else:
        t2 = agent2.history[-agent2.memory_length:-1]
    return t1, t2


def best_response(agent1, agent2, payoff_matrix):
    p1, p2 = [], []
    t1, t2 = agent1.history[-agent1.memory_length:], agent2.history[-agent2.memory_length:]

    for fictional_stra1 in agent1.strategies_region:
        fic_payoff1 = int(agent1.type == fictional_stra1 // 2) * len(t2)* agent1.individual + sum(
            payoff_matrix[fictional_stra1][his_stra] for his_stra in t2)
        p1.append(math.exp(agent1.rationality * fic_payoff1))
    strategy1 = random.choices(agent1.strategies_region, weights=p1, k=1)[0]

    for fictional_stra2 in agent2.strategies_region:
        fic_payoff2 = int(agent2.type == fictional_stra2 // 2) * len(t1) * agent2.individual + sum(
            payoff_matrix[fictional_stra2][his_stra] for his_stra in t1)
        p2.append(math.exp(agent2.rationality * fic_payoff2))
    strategy2 = random.choices(agent2.strategies_region, weights=p2, k=1)[0]
    # for fictional_stra1 in agent1.strategies_region:
    #     tmp2 = 0
    #     for his_stra in t2:
    #         tmp2 = tmp2 + payoff_matrix[fictional_stra1][his_stra] + agent1.individual * int(
    #             agent1.type == fictional_stra1 // 2)
    #     p1.append(math.exp(agent1.rationality * tmp2))
    # strategy1 = random.choices(agent1.strategies_region, weights=p1, k=1)[0]
    # for fictional_stra2 in agent2.strategies_region:
    #     tmp1 = 0
    #     for his_stra in t1:
    #         tmp1 = tmp1 + payoff_matrix[fictional_stra2][his_stra] + agent2.individual * int(
    #             agent2.type == fictional_stra2 // 2)
    #     p2.append(math.exp(agent2.rationality * tmp1))
    # strategy2 = random.choices(agent2.strategies_region, weights=p2, k=1)[0]
    return strategy1, strategy2


class RandomActivation(mesa.time.RandomActivation):
    def step(self, agent_keys, payoff_matrix):
        for agent_key in agent_keys:
            self._agents[agent_key].step(self.steps, payoff_matrix)
        self.steps += 1
        self.time += 1


class Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, para_model, para_agent):
        c = para_model['c']
        a = para_model['a']
        b = para_model['b']
        d = para_model['d']
        epsilon2 = para_model['epsilon2']
        delta2 = para_model['delta2']

        self.num_agents = N
        self.schedule = RandomActivation(self)
        # Create agents
        self.agent_keys = []
        self.payoff_matrix = np.array(
            [
                [a - epsilon2 / 2, a - epsilon2, 0, b - epsilon2 + delta2, a - epsilon2],
                [a, a, b, b, b],
                [0, c, d, d, d],
                [c - delta2, c, d, d, d],
                [a, c, d, d, d]
            ]
        )
        self.agent_information = []

        for i in range(self.num_agents):
            a = Agent(i, self, para_agent)
            self.schedule.add(a)
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Agent_keys": "agent_keys",
                "Agent_information": "agent_information"
            },
            agent_reporters={
                "individual": "individual",
                "agent_type": "type",
                "strategy_history": "opp_history_strategy",
                "opp_history": "opp_history",
                "opp_history_strategy": "opp_history_strategy",
                "interaction_time": "interaction_time"
            }
        )

    def step(self):
        agent_information = []
        for agent in self.schedule.agents:
            agent_information.append(agent.strategy)
        # agent_inforamtion = [int(agent.strategy) for agent in self.schedule.agents]
        self.agent_information.append(agent_information)
        agent_keys = random.choices(np.arange(0, self.num_agents), k=10)
        self.agent_keys.append(agent_keys)
        self.schedule.step(agent_keys, self.payoff_matrix)

def run(params):
    para_model, para_agent = params
    model = Model(100, para_model, para_agent)
    for i in range(1000):
        model.step()
    model.agent_keys = np.array(model.agent_keys)
    model.datacollector.collect(model)
    record = model.datacollector.get_model_vars_dataframe()
    Agent_keys = pd.DataFrame(record['Agent_keys'].values[0])
    Agent_information = pd.DataFrame(record['Agent_information'].values[0])
    a = model.datacollector.get_agent_vars_dataframe()
    path = 'result/' + str(list(para_model.values())) +'/' + str(list(para_agent.values()))
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    a.to_csv(path + '/agent_result.csv')
    Agent_keys.to_csv(path + '/agent_keys.csv')
    Agent_information.to_csv(path + '/agent_information.csv')
    np.save(path + '/para.npy', [para_model,para_agent])

    # tmp = Agent_information.values
    # s1 = []
    # for i in range(len(tmp)):
    #     # print(tmp[i][1:])
    #     s1.append(np.bincount(tmp[i][1:]))
    #
    # plt.plot(s1, label=['Com', 'C', 'D', 'fake', 'free'])
    # plt.legend()
    # plt.show()
    # record = pd.read_csv("result1/[0.9, 0.8, 0.8, 0.8, 0.01, 0.0]/[[0.01], [10, 0.1], [10, 10]]/model_result.csv")
    # tmp = record['Agent_information'].values[0]
    # s1 = []
    # for i in range(len(tmp)):
    #     s1.append([tmp[i].count(0), tmp[i].count(1), tmp[i].count(2), tmp[i].count(3), tmp[i].count(4)])
    #
    # plt.plot(s1)
    # plt.show()

def para_generat():
    params = []
    for d in np.arange(0.1, 0.8, 0.1):
        for b in np.arange(d+0.1, 0.9, 0.1):
            for a in np.arange(b+0.1, 1, 0.1):
                for c in np.arange(a+0.1, 1.01, 0.1):
                    for epsilon2 in np.arange(0.01, 0.10, 0.01):
                        for delta2 in np.arange(0.05, 2*(c - a), 0.05):
                            for ind_distri in np.arange(0.01, 0.02, 0.01):
                                for ind_rationality1 in np.arange(1, 3, 1):
                                    for ind_memory_length1 in np.arange(9, 11, 1):
                                        for ind_memory_length2 in np.arange(ind_memory_length1+1, 11, 1):
                                            para_model = {
                                                'c': np.around(c,2),
                                                'a': np.around(a,2),
                                                'b': np.around(b,2),
                                                'd': np.around(d,2),
                                                'epsilon2': np.around(epsilon2,2),
                                                'delta2': np.around(delta2,2)
                                            }
                                            para_agent = {
                                                "ind_distri": [np.around(ind_distri,2)],
                                                "ind_rationality": [np.around(ind_rationality1,1)],# np.around(ind_rationality2,2)],
                                                "ind_memory_length": [np.around(ind_memory_length1,2), np.around(ind_memory_length2,2)]
                                            }
                                            yield [para_model, para_agent]
                                            # params.append([para_model,para_agent])
                                            # return params

if __name__ == '__main__':

    params = para_generat()
    # with Pool(10) as p:
    #     p.map(run, params)
    # for param in params:
    #     print(param)
    #     run(param)
    #     raise Exception('s')
    print(time.time())
    result = Parallel(n_jobs = 200)(delayed(run)(param) for param in params)
    print(time.time())
