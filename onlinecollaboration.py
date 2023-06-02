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

    def __init__(self, unique_id, model, para_individual_information, para_init_distribution):
        super().__init__(unique_id, model)

        self.init_distribution(para_init_distribution)
        self.init_individual(para_individual_information)

        self.history_opp_strategy = []  # interaction history strategy
        self.history_opp_id = []  # interaction history
        self.history = []  # history strategy
        self.history.append(self.strategy)
        self.interaction_time = []

    def init_individual(self,para_individual_information):
        self.individual = para_individual_information["para_individual"]
        self.rationality = para_individual_information["rationality"]
        self.memory_length = para_individual_information["history_length"]

    def init_distribution(self, para_init_distribution):
        # type 0 cooperate  1 fake   2 no preference
        self.type = np.random.choice([0,1,2],p=para_init_distribution["ind_type"])
        self.strategy_region()
        if para_init_distribution["strategy"] == "random":
            self.strategy = random.choice(self.strategies_region)
        if para_init_distribution["strategy"] == "Free":
            tmp = random.uniform(0, 1)
            if tmp < 0.1:
                self.strategy = random.choice(self.strategies_region)
            else:
                self.strategy = 4
        if para_init_distribution["strategy"] == "D":
            tmp = random.uniform(0, 1)
            if tmp < 0.1:
                self.strategy = random.choice(self.strategies_region)
            else:
                self.strategy = 2

    def strategy_region(self):
        if self.type == 0:
            self.strategies_region = [0, 1, 4]
        if self.type == 1:
            self.strategies_region = [2, 3, 4]
        if self.type == 2:
            self.strategies_region = [0, 1, 2, 4]

        self.preference_interest = np.zeros([3,5])
        self.preference_interest[0][0:2] = 1
        self.preference_interest[1][2:5] = 1

    def interation(self, t, payoff_matrix):
        # Random Match a player
        other_agent = self.random.choice(self.model.schedule.agents)

        # Store basic information
        self.history_opp_id.append(other_agent.unique_id)
        other_agent.history_opp_id.append(self.unique_id)
        self.interaction_time.append(t)
        other_agent.interaction_time.append(t)
        # Strategy Choice
        tmp = random.uniform(0, 1)
        if tmp < 0.01:
            self.strategy = random.choice(other_agent.strategies_region)
            other_agent.rationality = random.choice(other_agent.strategies_region)
        else:
            self.strategy, other_agent.rationality = best_response(self, other_agent, payoff_matrix)




        self.history.append(self.strategy)
        self.history_opp_strategy.append(other_agent.rationality)
        other_agent.history.append(other_agent.rationality)
        other_agent.history_opp_strategy.append(self.strategy)

    def calculate_payoff(self, payoff_matrix):

        self.payoff = []
        t = self.sample_memory()
        for s in range(0, 5):
            tmp = 0
            for x in t:
                tmp = tmp + payoff_matrix[s][x] + self.individual * int(self.preference_interest[self.type][s])
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

    def __init__(self, para):
        para_payoff, para_game_settings, para_individual_information, para_init_distribution = para
        c = para_payoff['c']
        a = para_payoff['a']
        b = para_payoff['b']
        d = para_payoff['d']
        epsilon = para_payoff['epsilon']
        delta = para_payoff['delta']

        self.num_agents = para_game_settings["num_all_players"]
        self.num_participants = para_game_settings["num_participants"]
        self.schedule = RandomActivation(self)
        # Create agents
        self.agent_keys = []
        self.agent1_strategy = []
        self.agent2_strategy = []
        self.payoff_matrix = np.array(
            [
                [a - epsilon / 2, a - epsilon, 0, b - epsilon + delta, a - epsilon],
                [a, a, b, b, b],
                [0, c, d, d, d],
                [c - delta, c, d, d, d],
                [a, c, d, d, d]
            ]
        )
        self.agent_information = []

        for i in range(self.num_agents):
            a = Agent(i, self, para_individual_information, para_init_distribution)
            self.schedule.add(a)
        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Agent_keys": "agent_keys",
                "Agent_information": "agent_information",
                "Agent1_strategy": "agent1_strategy",
                "Agent2_strategy": "agent2_strategy"
            },
            agent_reporters={
                "individual": "individual",
                "agent_type": "type",
            }
        )

    def step(self):

        agent_keys = random.choices(np.arange(0, self.num_agents), k=self.num_participants)
        self.agent_keys.append(agent_keys)
        self.schedule.step(agent_keys, self.payoff_matrix)
        interaction1 = []
        interaction2 = []
        for i in agent_keys:
            agent = self.schedule.agents[i]
            interaction1.append(agent.strategy)
            interaction2.append(agent.history_opp_strategy[-1])
        self.agent1_strategy.append(interaction1)
        self.agent2_strategy.append(interaction2)
        agent_information = []
        for agent in self.schedule.agents:
            agent_information.append(agent.strategy)

        self.agent_information.append(agent_information)

