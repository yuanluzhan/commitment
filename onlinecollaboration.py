import math
import random

import mesa
import matplotlib.pyplot as plt
import numpy as np
import random




class Agent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.strategy = 2
        self.individual = random.gauss(0,1)
        self.payoff = 0
        self.beta = 0.2


    def learn(self,other_agent):
        fitness = self.payoff-other_agent.payoff

        p12 = 1/(1+math.exp(self.beta*(fitness)))
        p21 = 1/(1+math.exp(other_agent.beta*(-fitness)))


        t1 = random.choices([0,1],weights=[1-p12,p12])[0]
        if t1 == 1:
            self.strategy = other_agent.strategy

        t2 = random.choices([0, 1], weights=[1 - p21, p21])[0]
        if t2 == 1:
            other_agent.strategy = self.strategy




    def interation(self,payoff_matrix):
        other_agent = self.random.choice(self.model.schedule.agents)
        tmp = random.uniform(0,1)
        if tmp<0.01:
            self.strategy = random.randint(0,4)
            #other_agent.strategy = 0 #random.randint(0,4)
        self.payoff = payoff_matrix[self.strategy][other_agent.strategy]
        other_agent.payoff = payoff_matrix[other_agent.strategy][self.strategy]
        self.learn(other_agent)



    def step(self,payoff_matrix):
        self.interation(payoff_matrix)

class RandomActivation(mesa.time.RandomActivation):
    def step(self,payoff_matrix):
        for agent in self.agent_buffer(shuffled=True):
            agent.step(payoff_matrix)
        self.steps += 1
        self.time += 1


class Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        self.num_agents = N
        self.epsilon2 = 0.01
        self.delta2 = 5
        a = 3
        b = 2
        c = 4
        d = 1
        self.schedule = RandomActivation(self)
        self.payoff_matrix = np.array(
            [
                [a - self.epsilon2 / 2, a - self.epsilon2, 0, b - self.epsilon2 + self.delta2,
                 a - self.epsilon2],
                [a, a, b, b, b],
                [0, c, d, d, d],
                [c - self.delta2, c, d, d, d],
                [c, c, d, d, d]
            ]
        )
        # Create agents
        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Com":"strategy_record0",
                "C":"strategy_record1",
                "D":"strategy_record2",
                "Fake":"strategy_record3",
                "Free":"strategy_record4",
            },
            agent_reporters={"payoff":"payoff"})


    def step(self):
        agent_strategy = [agent.strategy for agent in model.schedule.agents]
        self.agent_payoff = np.mean([agent.payoff for agent in model.schedule.agents])
        self.strategy_record0 = agent_strategy.count(0)
        self.strategy_record1 = agent_strategy.count(1)
        self.strategy_record2 = agent_strategy.count(2)
        self.strategy_record3 = agent_strategy.count(3)
        self.strategy_record4 = agent_strategy.count(4)

        self.datacollector.collect(self)
        self.schedule.step(self.payoff_matrix)


model = Model(1000)
for i in range(200):
    model.step()
record = model.datacollector.get_model_vars_dataframe()
a = model.datacollector.get_agent_vars_dataframe()
# a.plot()
record.plot()
plt.show()