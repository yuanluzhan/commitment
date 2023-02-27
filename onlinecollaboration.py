import math
import random
import mesa.time
import mesa.datacollection
import matplotlib.pyplot as plt
import numpy as np
import random
from multiprocessing import Pool, cpu_count

a = 3
b = 2
c = 4
d = 1
epsilon2 = 0.1
delta2 = 5
class Agent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.investment = random.randint(0,1)
        self.strategy = random.randint(0,4)
        self.individual = random.gauss(0.02,1)
        self.payoff = 0
        self.beta = 0.2
        self.memory = []

    def calculate_payoff(self, other_agent):
        payoff_matrix = np.array(
            [
                [a - epsilon2 / 2, a - epsilon2, 0, b - epsilon2 + delta2, a - epsilon2],
                [a, a, b, b, b],
                [0, c, d, d, d],
                [c - delta2, c, d, d, d],
                [a, c, d, d, d]
            ]
        )
        p1 = payoff_matrix[self.strategy][other_agent.strategy]
        p2 = payoff_matrix[other_agent.strategy][self.strategy]
        p1, p2 = self.calculate_payoff(other_agent)
        self.payoff = self.payoff + p1
        other_agent.payoff = other_agent.payoff + p2
        return 1, 2

    def learn(self):
        other_agent = self.random.choice(self.model.schedule.agents)
        fitness = self.payoff-other_agent.payoff
        p12 = 1/(1+math.exp(self.beta*(fitness)))
        p21 = 1/(1+math.exp(other_agent.beta*(-fitness)))
        t1 = random.choices([0,1],weights=[1-p12,p12])[0]
        if t1 == 1:
            self.strategy = other_agent.strategy
        t2 = random.choices([0, 1], weights=[1 - p21, p21])[0]
        if t2 == 1:
            other_agent.strategy = self.strategy


    def interation(self):
        other_agent = self.random.choice(self.model.schedule.agents)
        tmp = random.uniform(0,1)
        if tmp<0.001:
            self.strategy = random.randint(0,4)
            #other_agent.strategy = 0 #random.randint(0,4)
        self.memory.append(other_agent.strategy)
        other_agent.memory.append(self.strategy)


    def step(self):
        self.interation()

class RandomActivation(mesa.time.RandomActivation):
    def step(self):
        for agent in self.agent_buffer(shuffled=True):
            agent.step()
        for agent in self.agent_buffer(shuffled=True):
            agent.learn()
        self.steps += 1
        self.time += 1


class Model(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N):
        self.num_agents = N


        self.schedule = RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)
        self.datacollector = mesa.datacollection.DataCollector(
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
        self.schedule.step()
def Parallel(func,params):
    with Pool(10) as p:
        ret_list = p.map(func,params)
model = Model(50)
for i in range(100):
    model.step()
record = model.datacollector.get_model_vars_dataframe()
a = model.datacollector.get_agent_vars_dataframe()
# a.plot()
record.plot()
plt.show()