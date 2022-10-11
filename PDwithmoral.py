import numpy as np
import math
import collections

class Moral():
    def __init__(self):
        self.population = 100
        self.imitation_strength =0.1
        self.beta = 0.1
        self.epsilon1 = 1
        self.epsilon2 = 1
        self.delta1 = 3
        self.delta2 = 3
        self.R = 3
        self.S = 1
        self.T = 4
        self.P = 0

    def set_payoff_matrix_PD(self):
        self.payoff_matrix = np.array([
            [2,0],
            [10,1]
        ])
        self.num_strategies = 2
        self.individuals = np.random.random_integers(0, self.num_strategies - 1, self.population)
        self.update_dis()

    def set_payoff_matrix_PD_with_Punish(self):
        self.payoff_matrix = np.array([
            [self.R, self.R, self.S - self.epsilon1, self.S - self.epsilon1 - self.delta1],
            [self.R, self.R, self.S, self.S - self.delta1],
            [self.T-self.delta1, self.T, self.P, self.P],
            [self.T - self.epsilon1 - self.delta1, self.T - self.epsilon1, self.P, self.P],

            ]
        ).T
        self.num_strategies = 4
        self.individuals = np.random.random_integers(0,self.num_strategies-1,self.population)
        # print(self.individuals)
        self.update_dis()

    def update_dis(self):
        tmp = collections.Counter(self.individuals)
        self.stra_dis = np.ones(self.num_strategies)
        for i in range(self.num_strategies):
            self.stra_dis[i] = tmp[i]

    def calculate_fitness(self):
        n = self.population
        fitness = np.zeros(self.num_strategies)
        for i in range(self.num_strategies):
            tmp = 0
            for j in range(self.num_strategies):
                if i==j:
                    tmp = tmp + (self.stra_dis[i]-1)* self.payoff_matrix[i][j]
                else:
                    tmp = tmp + self.stra_dis[j]*self.payoff_matrix[i][j]
            fitness[i] = tmp/(self.population-1)
            self.fitness = fitness
        print(self.fitness)

        self.transition_matrix = np.ones((self.num_strategies,self.num_strategies))
        for i in range(self.num_strategies):
            for j in range(self.num_strategies):
                if i!=j:
                    self.transition_matrix[i][j] = (self.stra_dis[i]*self.stra_dis[j])/(self.population*self.population)/(1+math.exp(self.beta*(self.fitness[i]-self.fitness[j])))
                    self.transition_matrix[i][i] = self.transition_matrix[i][i]- self.transition_matrix[i][j]
        print(self.transition_matrix)

    def transist(self):
        self.update_dis()
        self.calculate_fitness()
        for i in range(self.population):
            p = self.transition_matrix[self.individuals[i],:]
            tmp = np.random.choice(self.num_strategies, p=p)
            self.individuals[i] = tmp

    def simulation(self):
        # calculate the transition matrix
        self.morals = np.random.normal(loc=0, scale=1, size=self.population)
        self.set_payoff_matrix_PD()
        for iter in range(1000):
            self.transist()
            print(self.stra_dis)

if __name__ == "__main__":
    a = Moral()
    a.simulation()