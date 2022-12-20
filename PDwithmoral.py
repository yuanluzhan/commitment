import numpy as np
import math
import collections
import random

class Moral():
    def __init__(self):
        self.population = 1000
        self.imitation_strength =0.1
        self.beta = 0.1
        self.epsilon1 = 1
        self.epsilon2 = 1
        self.delta1 = 3
        self.delta2 = 3
        self.R = 3
        self.S = 0
        self.T = 4
        self.P = 1
        self.morals = np.random.normal(loc=0, scale=1, size=self.population)
        self.morals = np.zeros(self.population)


    def set_payoff_matrix_PD(self):
        self.payoff_matrix = np.array([
            [3,0],
            [4,1]
        ])
        self.num_strategies = 2
        self.moral_cof = [1, -1]
        self.individuals = np.random.random_integers(0,self.num_strategies-1, self.population)
        self.update_dis()

    def set_payoff_matrix_PD_with_Punish(self):
        self.payoff_matrix = np.array([
            [self.R, self.R, self.S - self.epsilon1, self.S - self.epsilon1 - self.delta1],
            [self.R, self.R, self.S, self.S - self.delta1],
            [self.T-self.delta1, self.T, self.P, self.P],
            [self.T - self.epsilon1 - self.delta1, self.T - self.epsilon1, self.P, self.P],

            ]
        )

        # self.moral_cof = [0,0,0,0]
        self.moral_cof = [1,1,-1,-1]
        self.num_strategies = 4
        self.individuals = np.random.random_integers(0,self.num_strategies-1,self.population)
        # print(self.individuals)
        self.update_dis()

    def set_payoff_matrix_PD_with_Commitment(self):
        self.payoff_matrix = np.array(
            [
                [self.R - self.epsilon2/2, self.R - self.epsilon2, 0,  self.S-self.epsilon2+self.delta2, self.R - self.epsilon2],
                [self.R, self.R, self.S, self.S, self.S],
                [0, self.T, self.P, self.P, self.P],
                [self.T - self.delta2, self.T, self.P, self.P, self.P],
                [self.T, self.T, self.P, self.P, self.P]
            ]
        )
        self.num_strategies = 5
        self.moral_cof = [1, 1, -1, -1,1]
        self.individuals = np.random.random_integers(0, self.num_strategies - 1, self.population)
        # print(self.individuals)
        self.update_dis()

    def set_payoff_matrix_PD_with_commitment_punishment(self):
        self.payoff_matrix = np.array(
            [
                [self.R - self.epsilon2/2, self.R - self.epsilon2, 0, self.R-self.epsilon2, 0 ,self.R-self.epsilon2, self.S-self.epsilon2+self.delta2-self.delta1],
                [self.R, self.R, self.S, self.R, self.S-self.delta1,self.S-self.delta1, self.S-self.delta1],
                [0, self.T, self.P, self.T-self.delta1, self.P, self.P, self.P],
                [self.R, self.R, self.R-self.epsilon1,self.R,self.S-self.epsilon1-self.delta1, self.S-self.epsilon1-self.delta1,self.S-self.epsilon1-self.delta1],
                [0, self.T-self.epsilon1, self.P, self.T-self.epsilon1-self.delta1, self.P, self.P, self.P],
                [self.R, self.T-self.epsilon1, self.P, self.T-self.epsilon1-self.delta1, self.P, self.P, self.P],
                [self.T-self.delta2-self.epsilon2,self.T-self.epsilon1, self.P, self.T-self.epsilon1-self.delta1, self.P, self.P, self.P]
            ]
        )
        self.num_strategies = 7
        self.individuals = np.random.random_integers(0, self.num_strategies - 1, self.population)
        # print(self.individuals)
        self.moral_cof = [1, 1, 1, -1, -1,-1,-1]
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
                    tmp = tmp + (self.stra_dis[i]-1) * self.payoff_matrix[i][j]
                else:
                    tmp = tmp + self.stra_dis[j]*self.payoff_matrix[i][j]
            fitness[i] = tmp/(self.population-1)
            self.fitness = fitness
        # print(self.fitness)

        self.transition_matrix = np.ones((self.num_strategies,self.num_strategies))
        for i in range(self.num_strategies):
            for j in range(self.num_strategies):
                if i!=j:
                    self.transition_matrix[i][j] = (self.stra_dis[j])/(self.population)/(1+math.exp(self.beta*(self.fitness[i]-self.fitness[j])))
                    self.transition_matrix[i][i] = self.transition_matrix[i][i]- self.transition_matrix[i][j]
        # print(self.transition_matrix)

    def transist(self):
        # calculate the transition matrix
        index = np.arange(0,self.population)
        random.shuffle(index)
        for i in range(int(self.population/2)):
            a = index[i]
            b = index[i+int(self.population/2)]
            a_strategy = self.individuals[a]
            b_strategy = self.individuals[b]
            payoffa = self.payoff_matrix[a_strategy, b_strategy]
            payoffb = self.payoff_matrix[b_strategy, a_strategy]
            transitiona = 1/(1+math.exp(-self.beta*(payoffb-(payoffa+self.moral_cof[a_strategy]*self.morals[a]))))
            self.individuals[a] = np.random.choice([a_strategy,b_strategy],p = [1-transitiona,transitiona])
            transitionb = 1 / (1 + math.exp(-self.beta * (payoffa - (payoffb + self.moral_cof[b_strategy]*self.morals[b]))))
            self.individuals[b] = np.random.choice([a_strategy,b_strategy],p = [transitionb,1-transitionb])

    def simulation(self):
        # calculate the transition matrix

        self.set_payoff_matrix_PD_with_commitment_punishment()
        for iter in range(1000):
            self.transist()
            self.update_dis()
            print(self.stra_dis)



if __name__ == "__main__":
    a = Moral()
    a.simulation()