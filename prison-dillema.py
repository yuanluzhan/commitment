import numpy as np
import math

class PD():
    def __init__(self):
        self.population=100
        self.imitation_strength = 0.1

        self.beta = 0.1
        self.epssilon1 = 1
        self.epssilon2 = 1
        self.delta1 = 2
        self.delta2 = 3
        self.R = 3
        self.S = 1
        self.T = 4
        self.P = 0



    def set_payoff_matrix_PD(self):
        self.payoff_matrix[0][0] = self.R
        self.payoff_matrix[0][1] = self.S
        self.payoff_matrix[1][0] = self.T
        self.payoff_matrix[1][1] = self.P

    def set_payoff_matrix_PD_with_Punish(self):
        self.payoff_matrix = np.array([
            [self.R, self.R, self.S - self.epssilon1, self.S - self.epssilon1 - self.delta1],
            [self.R, self.R, self.S, self.S - self.delta1],
            [self.T-self.delta1, self.T, self.P, self.P],
            [self.T - self.epssilon1 - self.delta1, self.T - self.epssilon1, self.P, self.P],

            ]
        )
        self.num_strategies = 4
        self.stra_distri = [25,25,25,25]

    def set_payoff_matrix_PD_with_Commitment(self):
        self.payoff_matrix = np.array(
            [
                [self.R - self.epssilon2/2, self.R - self.epssilon2, 0,  self.S-self.epssilon2+self.delta2, self.S - self.epssilon2],
                [self.R, self.R, self.S, self.S, self.S],
                [0, self.T, self.P, self.P, self.P],
                [self.T - self.delta2, self.T, self.P, self.P, self.P],
                [self.T, self.T, self.P, self.P, self.P]
            ]
        )
        self.num_strategies = 5
        self.stra_distri = [20, 20, 20, 20, 20]

    def set_payoff_matrix_PD_with_commitment_punishment(self):
        self.payoff_matrix = np.array(
            [
                [self.R - self.epssilon2/2, self.R - self.epssilon2, 0, self.R-self.epssilon2, 0 ,self.R-self.epssilon2, self.S-self.epssilon2+self.delta2-self.delta1],
                [self.R, self.R, self.S, self.R, self.S-self.delta1,self.S-self.delta1, self.S-self.delta1],
                [0, self.T, self.P, self.T-self.delta1, self.P, self.P, self.P],
                [self.R, self.R, self.R-self.epssilon1,self.R,self.S-self.epssilon1-self.delta1, self.S-self.epssilon1-self.delta1,self.S-self.epssilon1-self.delta1],
                [0, self.T-self.epssilon1, self.P, self.T-self.epssilon1-self.delta1, self.P, self.P, self.P],
                [self.R, self.T-self.epssilon1, self.P, self.T-self.epssilon1-self.delta1, self.P, self.P, self.P],
                [self.T-self.delta2-self.epssilon2,self.T-self.epssilon1, self.P, self.T-self.epssilon1-self.delta1, self.P, self.P, self.P]
            ]
        )
        self.num_strategies = 7
        self.stra_distri = [10,10,10,10,20,20,20]


    def scenario(self):

        # self.set_payoff_matrix_PD_with_commitment_punishment()
        self.set_payoff_matrix_PD_with_Punish()
        self.transition_probability = np.zeros((self.num_strategies, self.num_strategies))
        N = self.population


        #Two
        for i in range(self.num_strategies):
            for j in range(self.num_strategies):
                if i != j:
                    tmp = 1
                    tmp1 = 1
                    a = []
                    b = []
                    for k in range(1,self.population - 1):

                        fitnessA = ((k - 1) * self.payoff_matrix[i][i] + (N - k) * self.payoff_matrix[i][j] ) / (N - 1)
                        fitnessB = (k * self.payoff_matrix[j][i] + (N - k - 1) * self.payoff_matrix[j][j]) / (N - 1)
                        T_up = (N - k) * k / (N * N * (1 + math.exp(-self.beta * (fitnessA-fitnessB))))
                        T_down = (N - k) * k / (N * N * (1 + math.exp(self.beta * (fitnessA-fitnessB))))
                        a.append(T_down)
                        b.append(T_up)
                        tmp = tmp * (T_down / T_up)
                        tmp1 = tmp1 + tmp

                    self.transition_probability[i][j] = 1 / (tmp1*(self.num_strategies-1))
        for i in range(self.num_strategies):
            self.transition_probability[i][i] = 0
            self.transition_probability[i][i] = 1 - np.sum(self.transition_probability[i, :])
        print(np.around(self.transition_probability,3))



        A = self.transition_probability-np.eye(self.num_strategies)+np.ones((self.num_strategies,self.num_strategies))
        a = np.ones((1,self.num_strategies)).dot(np.linalg.inv(A))
        print(np.around(a,4))

        # for iteration in range(100):
        #     self.stra_distri = np.sum(self.stra_distri*self.transition_probability,axis=0)
        #     if np.sum(self.stra_distri)!=100:
        #         self.stra_distri = self.stra_distri*(100/np.sum(self.stra_distri))
        #     print(np.around(self.stra_distri))
        # print(np.around(self.transition_probability,2))


if __name__ == "__main__":
    a = PD()
    a.scenario()



