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
        self.payoff_matrix = np.array([
            [self.R,self.S],
            [self.T,self.P]
        ]).T
        self.num_strategies = 2
        self.stra_distri = [50,50]

    def set_payoff_matrix_PD_with_Punish(self):
        self.payoff_matrix = np.array([
            [self.R, self.R, self.S - self.epssilon1, self.S - self.epssilon1 - self.delta1],
            [self.R, self.R, self.S, self.S - self.delta1],
            [self.T-self.delta1, self.T, self.P, self.P],
            [self.T - self.epssilon1 - self.delta1, self.T - self.epssilon1, self.P, self.P],

            ]
        ).T
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
        ).T
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


    def fa(self,k,paa,pab):
        n = self.population
        fa = ((k-1)/n)*paa+((n-k)/n)*pab
        return fa

    def fb(self,k,pba,pbb):
        n = self.population
        fb = (k/n)*pba+((n-k-1)/n)*pbb
        return fb

    def t(self,k,a,b):
        p =self.payoff_matrix
        fa = self.fa(k,p[a,a],p[a,b])
        fb = self.fb(k,p[b,a],p[b,b])
        n = self.population
        tplus = (k)*(n-k)/(n*n)/(1+math.exp(-self.beta*(fa-fb)))
        tminus = (k) * (n - k) / (n * n) / (1 + math.exp(self.beta*(fa - fb)))

        return tminus/tplus


    def rho(self,a,b):
        tmp1 = 1
        n = self.population
        for i in range(1,n+1):
            tmp = 1
            for j in range(1,i):
                tmp = tmp*self.t(j,a,b)
            tmp1 = tmp+tmp1

        return 1/(tmp1*(self.num_strategies-1))
    def scenario(self):

        # self.set_payoff_matrix_PD_with_commitment_punishment()
        # self.set_payoff_matrix_PD_with_Commitment()
        self.set_payoff_matrix_PD()
        self.transition_probability = np.ones((self.num_strategies, self.num_strategies))
        N = self.population

        for i in range(self.num_strategies):
            for j in range(self.num_strategies):
                if i != j:

                    self.transition_probability[i][j] = self.rho(i,j)
                    # print(self.transition_probability[i][j])
                    self.transition_probability[i][i] = self.transition_probability[i][i]-self.transition_probability[i][j]




        x = np.matrix(self.transition_probability.T)
        e,v = np.linalg.eig(x)
        # print(e)
        # print(v[:,0]/np.sum(v[:,0]))
        print(np.around(self.transition_probability, 3))



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



