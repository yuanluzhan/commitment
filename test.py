import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
a = np.array([1.0])
b = np.array([0.1])
print(a+b)
for i in np.arange(0.9,1.09,0.1):
    print(i)
# agent_keys = pd.read_csv('result/[1.0, 0.9, 0.8, 0.7, 0.01, 0.05]/[[0.01], [1], [9, 10]]/agent_keys.csv')
path = "[0.8, 0.6, 0.4, 0.2, 0.01, 0.15]/[[0.01], [1], [9, 10]]"
record = pd.read_csv("result/"+path+"/agent_information.csv")
# agent_result = pd.read_csv("result/[1.0, 0.9, 0.8, 0.7, 0.01, 0.05]/[[0.01], [1], [9, 10]]/agent_result.csv")
tmp = record.values
# print(agent_keys['0'])
tmp1 = tmp[:,1:]

# tmp.to_csv('2.csv')
# ylz = pd.read_csv('2.csv')

s1 = []
for i in range(len(tmp)):
    # print(tmp[i][1:])
    # print(np.bincount(tmp[i][1:]).shape)
    # print([list(tmp[i][1:]).count(j) for j in range(5) ])
    s1.append([list(tmp[i][1:]).count(j) for j in range(5)])

plt.plot(s1, label = ['Com', 'C', 'D', 'fake', 'free'])
plt.title(path)
plt.legend()
plt.show()
print(time.time())