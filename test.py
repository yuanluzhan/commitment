import time
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib
from joblib import Parallel,delayed
import numpy
import numpy as np
import pandas as pd
import seaborn as sns


def generate_analysis_para():
    for strategy in ["D"]: #["random", "D", "Free"]:
        for x in np.arange(0, 0.21, 0.05):
            for y in np.arange(0, x+0.01, 0.05):
                z = 1 - x - y
                x = np.around(x, 2)
                y = np.around(y, 2)
                z = np.around(z, 2)
                yield [strategy, x, y, z]

def data_process(strategy,x,y,z):
        storage = []
        for trail in range(100):
            path = "F:/init_distribution/"+str(strategy)+str([x,y,z])+"/" + str(trail)
            # agent1_strategy = pd.read_csv(path + "/agent1_strategy.csv").values[:, 1:]
            # agent2_strategy = pd.read_csv(path + "/agent2_strategy.csv").values[:, 1:]
            record = pd.read_csv(path+"/agent_information.csv")
            tmp = record.values[:,1:]
            s1 = []
            for i in range(len(tmp)):
                s1.append([list(tmp[i][1:]).count(j) for j in range(5)])
            storage.append(s1)

        storage = np.array(storage)
        np.save("F:/init_distribution/"+str(strategy)+str([x,y,z])+"/data.npy",storage)

def plot_timeseries(strategy, x, y, z):
    storage = np.load("F:/init_distribution/"+str(strategy)+str([x,y,z])+"/data.npy")
    average = np.mean(storage, axis=0)

    std = np.std(storage, axis=0)
    plt.plot(average[0:5000:5])
    plt.title(str(strategy)+str([x,y,z]))
    plt.legend(labels=["Com", "C", "D", "Fake", "Free"], loc=0)
    plt.show()
    return np.mean(average[-1000:,0])

def run(para):
    strategy, x, y, z = para
    result = plot_timeseries(strategy, x, y, z)
    return result

def smooth(interval, window_size):
    tmp = interval
    for i in range(0, interval.shape[0]):
        for j in range(interval.shape[1]):
            tmp[i,j] = np.mean(interval[i:min(i+window_size,interval.shape[0]),j])
    return tmp

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['mathtext.fontset']='stix'
def plot_initial(paras):
    i = 0
    fig, ax = plt.subplots(1,1,figsize = [40,30],sharex=True,sharey=True)
    fig.tight_layout(pad=6)
    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    for para in paras:

        strategy, x, y, z = para
        storage = np.load("F:/init_distribution/" + str(strategy) + str([x, y, z]) + "/data.npy")
        average = np.mean(storage, axis=0)
        std = np.std(storage, axis=0)
        i = i + 1
        if i == 4:
            i = i + 1
        y_smooth = smooth(interval=average, window_size=100)
        plt.subplot(1, 1, i)

        plt.plot(y_smooth, linewidth=4)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        plt.title(str([x, y, z]))
        tmp[4 - int(y * 20)][int(x * 20)] = np.mean(average[-1000:, 0])
    fig.legend(labels=["Com", "C", "D", "Fake", "Free"], loc=1,
               bbox_to_anchor=(0.93, 0.9),
               prop={'size': 40}
               )
    fig.suptitle('Initial with all D strategy', fontsize=60)
    fig.supxlabel("Time", fontsize=50)
    fig.supylabel("Average players numbern  ", fontsize=50)
    raise
    # ylz = plt.subplot(4, 4, 4)
    # ylz.axis('off')
    plt.show()


if __name__ == '__main__':
    tmp = np.zeros([5,5])
    paras = generate_analysis_para()
    for i in range(5):
        for j in range(5):
            tmp[i,j] = np.nan


    i = 0
    # fig, ax = plt.subplots(4,4,figsize = [40,30],sharex=True,sharey=True)
    # fig.tight_layout(pad=6)
    # plt.subplots_adjust(wspace=0.2,hspace=0.2)
    for para in paras:

        strategy, x, y, z = para
        storage = np.load("F:/init_distribution/" + str(strategy) + str([x, y, z]) + "/data.npy")
        average = np.mean(storage, axis=0)
        std = np.std(storage, axis=0)
        y_smooth = smooth(interval=average, window_size=100)
        # print(storage.shape)


        # plt.plot(storage[5,:,:], linewidth=1)
        plt.plot(y_smooth)
        plt.title("average"+str([x, y, z]))
        plt.legend(["Com", "C", "D", "Fake", "Free"], loc=0)
        plt.xlabel("Time")
        plt.ylabel("Num of Agents")
        tmp[4-int(y*20)][int(x*20)] = np.mean(average[-1000:,0])
        plt.show()
        raise Exception("s")



    # print(result)

    cmap = sns.cubehelix_palette(start=0.2, rot=0.4, gamma=0.5, as_cmap=True)
    sns.heatmap(data=tmp,center=30,cmap=cmap,annot=True,fmt="0.2f",xticklabels=[0,0.05,0.10,0.15,0.20],yticklabels=[0.20,0.15,0.10,0.05,0])
    plt.show()

    # Parallel(n_jobs=1)(delayed(run)(para) for para in paras)
