import os
import numpy as np
import time
from joblib import Parallel,delayed
import pandas as pd
from onlinecollaboration import Model
import math

def para_generat(analyze_factor):

    para_payoff = {
        'c': 0.4,
        'a': 0.3,
        'b': 0.2,
        'd': 0.1,
        'epsilon': 0.001,
        'delta': 0.05
    }
    para_game_settings = {
        "num_all_players": 100,
        "num_participants": 1
    }
    para_individual_information = {
        "para_individual": 0.01,
        "rationality": 5,
        "history_length": 10
    }
    para_init_distribution = {
        "strategy": "random",
        "ind_type": [0.08, 0.02, 0.9]
    }
    # analyze_factor = "init_distribution"


    if analyze_factor == "payoff":
        for d in np.arange(0.1, 0.2, 0.1):
            for b in np.arange(d + 0.1, 0.3, 0.1):
                for a in np.arange(b + 0.1, 0.4, 0.1):
                    for c in np.arange(a + 0.1, 0.5, 0.1):
                        for epsilon in np.arange(0.001, 0.02, 0.001):
                            for delta in np.arange(0.01, 0.1, 0.01):
                                para_payoff = {
                                    'c': np.around(c, 2),
                                    'a': np.around(a, 2),
                                    'b': np.around(b, 2),
                                    'd': np.around(d, 2),
                                    'epsilon': np.around(epsilon, 3),
                                    'delta': np.around(delta, 3)
                                }
                                yield [para_payoff, para_game_settings, para_individual_information, para_init_distribution]
    if analyze_factor == "game_settings":
        for num_all_players in range(100,1000,100):
            for num_participants in range(10,101,10):
                para_game_settings = {
                    "num_all_players" : num_all_players,
                    "num_participants" : num_participants
                }
                yield [para_payoff, para_game_settings, para_individual_information, para_init_distribution]
    if analyze_factor == "individual_information":
        for ind_distri_para in np.arange(0.005, 0.05, 0.005):
            for rationality in range(1,10,1):
                for history_length in range(5,21,5):
                    para_individual_information = {
                        "para_individual" : ind_distri_para,
                        "rationality" : rationality,
                        "history_length" : history_length
                    }
                    yield [para_payoff, para_game_settings, para_individual_information, para_init_distribution]
    if analyze_factor == "init_distribution":
        for strategy in ["random", "D", "Free"]:
            for x in []:
                for y in []:
                    z = 1 - x - y
                    x = np.around(x,2)
                    y = np.around(y,2)
                    z = np.around(z,2)
                    para_init_distribution = {
                        "strategy" : strategy,
                        "ind_type" : [x,y,z]
                    }
                    yield [para_payoff, para_game_settings, para_individual_information, para_init_distribution]


def run(para, analyze_factor, num_trails):
    para_payoff, para_game_settings, para_individual_information, para_init_distribution = para
    if analyze_factor == "payoff":
        path1 = 'result/' + str(analyze_factor) + '/' + str(list(para_payoff.values()))
    elif analyze_factor == "game_settings":
        path1 = 'result/' + str(analyze_factor) + '/' + str(list(para_game_settings.values()))
    elif analyze_factor == "individual_information":
        path1 = 'result/' + str(analyze_factor) + '/' + str(list(para_individual_information.values()))
    elif analyze_factor == "init_distribution":
        path1 = 'result/' + str(analyze_factor) + '/' + str(para_init_distribution["strategy"]) +str(para_init_distribution["ind_type"])


    for trail in range(num_trails):
        model = Model(para)
        for i in range(5000):
            model.step()
        model.agent_keys = np.array(model.agent_keys)
        model.datacollector.collect(model)
        record = model.datacollector.get_model_vars_dataframe()
        # Agent_keys = pd.DataFrame(record['Agent_keys'].values[0])
        # Agent_information = pd.DataFrame(record['Agent_information'].values[0])
        # Agent1_strategy = pd.DataFrame(record['Agent1_strategy'].values[0])
        # Agent2_strategy = pd.DataFrame(record['Agent2_strategy'].values[0])
        a = model.datacollector.get_agent_vars_dataframe()
        path = path1 + '/' + str(trail)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        a.to_csv(path + '/agent_result.csv')
        # Agent1_strategy.to_csv(path + '/agent1_strategy.csv')
        # Agent2_strategy.to_csv(path + '/agent2_strategy.csv')
        # Agent_keys.to_csv(path + '/agent_keys.csv')
        # Agent_information.to_csv(path + '/agent_information.csv')

        np.save(path + '/para.npy', para)




if __name__ == '__main__':
    analyze_factor = "individual_information"
    print(analyze_factor)

    #game_settings   individual_information  init_distribution
    paras = para_generat(analyze_factor)
    num_trails = 100

    print(time.time())
    # Parallel(n_jobs=1)(delayed(math.sqrt)(i ** 2) for i in range(10))
    Parallel(n_jobs = 100)(delayed(run)(para,analyze_factor,num_trails) for para in paras)
    print(time.time())