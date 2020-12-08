#TODO:
# 1. Write a naive AutoVW: use FIFO, and fixed search space to test AutoVW 
# 2. Get HyperBand working: use a fixed search space. 
# 3. Write our algorithm. 



import numpy as np
import argparse
from learner import AutoVW
from vowpalwabbit import pyvw
import matplotlib.pyplot as plt

def plot_obj(obj_list, alias='reward', vertical_list=None):
    avg_list = [obj_list[i]/i for i in range(1, len(obj_list))]
    online_avg_loss = [(obj_list[i+100] -obj_list[i])/100 for i in range(len(obj_list)-100)]
    # plt.plot(range(len(online_avg_loss)), online_avg_loss, label = alias)
    # plt.plot(range(len(obj_list)), obj_list, label = alias)
    plt.plot(range(len(avg_list)), avg_list, label = alias)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('avg loss', fontsize=14)
    plt.legend()
    if vertical_list:
        for v in vertical_list:
            plt.axvline(x=v)
    plt.ylim([0.4,1.2])
    # fig_name = './results/' + alias + '.pdf'
    # plt.savefig(fig_name)
    online_avg_loss = [(obj_list[i+100] -obj_list[i])/100 for i in range(len(obj_list)-100)]




def online_learning_loop(iter_num, vw_example, vw_alg, name = ''):
    """ Implements the online learning loop.
    Args:
        iter_num (int): The total number of iterations
        vw_example (vw_example): A list of vw examples
        alg (alg instance): An algorithm instance has the following functions:
            - alg.learn(example)
            - alg.predict(example)
            - alg.get_loss()
            - alg.get_sum_loss()
    Outputs:
        cumulative_loss_list (list): the list of cumulative loss from each iteration.
            It is returned for the convenience of visualization.
    """
    cumulative_loss_list = []
    cum_loss = 0
    import copy
    vw_example = copy.deepcopy(vw_example)
    for i in range(iter_num):
        
        # y =  Y[i] #TODO: do we need y? vw_example already include x and y
        if 'auto' in name:
            # loss = vw_x.get_loss()
            # print( 'pre', vw_alg.incumbent_vw.get_sum_loss())
            loss_pre = vw_alg.incumbent_vw.get_sum_loss()
            # print(vw_example[i])
            vw_x = vw_example[i]
            #TODO: check how to convert to vw example
            # vw_x = pyvw.example(vw_alg.incumbent_vw(), vw_example[i])
            # y_pred= vw_alg.incumbent_vw.predict(vw_x)  
            y_pred= vw_alg.predict(vw_x)  
            vw_alg.learn(vw_x) 
            loss = vw_alg.incumbent_vw.get_sum_loss() - loss_pre
            print(loss)
            if loss <0:
                print('NEGATIVE loss', loss, vw_alg.incumbent_vw.get_sum_loss(), loss_pre)
            else:
                cum_loss += loss
            cumulative_loss_list.append(cum_loss)
            # cumulative_loss_list.append(vw_alg.incumbent_vw.get_sum_loss())
        else:
            vw_x = vw_example[i]
            # vw_x = pyvw.example(vw_alg, vw_example[i])
            loss_pre = vw_alg.get_sum_loss()
            y_pred= vw_alg.predict(vw_x)  
            vw_alg.learn(vw_x) 
            loss = vw_alg.get_sum_loss() - loss_pre 
            if loss >0:
                cum_loss +=loss
                cumulative_loss_list.append(cum_loss)
        # alg.finish_example(vw_x)
    # print(cumulative_loss_list)
    return cumulative_loss_list


if __name__=='__main__':
    inf_num = np.inf
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_num', metavar='iter_num', type = int, 
        default=1500, help="total iteration number")
    parser.add_argument('-b', '--policy_budget', metavar='policy_budget', 
    type = float, default= 3, help="budget for the policy that can be evaluated")
    parser.add_argument('-c', '--cost_budget', metavar='cost_budget', 
    type = float, default= 50, help="budget for the computation resources that can be evaluated")
    args = parser.parse_args()

    #set up the learning environment, which can generate the learning examples
    #currently from simulation (can also constructed from dataset)
    #TODO: need to construct the environment from a supervised dataset
    #TODO: --cbify

    #Generate data
    from data import get_data
    X, Y, vw_example = get_data(args.iter_num, vw_format=True)
    fixed_hp_config = {'l2': 0.1, 'loss_function': 'squared'}
    #instantiate several vw learners (as baselines) and an AutoOnlineLearner
    alg_dic = {}
    alg_dic['oracle'] = pyvw.vw(q=['ab', 'ac',  'cd'], **fixed_hp_config) #'ab', 'ac',
    alg_dic['naive'] = pyvw.vw(**fixed_hp_config)
    #specify a feature map generator
    # alg_dic['auto'] = AutoVW(fm_generator = fm_generator, 
    #     cost_budget = args.cost_budget, policy_budget = args.policy_budget, **fixed_hp_config)
    alg_dic['auto'] = AutoVW(min_resource_budget = args.cost_budget,#args.cost_budget, 
        policy_budget = args.policy_budget, 
        hp2tune_init_config = {'q': ['a', 'b', 'c', 'd', 'e']}, 
        fixed_hp_config = fixed_hp_config)
    import matplotlib.pyplot as plt
    for alg_name, alg in alg_dic.items():
        cumulative_loss_list = online_learning_loop(args.iter_num, vw_example, alg, name = alg_name)
        plot_obj(cumulative_loss_list, alias= alg_name)
    
    alias = 'loss_all'
    fig_name = './results/' + alias + '.pdf'
    plt.savefig(fig_name)
    
    

## command lines to run exp
# conda activate vw
# python tester.py -i 30


