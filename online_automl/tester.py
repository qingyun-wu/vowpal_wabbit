#TODO:
# 1. Write a naive AutoVW: use FIFO, and fixed search space to test AutoVW 
# 2. Get HyperBand working: use a fixed search space. 
# 3. Write our algorithm. 



import numpy as np
import argparse
from learner import AutoVW
from vowpalwabbit import pyvw
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

def plot_obj(obj_list, alias='reward', vertical_list=None):
    avg_list = [obj_list[i]/i for i in range(1, len(obj_list))]
    online_avg_loss = [(obj_list[i+100] -obj_list[i])/100 for i in range(len(obj_list)-100)]
    # plt.plot(range(len(online_avg_loss)), online_avg_loss, label = alias)
    # plt.plot(range(len(obj_list)), obj_list, label = alias)
    total_obs = len(avg_list)
    warm_starting_point = int(len(avg_list)*0.05)
    plt.plot(range(len(avg_list[warm_starting_point:])), avg_list[warm_starting_point:], label = alias)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('avg loss', fontsize=14)
    plt.legend()
    if vertical_list:
        for v in vertical_list:
            plt.axvline(x=v)
    # plt.ylim([0.2,3])
    # plt.ylim([0,5])
    # plt.xlim([10000,15000])
    # fig_name = './results/' + alias + '.pdf'
    # plt.savefig(fig_name)
    online_avg_loss = [(obj_list[i+100] -obj_list[i])/100 for i in range(len(obj_list)-100)]

def get_ns_feature_dim_from_vw_example(vw_example):
    ns_feature_dim = {}
    vw_e = vw_example[0]
    data = vw_e.split('|')
    for i in range(1, len(data)):
        logger.debug('name space feature dimension%s', data)
        if ':' in data[i]:
            ns, feature = data[i].split(' ')
            feature_dim = len(feature.split(':'))-1
        else:
            data_split = data[i].split(' ')
            ns = data_split[0]
            feature_dim = len(data_split)-1
        if len(ns) ==1:
            ns_feature_dim[ns] = feature_dim
    logger.debug('name space feature dimension%s', ns_feature_dim)
    return ns_feature_dim

def online_learning_loop(iter_num, vw_example, Y, vw_alg, name = ''):
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
        vw_x = vw_example[i]
        y =  Y[i] #TODO: do we need y? vw_example already include x and y
        if 'auto' in name:
            # loss_pre = vw_alg.incumbent_vw.get_sum_loss()
            y_pred= vw_alg.predict(vw_x)  
            vw_alg.learn(vw_x,y) 
            sum_loss = vw_alg.get_sum_loss()
            # loss = vw_alg.incumbent_vw.get_sum_loss() - loss_pre
            cum_loss += mean_squared_error([y_pred], [y]) 
            # cum_loss += mean_absolute_error([y_pred], [y]) 
            cumulative_loss_list.append(cum_loss)
        elif 'test' in name:
            y_pred= vw_alg.predict(vw_x)  
            vw_alg.learn(vw_x) 
            sum_loss = vw_alg.get_sum_loss() #- loss_pre 
            # TODO: mean_squared_loss is not exactly the same as that in vw
            cum_loss += mean_squared_error([y_pred], [y]) 
            # cumulative_loss_list.append(cum_loss)
            cumulative_loss_list.append(sum_loss)
            # print('sum', sum_loss, cum_loss)
        else:
            y_pred= vw_alg.predict(vw_x)  
            vw_alg.learn(vw_x) 
            sum_loss = vw_alg.get_sum_loss() #- loss_pre 
            # TODO: mean_squared_loss is not exactly the same as that in vw
            cum_loss += mean_squared_error([y_pred], [y]) 
            # cum_loss += mean_absolute_error([y_pred], [y]) 
            # cumulative_loss_list.append(cum_loss)
            cumulative_loss_list.append(cum_loss)
            # print('sum', sum_loss, cum_loss)

        # alg.finish_example(vw_x)
    # print(cumulative_loss_list)
    return cumulative_loss_list

if __name__=='__main__':
    inf_num = np.inf
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_num', metavar='iter_num', type = int, 
        default=1500, help="total iteration number")
    parser.add_argument('-policy_budget', '--policy_budget', metavar='policy_budget', 
    type = int, default= 5, help="budget for the policy that can be evaluated")
    parser.add_argument('-min_resource', '--min_resource', metavar='cost_budget', 
    type = float, default= 20, help="budget for the computation resources that can be evaluated")
    parser.add_argument('-dataset', '--dataset', metavar='dataset', 
    type = str, default= 'simulation', help="get dataset")
    parser.add_argument('-inter_order', '--inter_order', metavar='inter_order', 
    type = int, default= 3, help="inter_order")

    args = parser.parse_args()

    exp_alias = str(args.dataset) + '_' + str(args.min_resource) #+ '_interOrder_' + str(args.inter_order)
    log_file_name = './logs/' + exp_alias + '.log'
    #set up the learning environment, which can generate the learning examples
    #currently from simulation (can also constructed from dataset)
    #TODO: --cbify

    #Generate data
    from data import get_data
    
    logging.basicConfig(filename=log_file_name, format='%(asctime)s %(name)s:%(levelname)s:%(message)s',
        filemode="w", level=logging.DEBUG)
    # logger = logging.getLogger('./logs/test.log')
    X, Y, vw_example = get_data(args.iter_num, data_source = args.dataset, vw_format=True)
    namespace_feature_dim = get_ns_feature_dim_from_vw_example(vw_example)
    fixed_hp_config = {'l2': 0.0,  'loss_function': 'squared'}
    
    if 'simu' in args.dataset:
        fixed_hp_config = {'l2': 0.0, 'learning_rate': 0.1, 'alg': 'supervised',  'loss_function': 'squared'}
    else:
        fixed_hp_config = {'l2': 0.0,   'loss_function': 'squared'} # 'learning_rate': 0.1, 'alg': 'supervised',
    #instantiate several vw learners (as baselines) and an AutoOnlineLearner
    alg_dic = {}
    alg_dic['oracle'] = pyvw.vw(q=['ab','ac','cd'],   **fixed_hp_config)  #cubic =['acb'] ,
    alg_dic['naive'] = pyvw.vw(**fixed_hp_config)
    #TODO: how to get loss? progressive validation
    auto_method_alias = 'auto_' + exp_alias
    alg_dic[auto_method_alias + 'order_2'] = AutoVW(min_resource_budget = args.min_resource,#args.cost_budget, 
        policy_budget = args.policy_budget, 
        namespace_feature_dim = namespace_feature_dim, 
        inter_order = 2,
        fixed_hp_config = fixed_hp_config)
    alg_dic[auto_method_alias + 'order_3'] = AutoVW(min_resource_budget = args.min_resource,#args.cost_budget, 
        policy_budget = args.policy_budget, 
        namespace_feature_dim = namespace_feature_dim, 
        inter_order = 3,
        fixed_hp_config = fixed_hp_config)
    import matplotlib.pyplot as plt
    
    for alg_name, alg in alg_dic.items():
        iter_num = min(args.iter_num, len(Y)-1)
        cumulative_loss_list = online_learning_loop(iter_num, vw_example, Y, alg, name = alg_name)
        plot_obj(cumulative_loss_list, alias= alg_name)
    
    alias = 'loss_' + exp_alias
    fig_name = './results/' + alias + '.pdf'
    plt.savefig(fig_name)
    
    

## command lines to run exp
# conda activate vw
# python tester.py -i 10000 -c 200 >res.txt


# python tester.py  -i 10000 -c 200 -d 572 >res.txt
# python tester.py -i 10000 -c 200 -d 218  >res.txt
# python tester.py -i 1000 -c 200 -d simulation  >res.txt