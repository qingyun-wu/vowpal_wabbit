
import numpy as np
import argparse
from learner import VWLearner, ConfiguredVWLearner, AutoVWLearner
from vowpalwabbit import pyvw
import matplotlib.pyplot as plt

def plot_obj(obj_list, alias='reward', vertical_list=None):
    plt.plot(range(len(obj_list)), obj_list, label = alias)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('cumulative loss', fontsize=14)
    plt.legend()
    if vertical_list:
        for v in vertical_list:
            plt.axvline(x=v)
    # plt.ylim([0,1])
    # fig_name = './results/' + alias + '.pdf'
    # plt.savefig(fig_name)


def online_learning_loop(iter_num, vw_example, alg):
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
    for i in range(iter_num):
        # y =  Y[i] #TODO: do we need y? vw_example already include x and y
        vw_x = pyvw.example(alg, vw_example[i])
        # y_pred= alg.predict(vw_x)  
        alg.learn(vw_x)
        # loss = vw_x.get_loss()
        cumulative_loss_list.append(alg.get_sum_loss())
        alg.finish_example(vw_x)
    return cumulative_loss_list


if __name__=='__main__':
    inf_num = np.inf
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_num', metavar='iter_num', type = int, 
        default=1500, help="total iteration number")
    parser.add_argument('-b', '--policy_budget', metavar='policy_budget', 
    type = float, default= None, help="budget for the policy that can be evaluated")
    parser.add_argument('-c', '--cost_budget', metavar='cost_budget', 
    type = float, default= 100, help="budget for the computation resources that can be evaluated")
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
    alg_dic['oracle'] = Learner(q=['ab', 'ac', 'cd'], **fixed_hp_config)
    alg_dic['naive'] = Learner(**fixed_hp_config)
    #specify a feature map generator
    # problem = FSSProblem()
    # alg_dic['auto'] = AutoOnlineLearner(fm_generator = fm_generator, 
    #     cost_budget = args.cost_budget, policy_budget = args.policy_budget, learner = ConfiguredLearner, **fixed_hp_config)
    
    import matplotlib.pyplot as plt
    for alg_name, alg in alg_dic.items():
        cumulative_loss_list = online_learning_loop(args.iter_num, vw_example, alg)
        plot_obj(cumulative_loss_list, alias= alg_name)
    alias = 'loss_all'
    fig_name = './results/' + alias + '.pdf'
    plt.savefig(fig_name)
    
    

## command lines to run exp
# conda activate vw
# python tester.py -i 30


