import numpy as np
import argparse
from learner import AutoVW
from vowpalwabbit import pyvw
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import LOG_DIR, PLOT_DIR, WARMSTART_NUM
import logging
import time 
import os
from os import path
import matplotlib.pyplot as plt
from result_log import ResultLogReader,ResultLogWriter
from util import get_y_from_vw_example, get_ns_feature_dim_from_vw_example
logger = logging.getLogger(__name__)
# TODO:
# 2. add config info in file name; 3. setup; 4. how to re-run a particular method
def plot_obj_cumulative(obj_list, alias='reward', vertical_list=None):
    avg_list = [obj_list[i]/i for i in range(1, len(obj_list))]
    plt.plot(range(0, len(avg_list)), avg_list[0:], label = alias)
    plt.xlabel('# of interactions', fontsize=14)
    plt.ylabel('average loss', fontsize=14)
    plt.yscale('log')
    plt.legend()
    if vertical_list:
        for v in vertical_list:
            plt.axvline(x=v)
    online_avg_loss = [(obj_list[i+100] -obj_list[i])/100 for i in range(len(obj_list)-100)]


def plot_obj(obj_list, alias='loss', vertical_list=None):
    avg_list = [sum(obj_list[:i])/i for i in range(1, len(obj_list))]
    total_obs = len(avg_list)
    warm_starting_point =  WARMSTART_NUM#  int(total_obs*0.01) #100 #
    plt.plot(range(warm_starting_point, len(avg_list)), avg_list[warm_starting_point:], label = alias)
    plt.xlabel('# of interactions', fontsize=14)
    plt.ylabel('average ' + alias, fontsize=14)
    plt.yscale('log')
    plt.legend()
    if vertical_list:
        for v in vertical_list:
            plt.axvline(x=v)
    online_avg_loss = [(obj_list[i+100] -obj_list[i])/100 for i in range(len(obj_list)-100)]


def online_learning_loop(iter_num, vw_examples, Y, vw_alg, loss_func, dataset_name, \
    method_name = '', exp_alias='', rerun=False, shuffle=False, use_log=False):
    """ Implements the online learning loop.
    Args:
        iter_num (int): The total number of iterations
        vw_examples (list): A list of vw examples
        alg (alg instance): An algorithm instance has the following functions:
            - alg.learn(example)
            - alg.predict(example)
            - alg.get_loss()
            - alg.get_sum_loss()
        dataset_name (str):
        loss_func (str):
        method_name (str):
    Outputs:
        cumulative_loss_list (list): the list of cumulative loss from each iteration.
            It is returned for the convenience of visualization.
    """
    # setup the result logger
    res_file_name = ('-').join( [str(dataset_name), str(exp_alias), str(method_name), str(iter_num), str(shuffle), str(use_log)] ) + '.json'
    # res_file_name =dataset_name + '_' + exp_alias + '_' + method_name + '_' + str(iter_num) + '.json'
    res_dir = './result/result_log/oml_' + dataset_name + '/'
    if not os.path.exists(res_dir): os.makedirs(res_dir)
    result_file_address = res_dir+res_file_name
    print('res_file_name', res_file_name)
    loss_list = []
    if path.exists(result_file_address) and not rerun:
        result_log = ResultLogReader(result_file_address)
        result_log.open()
        print('---result file exists and loading res from:', result_file_address)
        for r in result_log.records():
            loss = r.loss
            loss_list.append(loss)
        print('---finished loading')
        return loss_list
    else:
        print('rerunning exp....')
        result_log = ResultLogWriter(result_file_address, loss_metric=loss_func, \
            method_name=method_name)
        result_log.open()
        loss_list = []
        y_predict_list =[]
        
        for i in range(iter_num):
            start_time = time.time()
            vw_x = vw_examples[i]
            y = get_y_from_vw_example(vw_x)
            # predict step
            y_pred =  vw_alg.predict(vw_x) 
            # learn step
            vw_alg.learn(vw_x)
            # calculate one step loss
            if 'squared' in loss_func:  
                loss = mean_squared_error([y_pred], [y]) 
            elif 'absolute' in loss_func:
                loss = mean_absolute_error([y_pred], [y]) 
            else: 
                loss = mean_squared_error([y_pred], [y]) 
                NotImplementedError
            loss_list.append(loss)
            y_predict_list.append([y_pred, y])
            # logging results
            result_log.append(record_id=i, y_predict=y_pred, y=y, loss=loss,
                time_used=time.time()-start_time, 
                incumbent_config=None,
                champion_config=None)
        result_log.close()
        # return cumulative_loss_list,
        return loss_list

if __name__=='__main__':
    inf_num = np.inf
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_num', metavar='iter_num', type = int, 
        default=2000, help="total iteration number")
    parser.add_argument('-policy_budget', '--policy_budget', metavar='policy_budget', 
    type = int, default=5, help="budget for the policy that can be evaluated")
    parser.add_argument('-min_resource', '--min_resource', metavar='cost_budget', 
    type = float, default=50, help="budget for the computation resources that can be evaluated")
    parser.add_argument('-dataset', '--dataset', metavar='dataset', 
    type = str, default= 'simulation', help="get dataset")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*' , 
        default= [], help="The method list")
    parser.add_argument('-ns_num', '--ns_num', metavar='ns_num', type = int, 
        default=10, help="max name space number")
    parser.add_argument('-rerun', '--force_rerun', action='store_true',
                        help='whether to force rerun.') 
    parser.add_argument('-shuffle', '--shuffle_data', action='store_true',
                        help='whether to force rerun.') 
    parser.add_argument('-log', '--use_log', action='store_true',
                        help='whether to use_log.') 

    args = parser.parse_args()
    task_alias = str(args.dataset) + '_' + str(args.min_resource) 
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    log_file_name = LOG_DIR + task_alias + '.log'
    #Generate data
    from data import get_data
    logging.basicConfig(filename=log_file_name, format='%(asctime)s %(name)s:%(levelname)s:%(message)s',
        filemode="w", level=logging.DEBUG)
    vw_examples, Y = get_data(args.iter_num, data_source = args.dataset, vw_format=True, max_ns_num=args.ns_num, \
        shuffle=args.shuffle_data, use_log=args.use_log)

    if vw_examples is not None:
        namespace_feature_dim = get_ns_feature_dim_from_vw_example(vw_examples) 
        fixed_hp_config = {'alg': 'supervised',   'loss_function': 'squared'}  
        #instantiate several vw learners (as baselines) and an AutoOnlineLearner
        alg_dic = {}
        if 'simulation' in args.dataset: alg_dic['oracleVW'] = pyvw.vw(q=['ab','ac','cd'], \
            **fixed_hp_config)
        alg_dic['naiveVW'] = pyvw.vw(**fixed_hp_config)
        auto_alg_common_args = {
            "min_resource_budget": args.min_resource,
            "concurrent_running_budget":args.policy_budget,
            "namespace_feature_dim": namespace_feature_dim, 
            "fixed_hp_config":fixed_hp_config,
            # 'model_select_policy': 'select:threshold_loss_avg',
            'model_select_policy': 'select:threshold_loss_ucb',
            # 'model_select_policy': 'select:loss_ucb',
            # 'model_select_policy': 'select:,
            # 'model_select_policy': 'chacha',
            # "champion_test_policy" :'loss_avg',
             "champion_test_policy" :'loss_ucb',
            }

        online_doubling_notest = {
            "trial_runner_name": 'SuccessiveDoubling',
            "champion_test_policy" :'notest',
            }
        # fixed = {
        #     "trial_runner_name": 'SuccessiveDoubling',
        #     "min_resource_budget": np.inf,
        #     "champion_test_policy" :'notest',
        #     }
        fixed = {
            "trial_runner_name": 'Chambent',
            "min_resource_budget": np.inf,
            "champion_test_policy" :'notest',
            }
        fixed50 = {
            "trial_runner_name": 'SuccessiveDoubling',
            "min_resource_budget": np.inf,
            "champion_test_policy" :'notest',
            "concurrent_running_budget":50,
            }
        autocross = {"trial_runner_name": 'autocross',}
        autocross_plus = {"trial_runner_name": 'autocross+',}
        online_sdsha_args = {"trial_runner_name": 'SuccessiveDoublingsha',}

        online_sd_args_both_keep_all_UCB_inf = {
            "trial_runner_name": 'Chambent',
            'keep_incumbent_running': 1,
            'keep_champion_running': 1,
             "min_resource_budget": np.inf,
            'remove_worse': 1,
            "champion_test_policy" :'loss_ucb',
            'model_select_policy': 'select:threshold_loss_ucb',
            }

        # Chambent = {
        #     "trial_runner_name": 'Chambent-Doubling',
        #     'keep_incumbent_running': 1,
        #     'keep_champion_running': 1,
        #     'remove_worse': 0,
        #     "champion_test_policy" :'loss_ucb',
        #     'model_select_policy': 'select:threshold_loss_avg',
        #     }
        Chambent = {
            "trial_runner_name": 'Chambent-Doubling',
            'keep_incumbent_running': 1,
            'keep_champion_running': 1,
            'remove_worse': 1,
            # "champion_test_policy" :'loss_ucb',
            # 'model_select_policy': 'select:threshold_loss_ucb',
            }

        Chambent_hybrid = {
            "trial_runner_name": 'Chambent-Hybrid',
            'keep_incumbent_running': 1,
            'keep_champion_running': 1,
            'remove_worse': 1,
            # "champion_test_policy" :'loss_ucb',
            # 'model_select_policy': 'select:threshold_loss_ucb',
            }
        Chambent_keep_all_running = {
            "trial_runner_name": 'Chambent-Inf',
            'keep_all_running': 1,
            'remove_worse': 1,
            }
        # baseline_auto_methods = [fixed,  ] #online_doubling_notest,autocross
        baseline_auto_methods = [fixed,] #fixed50
        # auto_alg_args_ist = [ online_sd_args_both, online_sd_args_both_re, online_sd_args_both_keep_all, online_sd_args_both_keep_all_UCB] #online_sd_args_both_re, online_sd_args_both,
        # auto_alg_args_ist = [online_sd_args, online_sd_args_both_re, online_sd_args_both_keep_all_UCB] #online_sd_args_both_re, online_sd_args_both,
        auto_alg_args_ist = [Chambent, Chambent_hybrid, Chambent_keep_all_running]#Chambent_keep_all_running, Chambent_test, Chambent_keep_all_running #Chambent_keep_all_running #online_sd_args_both_keep_all_UCB_inf
        for alg_args in (baseline_auto_methods + auto_alg_args_ist):
            autovw_args = auto_alg_common_args.copy()
            autovw_args.update(alg_args)
            if np.isinf(autovw_args['min_resource_budget']): 
                if 'concurrent_running_budget' in alg_args:
                    alg_alias='fixed-'+str(alg_args['concurrent_running_budget'])+'-VW'
                else: alg_alias='fixed-'+str(args.policy_budget)+'-VW'
            else:
                alg_alias = alg_args['trial_runner_name'] #+'-'+str(args.min_resource)
            alg_dic[alg_alias] = AutoVW(**autovw_args)
        if len(args.method_list)!=0: method_list = args.method_list 
        else: method_list = alg_dic.keys()
        logger.debug('method_list%s', method_list)
        for input_method_name in method_list:
            print(input_method_name, alg_dic.keys())
            final_name = input_method_name
            for alg_m in alg_dic.keys():
                print(input_method_name, alg_m)
                if input_method_name=='Chambent' and input_method_name==alg_m:
                    final_name = alg_m
                if input_method_name !='Chambent' and input_method_name in alg_m: 
                    final_name = alg_m
            print('final_name', final_name)
            if final_name in alg_dic.keys():
                alg_name = final_name
                alg = alg_dic[alg_name]
                iter_num = min(args.iter_num, len(Y)-1)
                time_start = time.time()
                print('----------running', alg_name, '-----------')
                if 'naive' in alg_name or 'oracle' in alg_name:
                    exp_alias=''
                else: exp_alias=str(args.policy_budget)+'_' + str(args.min_resource)
                cumulative_loss_list = online_learning_loop(iter_num, vw_examples, Y, alg, loss_func=fixed_hp_config['loss_function'],\
                    dataset_name=args.dataset, method_name = alg_name, exp_alias=exp_alias, rerun=args.force_rerun,
                    shuffle=args.shuffle_data, use_log=args.use_log)
                logger.critical('%ss running time: %s, total iter num is %s', alg_name, time.time() - time_start, iter_num)
                # generate the plots
                plot_obj(cumulative_loss_list, alias= alg_name)
            else:
                print('alg not exist')

        # save the plots
        alias = 'loss_' + 'ns_' + str(args.ns_num) + '_shuffle_' + str(args.shuffle_data) + '_log_' + str(args.use_log) + \
            args.dataset + '_' + exp_alias + '_' + str(iter_num)
        # alias = 'shuffled_data_loss' + args.dataset + '_' + exp_alias
        fig_name = PLOT_DIR + alias + '.pdf'
        plt.savefig(fig_name)

## command lines to run exp
# conda activate vw
# python tester.py -i 10000 -c 200 >res.txt


# -m naiveVW oracleVW fixed Chambent-Doubling Chambent-Inf

# python tester.py -i 1000 -min_resource 20 -policy_budget 5  -dataset simulation -m Chambent-SuccessiveDoubling Chambent-Inf -rerun
# python tester.py -i 10000 -min_resource 10 -policy_budget 5 -dataset 344  -rerun  -log -m Chambent-SuccessiveDoubling Chambent-Inf