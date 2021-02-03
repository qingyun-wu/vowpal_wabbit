import numpy as np
import argparse
from learner import AutoVW
from vowpalwabbit import pyvw
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import LOG_DIR, PLOT_DIR, MAIN_RES_LOG_DIR, WARMSTART_NUM, MIN_RES_CONST, \
    ICML_DATASET_10NS, AB_RES_LOG_DIR, ORACLE_RANDOM_SEED
import logging
import time 
import os
from os import path
import matplotlib.pyplot as plt
from result_log import ResultLogReader,ResultLogWriter
from util import get_y_from_vw_example, get_ns_feature_dim_from_vw_example
from config import FONT_size_label
SEED_LIST = [None, 9999, 4567, 8666, 2468]
FINAL_METHOD_alias = {
    'fixed-50-VW': 'ExhaustInit',
    'fixed-5-VW': 'RandomInit',
    'naiveVW': 'Naive',
    'ChaCha-Org': 'ChaCha',
    'Chambent-Hybrid': 'ChaCha',
    'ChaCha-nochampion-top0': 'ChaCha-w/o-Champion',
    'ChaCha-ucb-top0': 'ChaCha-AggressiveScheduling',
}

FINAL_METHOD_color = {
    'fixed-50-VW': 'tab:red',
    'fixed-5-VW': 'tab:blue',
    'naiveVW': 'tab:brown',
    'ChaCha-Org': 'tab:green',
    'Chambent-Hybrid': 'black',
    'ChaCha-nochampion-top0': 'tab:orange',
    'ChaCha-ucb-top0': 'tab:purple',
}

MAIN_NAME_LIST = ['fixed-5-VW','Chambent-Hybrid']
ABL_NAME_LIST = ['Chambent-Hybrid', 'ChaCha-nochampion-top0','ChaCha-ucb-top0']
MAIN_NAME_LIST_2 = ['fixed-5-VW','ChaCha-Org']
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
    print(obj_list[:5])
    avg_list = [sum(obj_list[:i])/i for i in range(1, len(obj_list))]
    total_obs = len(avg_list)
    warm_starting_point =  WARMSTART_NUM#  int(total_obs*0.01) #100 #
    plt.plot(range(warm_starting_point, len(avg_list)), avg_list[warm_starting_point:], label = alias)
    plt.xlabel('# of samples', fontsize=FONT_size_label)
    plt.ylabel('Progressive validation ' + alias, fontsize=FONT_size_label)
    plt.yscale('log')
    plt.legend()
    if vertical_list:
        for v in vertical_list:
            plt.axvline(x=v)
    # online_avg_loss = [(obj_list[i+100] -obj_list[i])/100 for i in range(len(obj_list)-100)]


def plot_progressive_loss(loss_dic, fig_name):
    """
        loss_dic: key: alg_name, value: list of loss list
    """
    print('genearting loss figures')
    progressive_loss_dic = {}
    prog_loss_mean_dic = {}
    prog_loss_std_dic = {}
    # converting loss to average loss
    print('converting loss to average loss')
    for k, v in loss_dic.items():
        progressive_loss_dic[k] = []
        res = 100
        for loss_list in v:
            avg_list = [sum(loss_list[:i*100])/(i*100) for i in range(1, int(len(loss_list)/res)-1)]
            progressive_loss_dic[k].append(avg_list)
        print(len(progressive_loss_dic[k]), 'nimber')
        progressive_loss_dic[k] = np.array(progressive_loss_dic[k])
        prog_loss_mean_dic[k] = np.mean(progressive_loss_dic[k], axis=0 )
        prog_loss_std_dic[k] = np.std(progressive_loss_dic[k], axis=0 )
    print('plotting')
    fig, ax = plt.subplots()
    for method, alias in FINAL_METHOD_alias.items():
        if method in prog_loss_mean_dic:
            avg_list = prog_loss_mean_dic[method] 
            std_list = prog_loss_std_dic[method]
            print(std_list[0:10])
            warm_starting_point =  WARMSTART_NUM#  int(total_obs*0.01) #100 #
            ax.plot(range(len(avg_list)), avg_list, color=FINAL_METHOD_color[method],label = alias)
            ax.fill_between(range(len(avg_list)), avg_list - std_list, avg_list + std_list,color=FINAL_METHOD_color[method], alpha=0.2)
    ax.set_xlabel('# of samples', fontsize=FONT_size_label)
    ax.set_ylabel('Progressive validation loss', fontsize=FONT_size_label)
    ticks = ax.get_xticks()*int(res)
    ax.set_xticklabels(ticks)
    ax.set_yscale('log')
    plt.legend()

    plt.savefig(fig_name)

def get_normalized_score(res_dic, m_name, labels, name_0='naive', name_1='fixed-50'):
    
    dataset_normalized_scores = {}
    label_scores = []
    # final_label = []
    for dataset in labels:
        res = res_dic[dataset]
        # generate loss
        loss_0, loss_1 = None, None
        for alg, value in res.items():
            if name_0 in alg: loss_0 = value
            if name_1 in alg: loss_1 = value
        normalized_loss = {}
        if loss_0 is not None and loss_1 is not None and loss_0 != loss_1 and m_name in res:
            score = (loss_0-res[m_name])/float( loss_0-loss_1)
            label_scores.append( float('{:.2f}'.format(float(score))))
            for alg in res:
                normalized_loss[alg] = (loss_0-res[alg])/float( loss_0-loss_1)
                normalized_loss[alg] = float('{:.2f}'.format(float(normalized_loss[alg])))
        else: label_scores.append(0) #TODO: check 
    return label_scores



def plot_normalized_scores(res_dic, alias='', chacha_name='ChaCha-nochampion-top0', error_bar=False, name_0='naive', name_1='fixed-50'):
    m1_name = 'fixed-5-VW'
    m2_name = chacha_name
    m1_alias = 'RandomInit'
    m2_alias = 'ChaCha-'
    all_names = ['ChaCha-CB','ChaCha-Org','ChaCha','ChaCha-ucb-top0','ChaCha-nochampion', 'fixed-5-VW', 'Chambent-Hybrid', 'Chambent-Van-ucb-tophalf']
    # print 
    print('resdic', res_dic)
    if not error_bar: 
        result = [res_dic]
    else:
        assert type(res_dic)==list 
        result = res_dic
    # print(error_bar, type(result), type(result[0]))
    datasets = list(result[0].keys())
    labels = datasets 
    all_res1=[]
    all_res2=[]
    print(datasets)
    for name in all_names:
        for res_dic in result:
            normalized_res = get_normalized_score(res_dic, name, datasets)
            print(name, normalized_res)
            if name == m1_name: 
                res_1 =normalized_res
                all_res1.append(res_1)
            if name == m2_name: 
                res_2 =normalized_res
                all_res2.append(res_2)

    all_res1=np.array(all_res1)
    all_res2=np.array(all_res2)
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    res_1 = np.mean(all_res1, axis=0)
    res_2 = np.mean(all_res2, axis=0)
    std_1 = np.std(all_res1, axis=0)
    std_2 = np.std(all_res2, axis=0)
   
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, res_1, width, yerr=std_1, label=m1_alias,
          ecolor='grey')
    rects2 = ax.bar(x + width/2, res_2, width, yerr=std_2, label=m2_alias, ecolor='grey')
    
    methods_to_show =['fixed-5-VW', 'Chambent-Hybrid', ]
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized score', fontsize=FONT_size_label)
    # ax.set_title('Normalzied Scores (naive=0, ExhaustInit=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=50)
    ax.set_ylim(-0.5,2.0)
    ax.legend(loc='upper left')
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)
    # fig.tight_layout()
    # plt.axhline(y=1.0, color='r', linestyle='--')
    fig_name = PLOT_DIR  + 'bar_plot_' + alias+ '.pdf'
    plt.savefig(fig_name)
    plt.close()


def normalized_scores_barplot(res_dic, alias='', method_list_to_show = [], error_bar=False, name_0='naive', name_1='fixed-50'):
    all_names = ['ChaCha-CB','ChaCha-Org','ChaCha','ChaCha-ucb-top0','ChaCha-nochampion-top0', 'fixed-5-VW', 'Chambent-Hybrid', 'Chambent-Van-ucb-tophalf']
    # print 
    print('resdic', res_dic)
    if not error_bar: 
        result = [res_dic]
    else:
        assert type(res_dic)==list 
        result = res_dic
    # print(error_bar, type(result), type(result[0]))
    datasets = list(result[0].keys())
    labels = datasets 
    all_res1=[]
    all_res2=[]
    print(datasets)
    fig, ax = plt.subplots()
    if len(method_list_to_show) <3: barWidth = 0.25
    else: barWidth = 0.15

    r_position = []
    r0 = np.arange(len(labels))
    r_position.append(r0)
    for i,m_name in enumerate(all_names):
        all_res = []
        for res_dic in result:
            normalized_score = get_normalized_score(res_dic, m_name, datasets)
            all_res.append(normalized_score)
        m_mean = np.mean(all_res, axis=0)
        m_std = np.std(all_res, axis=0)
        m_alias = FINAL_METHOD_alias[m_name] if m_name in FINAL_METHOD_alias else 'None'
        m_color = FINAL_METHOD_color[m_name] if m_name in FINAL_METHOD_alias else 'red'
        if m_name in method_list_to_show:
            r_new = [x + barWidth for x in r_position[-1]]
            r_position.append(r_new) 
            plt.bar(r_position[-1], m_mean, width=barWidth, yerr=m_std, label=m_alias, ecolor='grey' )
    # x = np.arange(len(labels))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized score', fontsize=FONT_size_label)
    # ax.set_title('Normalzied Scores (naive=0, ExhaustInit=1)')
    ax.set_xticks(r0)
    ax.set_xticklabels(labels,rotation=50)
    ax.set_ylim(-0.5,2.0)
    ax.legend(loc='upper left')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)
    # fig.tight_layout()
    # plt.axhline(y=1.0, color='r', linestyle='--')
    fig_name = PLOT_DIR  + 'bar_plot_' + alias+ '.pdf'
    plt.savefig(fig_name)
    plt.close()

def online_learning_loop(iter_num, vw_examples, Y, vw_alg, loss_func, \
    method_name = '', result_file_address='./res.json'):
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
    Outputs:
        cumulative_loss_list (list): the list of cumulative loss from each iteration.
            It is returned for the convenience of visualization.
    """
    print('rerunning exp....', len(vw_examples), iter_num)
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
        if 'squared' in loss_func: loss = mean_squared_error([y_pred], [y]) 
        elif 'absolute' in loss_func: loss = mean_absolute_error([y_pred], [y]) 
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
    return loss_list

if __name__=='__main__':
    inf_num = np.inf
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_num', metavar='iter_num', type = int, 
        default=2000, help="total iteration number")
    parser.add_argument('-policy_budget', '--policy_budget', metavar='policy_budget', 
    type = int, default=5, help="budget for the policy that can be evaluated")
    parser.add_argument('-d', '--dataset_list', metavar='dataset_list', nargs='*' , 
    type = str, default= ['simulation' ] , help="get dataset")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*' , 
        default= [], help="The method list")
    parser.add_argument('-ns_num', '--ns_num', metavar='ns_num', type = int, 
        default=10, help="max name space number")
    parser.add_argument('-seed', '--config_oracle_random_seed', metavar='config_oracle_random_seed', type = int, 
        default=None, help="set config_oracle_random_seed")
    parser.add_argument('-rerun', '--force_rerun', action='store_true',
                        help='whether to force rerun.') 
    parser.add_argument('-no_rerun', '--force_no_rerun', action='store_true',
                        help='whether to force no rerun.')
    parser.add_argument('-bar_plot_only', '--bar_plot_only', action='store_true',
                        help='whether to only generate plot.') 
    parser.add_argument('-shuffle', '--shuffle_data', action='store_true',
                        help='whether to force rerun.') 
    parser.add_argument('-ablation', '--ablation', action='store_true',
                        help='whether to ablation.') 
    parser.add_argument('-log', '--use_log', action='store_true',
                        help='whether to use_log.') 
    parser.add_argument('-barplot', '--barplot', action='store_true',
                        help='whether to use_log.') 

    args = parser.parse_args()
    all_dataset_result_list = [] # key: seed, value: result dictionary(key: dataset id; value: different method's name)
    # RES_LOG_DIR = './result/result_log_final/'
    method_cum_loss = {}
    print('args.dataset_list',args.dataset_list)
    if 'all' in args.dataset_list:
        dataset_list = ICML_DATASET_10NS
    else: dataset_list = args.dataset_list
    print('datasets', dataset_list)
    RES_LOG_DIR = AB_RES_LOG_DIR if args.ablation else MAIN_RES_LOG_DIR
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)
    if not os.path.exists(RES_LOG_DIR): os.makedirs(RES_LOG_DIR)
    for seed in SEED_LIST:
        all_dataset_result = {}
        for dataset in dataset_list:
            dataset = str(dataset)
            task_alias = str(dataset) 
            log_file_name = LOG_DIR + task_alias + '.log'

            if args.bar_plot_only: assert args.force_rerun == False
            ####get data 
            if args.bar_plot_only: 
                vw_examples, Y, namespace_feature_dim = None, None, {'d':1}
            else:
                ###Get data if rerun exp
                from data import get_data
                logging.basicConfig(filename=log_file_name, format='%(asctime)s %(name)s:%(levelname)s:%(message)s',
                    filemode="w", level=logging.DEBUG)
                vw_examples, Y = get_data(args.iter_num, data_source = dataset, vw_format=True, max_ns_num=args.ns_num, \
                    shuffle=args.shuffle_data, use_log=args.use_log)
                namespace_feature_dim = get_ns_feature_dim_from_vw_example(vw_examples) 
            feature_dim = sum([d for d in namespace_feature_dim.values()])
            # setup alg configs
            fixed_hp_config = {'alg': 'supervised', 'loss_function': 'squared'}  
            #instantiate several vw learners (as baselines) and an AutoOnlineLearner
            alg_dic = {}
            if 'simulation' in dataset: alg_dic['oracleVW'] = pyvw.vw(q=['ab','ac','cd'], \
                **fixed_hp_config)
            alg_dic['naiveVW'] = pyvw.vw(**fixed_hp_config)

            ## setup configs for other autoVW methods
            auto_alg_common_args = {
                "min_resource_budget": feature_dim*MIN_RES_CONST, # args.min_resource,
                "concurrent_running_budget":args.policy_budget,
                "namespace_feature_dim": namespace_feature_dim if namespace_feature_dim else None, 
                "fixed_hp_config":fixed_hp_config,
                # 'model_select_policy': 'select:threshold_loss_avg',
                'model_select_policy': 'select:threshold_loss_ucb',
                # 'model_select_policy': 'select:loss_ucb',
                # "champion_test_policy" :'loss_avg',
                "champion_test_policy" :'loss_ucb',
                'config_oracle_random_seed': args.config_oracle_random_seed,
                }

            fixed_b_vw = {
                "trial_runner_name": 'Chambent',
                "min_resource_budget": np.inf,
                "champion_test_policy" :'notest',
                }
            fixed_b_vw_50 = {
                "trial_runner_name": 'Chambent',
                "min_resource_budget": np.inf,
                "champion_test_policy" :'notest',
                "concurrent_running_budget":50,
                }
            autocross = {"trial_runner_name": 'autocross',}
            autocross_plus = {"trial_runner_name": 'autocross+',}
            online_sdsha_args = {"trial_runner_name": 'SuccessiveDoublingsha',}
            Chambent_Doubling = {
                "trial_runner_name": 'Chambent-Doubling',
                'keep_incumbent_running': 1,
                'keep_champion_running': 1,
                'remove_worse': 1,
                }

            Chambent_hybrid = {
                "trial_runner_name": 'Chambent-Hybrid',
                'remove_worse': 1,
                }

            ChaCha = {
                "trial_runner_name": 'ChaCha-Org',
                'remove_worse': 1,
                }
    
            ChaCha_keep_0 = {
                "trial_runner_name": 'ChaCha-ucb-top0',
                'remove_worse': 1,
                }
            ChaCha_no_champion = {
                "trial_runner_name": 'ChaCha-nochampion-top0',
                'remove_worse': 1,
                }
            ChaCha_CB = {
                "trial_runner_name": 'ChaCha-CB',
                'remove_worse': 1,
                }

            Chambent_vanilla = {
                "trial_runner_name": 'Chambent-Vanilla',
                'remove_worse': 1,
                }

            Chambent_vanilla_champion = {
                "trial_runner_name": 'Chambent-champion-Vanilla',
                'remove_worse': 1,
                }

            Chambent_van_top1 = {
                "trial_runner_name": 'Chambent-Van-ucb-top1',
                'remove_worse': 1,
                }

            Chambent_van_tophalf = {
                "trial_runner_name": 'Chambent-Van-ucb-tophalf',
                'remove_worse': 1,
                }


            Chambent_van_top1_lcb = {
                "trial_runner_name": 'Chambent-Van-lcb-top1',
                'remove_worse': 1,
                }

            Chambent_van_tophalf_lcb = {
                "trial_runner_name": 'Chambent-Van-lcb-tophalf',
                'remove_worse': 1,
                }


            Chambent_van_tophalf_avg = {
                "trial_runner_name": 'Chambent-Van-avg-tophalf',
                'remove_worse': 1,
                }
                
            Chambent_van_tophalf_avg_champion = {
                "trial_runner_name": 'Chambent-Van-avg-champion-tophalf',
                'remove_worse': 1,
                }

            Chambent_van_tophalf_champion = {
                "trial_runner_name": 'Chambent-Van-ucb-champion-tophalf',
                'remove_worse': 1,
                }

            Chambent_van_tophalf_champion_inc = {
                "trial_runner_name": 'Chambent-Van-ucb-championINC-tophalf',
                'model_select_policy': 'select:threshold_ucb_champion',
                'remove_worse': 1,
                }

            # Chambent_van_tophalf_avg_champion_incumbent = {
            #     "trial_runner_name": 'Chambent-Van-avg-champion-tophalf',
            #     'model_select_policy': 'select:threshold_champion',
            #     'remove_worse': 1,
            #     }

            Chambent_test = {
                "trial_runner_name": 'Chambent-Test',
                'remove_worse': 1,
                }
            Chambent_noreset = {
                "trial_runner_name": 'Chambent-Noreset',
                'remove_worse': 1,
                }
            Chambent_pause = {
                "trial_runner_name": 'Chambent-Pause',
                'remove_worse': 1,
                }
            Chambent_Inf = {
                "trial_runner_name": 'Chambent-Inf',
                'keep_all_running': 1,
                'remove_worse': 1,
                }
            baseline_auto_methods = [fixed_b_vw, fixed_b_vw_50] #autocross
            auto_alg_args_ist = [ChaCha,ChaCha_CB, ChaCha_keep_0, ChaCha_no_champion,  Chambent_test, Chambent_hybrid, Chambent_vanilla, Chambent_van_top1, Chambent_van_tophalf,
                Chambent_van_top1_lcb, Chambent_van_tophalf_lcb, Chambent_van_tophalf_champion, Chambent_van_tophalf_champion_inc] # Chambent_test [Chambent_Doubling, Chambent_hybrid, Chambent_Inf] Chambent_test
            # auto_alg_args_ist = [Chambent_vanilla, Chambent_van_top1, Chambent_van_tophalf] Chambent_vanilla, Chambent_van_top1, Chambent_van_tophalf
            for alg_args in (baseline_auto_methods + auto_alg_args_ist):
                autovw_args = auto_alg_common_args.copy()
                autovw_args.update(alg_args)
                if np.isinf(autovw_args['min_resource_budget']): 
                    if 'concurrent_running_budget' in alg_args:
                        alg_alias='fixed-'+str(alg_args['concurrent_running_budget'])+'-VW'
                    else: alg_alias='fixed-'+str(args.policy_budget)+'-VW'
                else:
                    alg_alias = alg_args['trial_runner_name'] #+'-'+str(args.min_resource)
                    logger.debug('exp autovw_args %s %s', autovw_args, alg_args)
                alg_dic[alg_alias] = AutoVW(**autovw_args)
            if len(args.method_list)!=0: method_list = args.method_list 
            else: method_list = alg_dic.keys()
            logger.debug('method_list%s', method_list)
            
            method_results = {} #key: method name, value: result 
            # method_cum_loss = {} #key: method name, value: result 
            # convert method names from input to the names in alg_dic
            # for seed in SEED_LIST:
            for input_method_name in method_list:
                print(input_method_name, alg_dic.keys())
                final_name = input_method_name
                for alg_m in alg_dic.keys():
                    print(input_method_name, alg_m)
                    if input_method_name in alg_m: final_name = alg_m
                print('final algname', final_name)
                
                if final_name in alg_dic.keys():
                    alg_name = final_name
                    alg = alg_dic[alg_name]
                    #TODO: historical problem: iter_num = min(args.iter_num, len(Y)-1) 
                    # iter_num = args.iter_num
                    iter_num = min(args.iter_num, len(Y)-1) 
                    time_start = time.time()
                    print('----------running', alg_name, '-----------')
                    
                    # for seed in SEED_LIST:
                    if 'naive' in alg_name or 'oracle' in alg_name or 'fixed' in alg_name: exp_alias=''
                    else: exp_alias=str(args.policy_budget)

                    if 'naive' not in alg_name and seed is not None:
                        exp_alias = exp_alias+ '_seed_'+str(seed)

                    ### get result file name
                    if seed is not None:
                        res_file_name = ('-').join( [str(dataset), 'ns'+str(args.ns_num), str(exp_alias), 
                        str(alg_name), str(iter_num),str(args.shuffle_data), str(args.use_log) , \
                        'seed'+str(seed)]) + '.json'
                    else:
                        res_file_name = ('-').join( [str(dataset), 'ns'+str(args.ns_num), str(exp_alias), \
                            str(alg_name), str(iter_num), str(args.shuffle_data), str(args.use_log)] ) + '.json'
                    res_dir = RES_LOG_DIR + 'oml_' + dataset + '/'
                    if not os.path.exists(res_dir): os.makedirs(res_dir)
                    result_file_address = res_dir+res_file_name
                    ### load result from file
                    if path.exists(result_file_address) and not args.force_rerun:
                        cumulative_loss_list = []
                        # result_log = ResultLogReader(result_file_address)
                        # result_log.open()
                        # print('---result file exists and loading res from:', result_file_address)
                        # for r in result_log.records():
                        #     cumulative_loss_list.append(r.loss)
                        # print('---finished loading')
                        print('---result file exists and loading res from:', result_file_address)
                        import json
                        with open(result_file_address) as f:
                            for line in f:
                                cumulative_loss_list.append(json.loads(line)['loss'])
                        print('---finished loading')
                    # generate the plots
                    ##TODO: add more visualization 
                    if cumulative_loss_list:
                        # if not args.bar_plot_only: plot_obj(cumulative_loss_list, alias= alg_name)
                        # plot_obj(cumulative_loss_list, alias= alg_name)
                        method_results[alg_name] = sum(cumulative_loss_list)/len(cumulative_loss_list) 
                        if alg_name not in method_cum_loss:  method_cum_loss[alg_name] = []
                        method_cum_loss[alg_name].append(cumulative_loss_list)
                else:
                    print('alg not exist')
            all_dataset_result[dataset] = method_results
        all_dataset_result_list.append(all_dataset_result)
    
    # plot_progressive_loss(method_cum_loss)
    if method_cum_loss:
        # save the plots
        alias = 'loss_' + 'ns_' + str(args.ns_num) + '_shuffle_' + str(args.shuffle_data) + '_log_' + str(args.use_log) + \
            dataset + '_' + exp_alias + '_' + str(iter_num)
        # alias = 'shuffled_data_loss' + dataset + '_' + exp_alias
        fig_name = PLOT_DIR + alias + '.pdf'
        plot_progressive_loss(method_cum_loss, fig_name)
    # plot_normalized_scores(all_dataset_result_list,'cha-org', 'ChaCha-nochampion-top0', error_bar=True)
    ## main result
    if args.barplot:
        normalized_scores_barplot(all_dataset_result_list,'main', MAIN_NAME_LIST, error_bar=True)
        normalized_scores_barplot(all_dataset_result_list,'ablation', ABL_NAME_LIST, error_bar=True)
        normalized_scores_barplot(all_dataset_result_list,'main_2', MAIN_NAME_LIST_2, error_bar=True)
## command lines to run exp
# conda activate vw
# python tester.py -i 10000 -c 200 >res.txt


# -m naiveVW oracleVW fixed Chambent-Doubling Chambent-Inf

# python tester.py -i 1000 -policy_budget 5  -dataset simulation -m Chambent-SuccessiveDoubling Chambent-Inf -rerun
# python tester.py -i 10000 -policy_budget 5 -dataset 344  -rerun  -log -m Chambent-SuccessiveDoubling Chambent-Inf