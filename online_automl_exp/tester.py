
import numpy as np
import argparse
import itertools
import pandas as pd
from sklearn.metrics import mean_squared_error
from util import squared_error, plot_obj, fm2inter_arg
from sklearn.preprocessing import PolynomialFeatures

from trial import Learner, ConfiguredLearner
from automl.search_space import FSSProblem
from automl.search_policy import LocalSearcher


class OnlineAML:
    """the class for AML
        problem: automl problem
    """

    def __init__(self, problem: FSSProblem, init_config = None,  
        gs: GS = None, ls: LS = None, **kwargs):
        self.problem = problem
        self.kwargs = kwargs
        #TODO: need to think about the resource schedule
        self.resouce_name, self._resource_schedule = \
            self.problem.generate_resource_schedule(self.reduction_factor)
        self.incumbent_config = init_config
        self.gs = gs
        self.ls = ls 
        self.evaluted = []
        if 'search_policy' in self.kwargs:
            self.search_policy = self.kwargs['search_policy']

    def propose_next(self, **kwargs):
        self._process_observation()
        self.search_space = self._get_search_space(self.problem, self.incumbent_config)
        choice = self._sample1_from_space()
        return choice

    def _process_observation(self):
        NotImplemented

    def _sample1_from_space(self, incumbent_config, search_space):
        choice = NotImplemented
        return choice

    def _get_search_space(self, problem, incumbent_config):
        search_space = problem.search_space(incumbent_config, 2)
        return search_space 



class AutoOnlineLearner:
    """The AutoOnlineLearner object is auto online learning object.
    It has the same main API with pyvw.vw:
        - predict(example)
        - learn(example)
    
    Parameters
    ----------
    fm_generator: 
        It can generate a list of feature maps through fm_generator.feature_map_set_generator
        It can rank the generated feature map set through fm_generator.rank_feature_map_set
    cost_budget: budget on the cost
    policy_budget: budget on the number of policies that can be evaluted at each iteration
    **basic_vw_args: the args dic used to build a basic vw learner
    """

    def __init__(self, AML:OnlineAML, cost_budget = None, policy_budget = None,
        learner = ConfiguredLearner, **basic_vw_args):
        self.AML = AML
        #maintain a dictionary of learners. key: fm_id, value: learner
        self.learner_dic = {}
        self.basic_vw_args = basic_vw_args

        self.cost_budget = cost_budget
        self.policy_budget = policy_budget
        self.seed_fm = None
        self.best_fm = None
        self.fm_id_dic = {}
        self.call_generator_index = []
        self.best_learner_id = None 
        self.seed_learner_id = None
        self.seed_fm_id = None
        self.GENERATE_new_fm_set = False
        self.iter_count = 0
        self.i = 0
        self.base_learner = learner
        self.searcher = LocalSearcher()
        
    def predict(self, x):
        """ Predict on the example
        Parameters
        ---------
        x: vw_example
        
        Returns
        -------
        predict label of input 
        """
        self.i+=1
        self._regulate_fm_pool()
        self._best_learner_selection()
        y_pred_final = None
        for fm in self.feature_map_ordered_list:
            fm_id = self.fm_id_dic[fm]
            if fm_id== self.best_learner_id: 
                y_pred =self.learner_dic[fm_id].predict(x)
                y_pred_final = y_pred
                return y_pred_final

    def learn(self, x, y = None):
        """Perform an online update
        Parameters 
        ----------
        x : example/str/list
            examples on which the model gets updated
        y : label of the example 
        #TODO: label can be obtained from x
        """
        self.best_learner = self.learner_dic[self.best_learner_id]
        self.seed_learner = self.learner_dic[self.seed_fm_id]
        #update learners
        self._update_all_learners(x, y)
        self.iter_count +=1
        #examine whether to call the feature set generator
        if self._generate_new_fm_condition_satisfied(self.best_learner, self.seed_learner): # or \
            # self.iter_count > n_lower:
            #or self._computation_exhausted():
            self.seed_fm = self.best_learner.feature_map
            self.GENERATE_new_fm_set = True
            print('GENERATE_new_fm_set', self.i)
            self.iter_count = 0
        # #if there is computational budget
        # if self.cost_budget and len(self.learner_dic)!=0:
        #     self._learner_cleaning()
    
    def _initialize_learner(self, interaction):
        return self.base_learner( q = interaction, **self.basic_vw_args)

    def _generate_new_fm_condition_satisfied(self, best_learner, seed_learner):
        # print(best_learner.loss_ub, best_learner.loss_lb, seed_learner.loss_lb)
        return best_learner.loss_ub + 0.1*(seed_learner.loss_ub - seed_learner.loss_lb
                ) < seed_learner.loss_lb
    
    def _update_all_learners(self, x, y):
        for key, learner in self.learner_dic.items():
            learner.learn(x, y)

    def _rank_feature_map_set():
        pass

    def _learner_cleaning(self):
        # pass
        for fm in self.feature_map_ordered_list:
            fm_id = self.fm_id_dic[fm]
            if self.learner_dic[fm_id].cost > self.cost_budget:
                del self.learner_dic[fm_id]
                self.feature_map_set.remove(fm)
                for f in self.feature_map_set:
                    f_id = self.fm_id_dic[f]
                try:
                    b_id = np.argmin([self.learner_dic[self.fm_id_dic[f]].cost for f in self.feature_map_set])
                    print('-----------clearning up out of the budget learners-----------')
                    print('id', b_id)
                    b = self.learner_dic[b_id].feature_map
                    inter_arg = fm2inter_arg(fm)
                    # vw_model = pyvw.vw(" --quiet " + str(inter_arg))
                    self.learner_dic[b_id] = self._initialize_learner(b, b_id, inter_arg)
                    # self.learner_dic[b_id] = Learner(b, b_id, " --quiet " + str(inter_arg))
                    self.cost_budget = self.cost_budget*2+1
                except:
                    pass

    def _computation_exhausted(self):
        total_comput = 0
        for key, learner in self.learner_dic.items():
            # print('learner', learner)
            if learner.cost < self.cost_budget: 
                return False
        #TODO: should we double the computational budget
        #when we exhaust the budget
        self.cost_budget *=2
        print('------------exhausted------------')
        return True 

    def _best_learner_selection(self):
        loss_ub = {}
        for fm in self.feature_map_ordered_list:
            fm_id = self.fm_id_dic[fm]
            if fm_id not in self.learner_dic:
                inter_arg = fm2inter_arg(fm)
                print('add learner', inter_arg)
                vw_model = pyvw.vw()
                self.learner_dic[fm_id] = self._initialize_learner(fm, fm_id, str(inter_arg))
                # self.learner_dic[fm_id] = Learner(fm, fm_id, self.loss_func,
                #      " --quiet " + str(inter_arg))
            loss_ub[fm_id] = self.learner_dic[fm_id].loss_ub
        import operator
        self.best_learner_id = min(loss_ub.items(), key=operator.itemgetter(1))[0]

    def _regulate_fm_pool(self):
        #construct the initial feature set map
        #we define the seed_fm to be a list of tuples specifying which dimensions 
        #should be used to construct the feature

        #maintain a dictionary of feature map id: key: fm, valu:id of the feature set.
        #feature map which is generated according to when the order of fm
        if len(self.learner_dic)==0 or self.GENERATE_new_fm_set:
            self.GENERATE_new_fm_set = False
            self.call_generator_index.append(self.i)
            self.feature_map_set, init_seed_fm = self.fm_generator.feature_map_set_generator(self.seed_fm)
            #generate id for each feature map. Note that fm id is also learner id
            if self.seed_fm is None:
                self.seed_fm = init_seed_fm
            for fm in self.feature_map_set:
                if fm not in self.fm_id_dic: 
                    self.fm_id_dic[fm] = len(self.fm_id_dic)
                    fm_id = self.fm_id_dic[fm]
                    if fm_id not in self.learner_dic:
                        inter_arg = fm2inter_arg(fm)
                        print('add learner',inter_arg)
                        # vw_model = pyvw.vw(" --quiet " + str(inter_arg))
                        self.learner_dic[fm_id]= self._initialize_learner(fm, fm_id, str(inter_arg))
                        # self.learner_dic[fm_id]= Learner(fm, fm_id, self.loss_func, " --quiet " + str(inter_arg))
                #TODO: need to create learner for all fm_set
            #generate a ranked list of the feature map set (if policy_budget is not None and 
            # is smaller than the total size of the feature map set, it will return the top ranked
            # feature map)
            # print('feature map set', self.feature_map_set)
            self.feature_map_ordered_list = self.fm_generator.rank_feature_map_set(self.feature_map_set, 
                self.policy_budget)
        self.seed_fm_id = self.fm_id_dic[self.seed_fm]


def online_learning_loop(iter_num, X, Y, vw_example, alg):
    """ the online learning loop
    Inputs
    ----------
    iter_num: total number of iterations
    env: interaction environment which provides the learning examples and 
        their corresponding labels
        Useful variables:
        - env.vw_examples: a list of vw_example
        - env.Y: the list of labels from the vw_example
        - env.vw_x_dic_list: the list of namespaces and features from the vw_example
    alg_dic: a dictionary of online learning algorithms. 
        The keys of the dictionary are algorithm alias, and values are learning algorithms
        An algorithm instance has the following functions:
        - method.learn(example)
        - method.predict(example)
    """
    cumulative_loss_list = []
    for i in range(iter_num):
        x,y = vw_example[i], Y[i]
        vw_x = pyvw.example(alg, x)
        # y_pred= alg.predict(vw_x)  
        alg.learn(vw_x)
        loss = vw_x.get_loss()
        cumulative_loss_list.append(alg.get_sum_loss())
        # print( 'sum loss', y_pred, loss, alg.get_sum_loss(),)
        # print('loss', y_pred, vw_x.get_feature_number(), vw_x.get_tag(), vw_x.get_loss())
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
    basic_vw_args = {'l2': 0.1, 'loss_function': 'squared'}
    #instantiate several vw learners (as baselines) and an AutoOnlineLearner
    alg_dic = {}
    alg_dic['oracle'] = Learner(q=['ab', 'ac', 'cd'], **basic_vw_args)
    alg_dic['naive'] = Learner(**basic_vw_args)
    #specify a feature map generator
    # problem = FSSProblem()
    alg_dic['auto'] = AutoOnlineLearner(fm_generator = fm_generator, 
        cost_budget = args.cost_budget, policy_budget = args.policy_budget, learner = ConfiguredLearner, **basic_vw_args)
    
    import matplotlib.pyplot as plt
    for alg_name, alg in alg_dic.items():
        cumulative_loss_list = online_learning_loop(args.iter_num, X, Y, vw_example, alg)
        # if 'auto' in alg_name:
        #     vertical_list = alg.call_generator_index 
        # else: vertical_list = [] 
        plot_obj(cumulative_loss_list, alias= alg_name)
    alias = 'loss_all'
    fig_name = './results/' + alias + '.pdf'
    plt.savefig(fig_name)
    
    

## command lines to run exp
# conda activate vw
# python tester.py -i 30


