
import numpy as np
import argparse
import itertools
import pandas as pd
from sklearn.metrics import mean_squared_error
from util import squared_error, plot_obj, fm2inter_arg
from vowpalwabbit import pyvw
import pylibvw

from sklearn.preprocessing import PolynomialFeatures
import itertools

def convert_tuple_list(tuple_list):
    converted_tuple_list = []
    for element in tuple_list:
        tuple_ = ele_from_nested_tuple(element)
        converted_tuple_list.append(tuple_)
    return converted_tuple_list

def ele_from_nested_tuple(test_tuple): 
    res = tuple() 
    if type(test_tuple) is int: return (test_tuple,)
    else:
        for ele in test_tuple: 
            if isinstance(ele, tuple): 
                res += ele_from_nested_tuple(ele)
            else:
                res += (ele,)
        # print('res', res, len(res))
        return res 

def cob_list_of_tuple(list1, list2):
    # get combinatorial list of tuples
    new_list = []
    for i in list1:
        for j in list2:
            new_tuple = (i,) + (j,)
            #TODO: is there a more efficient way to do this?
            if new_tuple not in new_list and ((j,) + (i,)) not in new_list:
                new_list.append(new_tuple)
    return new_list

class FM_Set_Generator:
    """This is the class which specifies how to generate new feature maps 
        Important functions
        - feature_map_set_generator(seed_fm): based on the input feature map, 
        it generates all feature maps that have a higher order of interactions
        as the candidate feature set pool
        - rank_feature_map_set(feature_map_set): generate an ordered list of 
        the input feature_map_set
    """

    def __init__(self, ns_list, max_poly_degree=3):
        self.max_poly_degree = max_poly_degree
        self.high_order_combinations = [] #get all high order combinations
        self.ns_list = ns_list
        # self.expand_magnitute = 5
        poly_comb_dic = {}
        poly_comb_dic[1] = ns_list
        for i in range(2, max_poly_degree+1):
            # poly_comb_dic[i] = list(itertools.product(* [poly_comb_dic[i-1], poly_comb_dic[1]]))
            poly_comb_dic[i] = cob_list_of_tuple(poly_comb_dic[i-1], poly_comb_dic[1])
        all_poly_list = []
        for i in range(1, max_poly_degree+1):
            poly_i_list = list(poly_comb_dic[i])
            all_poly_list.extend(poly_i_list)
        # get all possible interactions up to max_poly_degree 
        self.all_poly_list = convert_tuple_list(all_poly_list)
        self.seed_fm = []

    def feature_map_set_generator(self, seed_fm, expand_magnitute=50):
        """
        input: seed feature set
        output: a set of feature set
        """
        #TODO: construct the feature map set generator
        feature_map_set = set()
        if seed_fm is None:
            fm_first = []
            for j in self.all_poly_list:
                if len(j)==1: fm_first.append(j)
            seed_fm =  tuple(fm_first)
        added = []
        for i in range(expand_magnitute):
            new_fm = seed_fm
            for j in self.all_poly_list:
                if j in seed_fm or j in added:
                    continue
                new_fm += (j,)
                added.append(j)
                break
            # if new_fm is still None it means all poly has been tried
            if new_fm == seed_fm: break
            feature_map_set.add(new_fm)
        feature_map_set.add(seed_fm)
        return feature_map_set, seed_fm

    #TODO: reconstruct
    def rank_feature_map_set(self, feature_map_set, policy_budget=None, *kw):
        ranked_feature_maps=[]
        # ranked_feature_maps.append(seed_fm)
        #TODO: rank feature maps
        feature_map_list=list(feature_map_set)
        for i in range(len(feature_map_list)):
            ranked_feature_maps.append(feature_map_list[i])
        if policy_budget and policy_budget<len(ranked_feature_maps):
            return ranked_feature_maps[:policy_budget]
        else: return ranked_feature_maps


class Environment:
##TODO: add description about the environment
##TODO: currently building the environment from simulation. 
##May need to consider constructing the environment from supervised datasets

    def __init__(self, iter_num, parameter=None):
        self.Y = None 
        # self.raw_ns = ['a', 'b', 'c', 'd', 'e']
        # #key is namespace id, and value is the dim of the namespace
        # self.raw_ns_dic = {'a':3, 'b':3, 'c':3, 'd':3, 'e':3}
        # self.ground_truth_ns = ['a', 'b', 'c', 'd', 'e' 'ab', 'ac', 'cd']

        self.raw_ns = ['a', 'b',]
        #key is namespace id, and value is the dim of the namespace
        self.raw_ns_dic = {'a':3, 'b':3,}
        self.ground_truth_ns = ['a', 'b', 'ab',]
        self.generate_raw_X(iter_num)
        self.generate_parameter()
        self.generate_reward(self.vw_x_dic_list)
        self.construct_vw_example()

    def generate_raw_X(self, iter_num):
        self.vw_x_dic_list = []
        for i in range(iter_num):
            vw_x_dic = {}
            count = 0
            for ns, ns_dim in self.raw_ns_dic.items():
                feature = np.array([np.random.normal(0, 1.0, 1)[0] for i in range(ns_dim)])  
                feature = feature/np.linalg.norm(feature)
                vw_x_dic[ns] = feature
                count +=1
            self.vw_x_dic_list.append(vw_x_dic)
        
    def generate_parameter(self):
        #generate reward paramter dictionary
        vw_parameter_dic = {}
        count =0
        for ns_int in self.ground_truth_ns:
            ns_dim = 1
            for ns in ns_int:
                ns_dim *= self.raw_ns_dic[ns]
            parameter = np.array([np.random.normal(0, 0.5, 1)[0] for i in range(ns_dim)] )  
            vw_parameter_dic[ns_int] = parameter

            count +=1
        self.vw_parameter_dic = vw_parameter_dic

    def generate_reward(self, vw_x_list):
        # input is a dictionary of feature in vw format
        y_list = []
        for vw_x_dic in vw_x_list:
            reward_fs = 0
            for ns, fs in vw_x_dic.items():
                if ns in self.vw_parameter_dic:
                    # print(ns, fs, self.vw_parameter_dic[ns])
                    reward_fs += np.dot(vw_x_dic[ns], self.vw_parameter_dic[ns])
            noise = np.random.normal(0, 0.001, 1)[0]
            # noise  = 0
            r = reward_fs + noise
            log_r = 1/(1 + np.exp(-r))
            if log_r >0.5:
                label = 1
            else: label = 0
            y_list.append(r)
        self.Y = y_list
        # print(self.Y)
        # import matplotlib.pyplot as plt
        # plot_obj(self.Y)
        # fig_name = './results/' + 'y.pdf'
        # plt.savefig(fig_name)

    def construct_vw_example(self):
        # construct a list of vw example
        self.vw_examples = []
        for i, x_dic in enumerate(self.vw_x_dic_list):
            raw_vw_example = str(self.Y[i]) + ' '
            for ns, ns_x in x_dic.items():
                raw_vw_example = raw_vw_example + '|' + str(ns) + ' ' + ' '.join([str(s) for s in ns_x]) + ' '
            # pyvw_example = pyvw.example(raw_vw_example)
            self.vw_examples.append(raw_vw_example)
        # self.vw_examples = [
        # "0 | price:.23 sqft:.25 age:.05 2006",
        # "1 | price:.18 sqft:.15 age:.35 1976",
        # "0 | price:.53 sqft:.32 age:.87 1924",]

#TODO: custromized wrapper around vw learner
class Learner(pyvw.vw):
    """ This is a custromized wrapper around vw learner.
        Comparing to pyvw.vw, it has additional information including
        - id
        - cb: the confidence bound of the learner average loss
        - loss_ub: the upper bound of the loss
        - loss_lb: the lower bound of the loss
        - cost: accumulated cost consumed
        - feature_map: feature map used, which is a list of namespaces and their interactions
    """
    C = 0.05 #constant parameter used when constructing cb
    def __init__(self, feature_map, feature_map_id, interactions, **basic_vw_args):
        self.feature_map=feature_map
        self.id = feature_map_id
        self.loss_avg=0.0
        self.cb=0.0
        self.loss_ub= self.loss_avg + self.cb #upper bound of the loss
        self.loss_lb= self.loss_avg - self.cb #lower bound of the loss
        self.use_count=0.0
        self.fm_complexity=len(self.feature_map)
        self.C = Learner.C
        self.cost = 0
        self.loss_lower = 0.0
        self.loss_upper = 1.0
        super().__init__(q=interactions, **basic_vw_args)
    
    def learn(self, x, y, loss):
        self.loss_avg = (self.loss_avg*self.use_count + loss
                        )/(self.use_count+1.0)
        # print(self.loss_avg, self.loss_func(y,y_pred) )
        self.use_count+=1.0
        self.cb=self.C*np.sqrt(self.fm_complexity)/np.sqrt(self.use_count)
        self.loss_ub= self.loss_avg + self.cb #upper bound of the loss
        self.loss_lb= max(self.loss_avg - self.cb, self.loss_lower)  #lower bound of the loss
        #TODO: maybe add epoch to avoid updating every singe observation
        # currently adding 1 unit of cost each update
        self.cost +=1 
        super().learn(x)

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

    def __init__(self, fm_generator, cost_budget = None, policy_budget = None, **basic_vw_args):
        self.fm_generator = fm_generator
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
        self._update_learners(x,y)
    
    def _initialize_learner(self, fm, fm_id, interaction):
        return Learner(fm, fm_id, interaction, **self.basic_vw_args)

    def _generate_new_fm_condition_satisfied(self, best_learner, seed_learner):
        # print(best_learner.loss_ub, best_learner.loss_lb, seed_learner.loss_lb)
        return best_learner.loss_ub + 0.1*(seed_learner.loss_ub - seed_learner.loss_lb
                ) < seed_learner.loss_lb
    
    def _update_all_learners(self, x, y):
        for key, learner in self.learner_dic.items():
            y_pred = learner.predict(x)
            loss = self.loss_func(y, y_pred)
            learner.learn(x, y, loss)

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

    def _update_learners(self, x, y):
        # y_pred_final, best_learner_id = self.predict(x, self.feature_map_ordered_list, self.ns_list)   
        self.best_learner = self.learner_dic[self.best_learner_id]
        self.seed_learner = self.learner_dic[self.seed_fm_id]
        #update learners
        self._update_all_learners(x, y)
        self.iter_count +=1
        #examine whether to call the feature set generator
        if self._generate_new_fm_condition_satisfied(self.best_learner, self.seed_learner): # or \
            # self.iter_count > n_lower:
            #or self._computation_exhausted():
            self.seed_fm = best_learner.feature_map
            self.GENERATE_new_fm_set = True
            print('GENERATE_new_fm_set', self.i)
            self.iter_count = 0
        else: pass
        #if there is computational budget
        if self.cost_budget and len(self.learner_dic)!=0:
            self._learner_cleaning()

def online_learning_loop(iter_num, env, alg_dic):
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
    X = env.vw_x_dic_list
    Y = env.Y
    vw_example = env.vw_examples
    cumulative_loss_list, cumulative_loss = {}, {}
    for m in alg_dic: cumulative_loss_list[m], cumulative_loss[m] = [], 0
    iter_count = 0
    for i in range(iter_num):
        x = vw_example[i]
        y = Y[i]
        #get label and accumuate loss
        for m in alg_dic:
            # x = pyvw.example(alg_dic[m], x)
            vw_x = pyvw.example(alg_dic[m], x)
            y_pred= alg_dic[m].predict(vw_x)  
            # print('learner', m, y_zpred, y, loss)
            if 'auto' not in m: 
                alg_dic[m].learn(vw_x)
                loss = vw_x.get_loss()
                cumulative_loss[m] = alg_dic[m].get_sum_loss()
                cumulative_loss_list[m].append(cumulative_loss[m])
                print( 'sum loss', alg_dic[m].get_sum_loss(), cumulative_loss[m])
                print('loss', y_pred, vw_x.get_feature_number(), vw_x.get_tag(), vw_x.get_loss())
                alg_dic[m].finish_example(vw_x)
            else: 
                alg_dic[m].learn(vw_x, y)

    #plot cumulative losses
    for b in alg_dic:
        if 'auto' in b:
            vertical_list = alg_dic[b].call_generator_index 
        else: vertical_list = [] 
        plot_obj(cumulative_loss_list[b], alias= b )
    import matplotlib.pyplot as plt
    alias = 'loss_all'
    fig_name = './results/' + alias + '.pdf'
    plt.savefig(fig_name)


if __name__=='__main__':
    inf_num = np.inf
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter_num', metavar='iter_num', type = int, 
        default=100, help="total iteration number")
    parser.add_argument('-b', '--policy_budget', metavar='policy_budget', 
    type = float, default= None, help="budget for the policy that can be evaluated")
    parser.add_argument('-c', '--cost_budget', metavar='cost_budget', 
    type = float, default= 100, help="budget for the computation resources that can be evaluated")
    args = parser.parse_args()

    #set up the learning environment, which can generate the learning examples
    #currently from simulation (can also constructed from dataset)
    #TODO: need to construct the environment from a supervised dataset
    #TODO: --cbify
    env = Environment(args.iter_num)

    #specify a feature map generator
    fm_generator = FM_Set_Generator(env.raw_ns)
    
    basic_vw_args = {'l2': 0.1, 'loss_function': 'squared'}
    #instantiate several vw learners (as baselines) and an AutoOnlineLearner
    alg_dic = {}
    alg_dic['naive learner'] = pyvw.vw(**basic_vw_args)
    alg_dic['oracle learner'] = pyvw.vw(q='ab', **basic_vw_args)
    # alg_dic['auto learner'] = AutoOnlineLearner(fm_generator = fm_generator, 
    #     cost_budget = args.cost_budget, policy_budget = args.policy_budget, **basic_vw_args)

    online_learning_loop(args.iter_num, env, alg_dic)

## command lines to run exp
# conda activate vw
# python tester.py -i 30


