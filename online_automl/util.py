
import numpy as np
import matplotlib.pyplot as plt


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

def squared_error(y, y_pred):
    #use l2 norm as the loss function
    #TODO: may need to generalize this loss function
    # loss = np.linalg.norm(y-y_pred, 2)**2
    loss = (y-y_pred)**2
    # print(y, y_pred)
    return loss

import re
import pandas as pd
import string

def to_vw_format(line):
    chars = re.escape(string.punctuation)
    res = f'{int(line.y)} |'
    for idx, value in line.drop(['y']).iteritems():
        feature_name = re.sub(r'(['+chars+']|\s)+', '_', idx)
        res += f' {feature_name}:{value}'
    return res

def fm2inter_arg(fm):
    for l in fm:
        if len(l)>1: 
            inter_arg = '--interactions  '
            for jj in l: inter_arg += str(jj)
        else: inter_arg = ''
    return inter_arg



# def _better_champion_found(self, best_learner, seed_learner):
#         # print(best_learner.loss_ub, best_learner.loss_lb, seed_learner.loss_lb)
#         return best_learner.loss_ub + 0.1*(seed_learner.loss_ub - seed_learner.loss_lb
#                 ) < seed_learner.loss_lb
    
#     def _update_all_learners(self, x, y):
#         for _, learner in self.learner_dic.items():
#             learner.learn(x, y)

#     def _rank_feature_map_set(self,):
#         pass

#     def _learner_cleaning(self):
#         # pass
#         for fm in self.feature_map_ordered_list:
#             fm_id = self.fm_id_dic[fm]
#             if self.learner_dic[fm_id].cost > self.cost_budget:
#                 del self.learner_dic[fm_id]
#                 self.feature_map_set.remove(fm)
#                 for f in self.feature_map_set:
#                     f_id = self.fm_id_dic[f]
#                 try:
#                     b_id = np.argmin([self.learner_dic[self.fm_id_dic[f]].cost for f in self.feature_map_set])
#                     print('-----------clearning up out of the budget learners-----------')
#                     print('id', b_id)
#                     b = self.learner_dic[b_id].feature_map
#                     inter_arg = fm2inter_arg(fm)
#                     # vw_model = pyvw.vw(" --quiet " + str(inter_arg))
#                     self.learner_dic[b_id] = self._initialize_learner(inter_arg)
#                     # self.learner_dic[b_id] = Learner(b, b_id, " --quiet " + str(inter_arg))
#                     self.cost_budget = self.cost_budget*2+1
#                 except:
#                     pass

#     def _computation_exhausted(self):
#         total_comput = 0
#         for key, learner in self.learner_dic.items():
#             # print('learner', learner)
#             if learner.cost < self.cost_budget: 
#                 return False
#         #TODO: should we double the computational budget
#         #when we exhaust the budget
#         self.cost_budget *=2
#         print('------------exhausted------------')
#         return True 

#     def _best_learner_selection(self):
#         loss_ub = {}
#         for fm in self.feature_map_ordered_list:
#             fm_id = self.fm_id_dic[fm]
#             if fm_id not in self.learner_dic:
#                 inter_arg = fm2inter_arg(fm)
#                 print('add learner', inter_arg)
#                 self.learner_dic[fm_id] = self._initialize_learner(str(inter_arg))
#                 # self.learner_dic[fm_id] = Learner(fm, fm_id, self.loss_func,
#                 #      " --quiet " + str(inter_arg))
#             loss_ub[fm_id] = self.learner_dic[fm_id].loss_ub
#         import operator
#         self.best_learner_id = min(loss_ub.items(), key=operator.itemgetter(1))[0]

#     def _regulate_fm_pool(self):
#         #construct the initial feature set map
#         #we define the seed_fm to be a list of tuples specifying which dimensions 
#         #should be used to construct the feature

#         #maintain a dictionary of feature map id: key: fm, valu:id of the feature set.
#         #feature map which is generated according to when the order of fm
#         if len(self.learner_dic)==0 or self.GENERATE_new_fm_set:
#             self.GENERATE_new_fm_set = False
#             self.call_generator_index.append(self.i)
#             self.feature_map_set, init_seed_fm = self.fm_generator.feature_map_set_generator(self.seed_fm)
#             #generate id for each feature map. Note that fm id is also learner id
#             if self.seed_fm is None:
#                 self.seed_fm = init_seed_fm
#             for fm in self.feature_map_set:
#                 if fm not in self.fm_id_dic: 
#                     self.fm_id_dic[fm] = len(self.fm_id_dic)
#                     fm_id = self.fm_id_dic[fm]
#                     if fm_id not in self.learner_dic:
#                         inter_arg = fm2inter_arg(fm)
#                         print('add learner',inter_arg)
#                         # vw_model = pyvw.vw(" --quiet " + str(inter_arg))
#                         self.learner_dic[fm_id]= self._initialize_learner(str(inter_arg))
#                         # self.learner_dic[fm_id]= Learner(fm, fm_id, self.loss_func, " --quiet " + str(inter_arg))
#                 #TODO: need to create learner for all fm_set
#             #generate a ranked list of the feature map set (if policy_budget is not None and 
#             # is smaller than the total size of the feature map set, it will return the top ranked
#             # feature map)
#             # print('feature map set', self.feature_map_set)
#             self.feature_map_ordered_list = self.fm_generator.rank_feature_map_set(self.feature_map_set, 
#                 self.policy_budget)
#         self.seed_fm_id = self.fm_id_dic[self.seed_fm]



# class TrainableVW(tune.Trainable):
    
#     """
#     setup function is invoked once training starts.

#     step is invoked multiple times. Each time, the Trainable object 
#     executes one logical iteration of training in the tuning process, 
#     which may include one or more iterations of actual training.

#     cleanup is invoked when training is finished.
#     """
#     const = 0.1
#     trainable_class = pyvw.vw
#     min_resource_budget = 100
#     def setup(self, **config):
#         """setup function is invoked once training starts.
#         """
#         self.model = TrainableVW.trainable_class(**config)
#         self.update_count = 0
#         self.datasample_count = 0

#         #info about resource consumption
#         self.resource_budget = TrainableVW.min_resource_budget
#         self.resource_used = 0

#         #info about performance
#         self._sum_loss = 0
#         self._eval_count = 0
#         self.avg_loss = 0 # usually use empirical negative loss as the empirical score
#         self.avg_loss_cb = 0 # confidence bound of the empirical score estimation

#         self.config = config if config else {}
       
#     def step(self, data_x, data_y):
#         """step is invoked multiple times. Each time, the Trainable object 
#         executes one logical iteration of training in the tuning process, which
#         may include one or more iterations of actual training.
#         """
#         self.model.predict(data_x)
#         self.model.learn(data_x, data_y)
#         #update resource consumption 
#         self.resource_used += 1.0
#         if self.resource_used >= self.resource_budget:
#             self.resource_budget *= 2
#         self.update_count +=1.0
#         self.datasample_count += len(data_y)
        
#         # update performance measurements
#         self._sum_loss = self.model.get_sum_loss()
#         self._eval_count += len(data_y)
#         self.avg_loss = self._sum_loss/self._eval_count

#         #TODO: addd the calculation of loss_cb
#         self.avg_loss_cb = TrainableVW.const*math.sqrt(1.0/self.datasample_count) 
#         self.avg_loss_ucb = self.avg_loss + self.avg_loss_cb
#         self.avg_loss_ucb = self.avg_loss + self.avg_loss_cb
#         result = {
#             'avg_loss': self.avg_loss,
#             'avg_loss_cb': self.avg_loss_cb,
#             'avg_loss_ucb': self.avg_loss_ucb,
#             'avg_loss_lcb': self.avg_loss_lcb,
#             'resource_used': self.resource_used,
#             'resource_budget': self.resource_budget,
#         }
#         #in addition to the results, Tuen will automatically log many metrics,
#         # such as config, trial_id, timesteps_tota and etc,
#         # https://docs.ray.io/en/master/tune/user-guide.html#tune-autofilled-metrics
#         return result


def trainable_vw(learner_class, fixed_hp_config, data_buffer, hp_config):
    print(hp_config)
    vw_learner = learner_class(**hp_config, **fixed_hp_config)
    data_sample_count = 0
    total_resource_consumed = 0
    const = 0.1
    while True:
        data = data_buffer.feed()
        print('data', data)
        if data:
            vw_learner.learn(data[0])
            data_sample_count += len(data)
            #TODO: where to get c
            c = 1.0
            resource_consumed = len(data)*c
            loss_sum = vw_learner.get_sum_loss()

            loss_avg = loss_sum/float(data_sample_count)
            cb = const*math.sqrt(1.0/data_sample_count) 
            total_resource_consumed += resource_consumed

            result = {
            'total_resource_consumed': total_resource_consumed,
            'data_sample_count': data_sample_count,
            'loss_sum': loss_sum,
            'loss_avg': loss_avg, 
            'cb': cb,
            'loss_ucb': loss_avg + cb, 
            'loss_lcb': loss_avg - cb,
            'trained_model': vw_learner,}
        else:
            result = None
        tune.report(**result)


def strip_noninteractive_feature(config):
    import copy
    new_config = copy.deepcopy(config)
    for k,v in new_config.items():
        if k == 'q':
            to_remove = []
            for inter in v: 
                if len(inter) ==1: to_remove.append(inter)
            for inter in to_remove: v.remove(inter) 
        new_config[k] = v
    return new_config

# -*- coding: UTF-8 -*-

########################################################
# __Author__: Triskelion <info@mlwave.com>             #
# Kaggle competition "Display Advertising Challenge":  #
# http://www.kaggle.com/c/criteo-display-ad-challenge/ #
# Credit: Zygmunt Zając <zygmunt@fastml.com>           #
########################################################

from datetime import datetime
from csv import DictReader

def csv_to_vw(loc_csv, loc_output, train=True):
  """
  Munges a CSV file (loc_csv) to a VW file (loc_output). Set "train"
  to False when munging a test set.
  TODO: Too slow for a daily cron job. Try optimize, Pandas or Go.
  """
  start = datetime.now()
  print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
  
  with open(loc_output,"wb") as outfile:
    for e, row in enumerate( DictReader(open(loc_csv)) ):
	
	  #Creating the features
      numerical_features = ""
      categorical_features = ""
      for k,v in row.items():
        if k not in ["Label","Id"]:
          if "I" in k: # numerical feature, example: I5
            if len(str(v)) > 0: #check for empty values
              numerical_features += " %s:%s" % (k,v)
          if "C" in k: # categorical feature, example: C2
            if len(str(v)) > 0:
              categorical_features += " %s" % v
			  
	  #Creating the labels		  
      if train: #we care about labels
        if row['Label'] == "1":
          label = 1
        else:
          label = -1 #we set negative label to -1
        outfile.write( "%s '%s |i%s |c%s\n" % (label,row['Id'],numerical_features,categorical_features) )
		
      else: #we dont care about labels
        outfile.write( "1 '%s |i%s |c%s\n" % (row['Id'],numerical_features,categorical_features) )
      
	  #Reporting progress
      if e % 1000000 == 0:
        print("%s\t%s"%(e, str(datetime.now() - start)))

  print("\n %s Task execution time:\n\t%s"%(e, str(datetime.now() - start)))

#csv_to_vw("d:\\Downloads\\train\\train.csv", "c:\\click.train.vw",train=True)
#csv_to_vw("d:\\Downloads\\test\\test.csv", "d:\\click.test.vw",train=False)