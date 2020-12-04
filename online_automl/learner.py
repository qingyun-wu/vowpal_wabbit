from vowpalwabbit import pyvw
import uuid
import numpy as np
import copy
import itertools
import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from automl.search_space import FSSProblem
# from automl.search_policy import LocalSearcher
from ray import tune
import math
from functools import partial, partialmethod
from searcher import FeatureSearcher
from typing import Dict, Optional, List, Tuple

# class TrainedVW:

#     def __init__(self, config, model):
#         self.config = config
#         self.id = config2id(config)
#         self.model = model
#         self._data_sample_count = 0
#         self._total_resource_consumed = 0
#         self._const = 0.1
    
#     def learn(self):
#         self.model.learn(data)
#         self.data_sample_count += len(data)
        
#         self.loss_sum = self.model.get_sum_loss
#         #TODO: where to get c
#         resource_consumed = 1.0
#         self.total_resource_consumed += resource_consumed

#         loss_avg = loss_sum/float(data_sample_count)
#         cb = const*math.sqrt(1.0/data_sample_count) 
       
#         result = {
#         'config': config,
#         'total_resource_consumed': self.total_resource_consumed,
#         'data_sample_count': self._data_sample_count,
#         'loss_sum': loss_sum,
#         'loss_avg': loss_avg, 
#         'cb': cb,
#         'loss_ucb': loss_avg + cb, 
#         'loss_lcb': loss_avg - cb,
#         }
#         return result

#     @staticmethod
#     def config2id(config):
#         config_id = tuple(config)
#         return config_id
#TODO: what is the name of the algorithm: how to reflect the feature generation part?

class AutoVW:
    #TODO: do we need to rewrite every function of vw?
    """The AutoOnlineLearner object is auto online learning object.
    class variable: learner_class, which is set to be vw learner
    Args:
        min_resource_budget (int): the number of min resource
        policy_budget (int): the number of policy budget
        fixed_hp_config (dict): the default config of the non-changing hyperparameters
        hp2tune_init_config (dict): the init config for the hyperparameters to tune
    """
    learner_class = pyvw.vw
    def __init__(self, min_resource_budget: int, policy_budget: int, 
        hp2tune_init_config: dict, fixed_hp_config: dict, **kwargs):
        from AML.blendsearch.trial_runner import OnlineTrialRunner
        from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
        from AML.blendsearch.scheduler.online_scheduler import OnlineDoublingScheduler 
        from AML.blendsearch.searcher.online_searcher import NaiveSearcher, FeatureInteractionSearcher 
        self._min_resource_budget = min_resource_budget
        self._policy_budget = policy_budget
        self._fixed_hp_config = fixed_hp_config
        self._hp2tune_init_config = hp2tune_init_config
        self._learner_class = AutoVW.learner_class
        self._init_result = {
        'resource_used': 0,
        'resource_budget': self._min_resource_budget,
        'data_sample_count': 0,
        'loss_sum': 0,
        'loss_avg': 0, 
        'cb': 100,
        'loss_ucb': 0+100, 
        'loss_lcb': 0-100,
        }
        
        self._learner_dic = {} 
        # the incumbent_learner is the incumbent vw when the AutoVW is called
        # the incumbent_learner is udpated everytime the learn function is called.
        print(self._hp2tune_init_config, self._fixed_hp_config)
        self._incumbent_learner = AutoVW.learner_class(
            **self._fixed_hp_config, **self._hp2tune_init_config)
        # config_id = self._get_config_id(self._hp2tune_init_config['q'])
        self._learner_dic = {}

        self._searcher = NaiveSearcher(
            metric='loss_ucb',
            mode='min',
            init_config = hp2tune_init_config,
            init_result = self._init_result,
            )
        self._scheduler = OnlineDoublingScheduler(
                resource_used_attr = "resource_used",
                resource_budget_attr = "resource_budget",
                doubling_factor = 2
                )
        self._trial_runner = OnlineTrialRunner(
            search_alg=self._searcher,
            scheduler=self._scheduler,
            running_budget=self._policy_budget
            )
    @property  
    def incumbent_vw(self):
        return self._incumbent_learner

    @property  
    def init_result(self):
        return self._init_result

    @staticmethod
    def _get_config_id(config_value):
        return tuple(config_value)

    def _train_vw(self, old_result, config_id, config, data):
        if config_id not in self._learner_dic:
            print(config_id, config)
            assert len(config) == 1, 'only tunning 1 hp'
            self._learner_dic[config_id] = self._learner_class(**config, **self._fixed_hp_config) 
        # vw_learner = pyvw.vw(config, **self._fixed_hp_config)
        data_sample_count = old_result['data_sample_count']
        total_resource_consumed = old_result['resource_used']
        const = 0.1
        self._learner_dic[config_id].learn(data)
        data_sample_count += len(data)
        #TODO: where to get c
        c = 1.0
        resource_consumed = len(data)*c
        loss_sum = self._learner_dic[config_id].get_sum_loss()

        loss_avg = loss_sum/float(data_sample_count)
        cb = const*math.sqrt(1.0/data_sample_count) 
        total_resource_consumed += resource_consumed
        # current_resource_budget = trial 
        result = {
        'config': config,
        'resource_used': total_resource_consumed,
        'data_sample_count': data_sample_count,
        'loss_sum': loss_sum,
        'loss_avg': loss_avg, 
        'cb': cb,
        'loss_ucb': loss_avg + cb, 
        'loss_lcb': loss_avg - cb,
        }
        return result
                  
    def predict(self, x):
        """ Predict on the input example
        """
        print('predict',self._incumbent_learner.predict(x) )
        return self._incumbent_learner.predict(x)

    def learn(self, x, y = None):
        """Perform an online update
        Args: 
            x (vw_example/str/list) : 
                examples on which the model gets updated
            y (float): label of the example (optional) 
            #TODO: label can be obtained from x
        """
        running_trials = self._trial_runner.get_running_trials()
        for trial in running_trials:
            result = self._train_vw(trial.result, trial.trial_id, trial.config, x)
            self._trial_runner.process_trial(result, trial)
        best_trial = self._trial_runner.get_best_running_trial(metric ='loss_ucb')
        self._incumbent_learner = self._learner_dic[best_trial.trial_id]
    def get_sum_loss(self):
        return self._incumbent_learner.get_sum_loss()

    def finished(self):
        return self._incumbent_learner.finished()
    
    


    