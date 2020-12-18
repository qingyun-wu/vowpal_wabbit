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
from typing import Dict, Optional, List, Tuple
from util import strip_noninteractive_feature
from sklearn.metrics import mean_squared_error
import logging
logger = logging.getLogger(__name__)

class TrainableVWTrial:

    
    """ Class for trainable VW
        Args:
            trial_id (str): the id of the trial
            feature_set (set): the set of features (more precisely, name spaces in vw) to use in 
                building a vw model
            fixed_config (dict): the dictionary of fixed configurations 
            init_result (dict): init result
        
        TODO: better organize the results
        1. training related results (need to be updated in the trainable class)
        2. result about resources allocation (need to be updated externally)

        Other info: 
            - Info about name spaces in vw: 
            https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Namespaces
            - Command-line options related to name-spaces:
                Option	Meaning
                --keep c	Keep a name-space staring with the character c
                --ignore c	Ignore a name-space starting with the character c
                --redefine a:=b	redefine namespace starting with b as starting with a
                --quadratic ab	Cross namespaces starting with a & b on the fly to generate 2-way interacting features
                --cubic abc	Cross namespaces starting with a, b, & c on the fly to generate 3-way interacting features
    """
    model_class = pyvw.vw
    cost_unit = 1.0
    const = 0.5
    # quartic_config_key = 'q'
    # cubic_config_key = 'cubic'
    interactions_config_key = 'interactions'
    # the following ones are not used for now
    ignore_config_key = 'ignore'
    keep_config_key = 'keep'
    redefine_config_key = 'redefine'
    LOSS_MIN = 0
    LOSS_MAX = np.inf
    def __init__(self, trial_id, feature_set, namespace_feature_dim, fixed_config, init_result): 
        """ need to provide init_result
            #TODO: the config should be a set instead of a list
        """
        self.trial_id = trial_id
        self._feature_set = feature_set
        self._vw_fs_config = self._feature_set_to_vw_fs_config(self._feature_set)
        self._fixed_config = fixed_config
        self.trained_model = TrainableVWTrial.model_class(
            **self._vw_fs_config, **self._fixed_config )
        # TODO: how to get the dim of parameters
        self._dim = self.get_dim(namespace_feature_dim, feature_set)
        self._resouce_used_attr = 'resource_used'
        self._resource_budget_attr = 'resource_budget'
        self._result = {'config': self._feature_set,}
        assert init_result is not None, 'need to provide init result'
        for k,v in init_result.items():
            if k != 'config': self._result[k] = v
        #TODO: why I can not do this? 
        # self._result['config']  = copy.deepcopy(self._feature_set) 

    @staticmethod
    def get_dim(namespace_feature_dim, namespace_set):
        #TODO: how to decide the dimensionality of the interaction features?
        dim = 0
        for f in namespace_set:
            for c in f:
                dim += namespace_feature_dim[c]
        logger.debug('dim %s %s', dim, namespace_set)
        return dim

    def train_vw(self, data, y, old_result):
        """ train vw model
        """
        y_pred = self.trained_model.predict(data)
        self.trained_model.learn(data)
        # update training related results accordingly
        loss_sum = self.trained_model.get_sum_loss()
        # TODO: remove the hard code part and assumeing the size of data sample each step is 1
        data_sample_size = 1
        # TODO: need to re-write the dim of feature
        self._result[self._resouce_used_attr] +=  \
            self._dim*TrainableVWTrial.cost_unit*data_sample_size
        self._result['data_sample_count'] += data_sample_size
        self._result['loss_sum'] = loss_sum
        # TODO: what if the loss function in the final evaluation phase is not the same as the one in get_sum_loss?
        
        # loss = mean_squared_error([y_pred], [y])
        # loss_sum = loss + old_result['loss_sum']
        # print('loss sum', loss_sum)
        self._result['loss_avg'] =  self._result['loss_sum']/self._result['data_sample_count']
        self._result['cb'] =  TrainableVWTrial.const*math.sqrt(self._dim/self._result['data_sample_count']) 
        self._result['loss_ucb'] = min(self._result['loss_avg'] + self._result['cb'],
            TrainableVWTrial.LOSS_MAX)
        self._result['loss_lcb'] = max(self._result['loss_avg'] - self._result['cb'], 
            TrainableVWTrial.LOSS_MIN)
        logger.debug('result %s', self._result)
    def update_res_budget_in_result(self, result):
        """ the resource update usually needs to be updated externally 
        by some algorithm or scheduler
        """
        if self._resource_budget_attr in result:
            self._result[self._resource_budget_attr] = result[self._resource_budget_attr]

    def get_result(self):
        return self._result

    @classmethod
    def _feature_set_to_vw_fs_config(cls, feature_set):
        vw_fs_config = {}
        for c in feature_set: 
            if len(c) == 1: hp_name = None 
            else: hp_name = cls.interactions_config_key
            if hp_name:
                if hp_name not in vw_fs_config:
                    vw_fs_config[hp_name] = []
                vw_fs_config[hp_name].append(c)
            print('vw confi', vw_fs_config)
        return vw_fs_config

        
class AutoVW:
    #TODO: do we need to rewrite every function of vw?
    """The AutoOnlineLearner object is auto online learning object.
    class variable: learner_class, which is set to be vw learner
    Args:
        min_resource_budget (int): the number of min resource
        policy_budget (int): the number of policy budget
        fixed_hp_config (dict): the default config of the non-changing hyperparameters
        init_feature_set (dict): the init config for the hyperparameters to tune
    """
    learner_class = pyvw.vw
    MAX_NUM = 2**16
    def __init__(self, min_resource_budget: int, policy_budget: int, 
        namespace_feature_dim: dict, inter_order: int, fixed_hp_config: dict, scheduler_name: str=None, **kwargs):
        from AML.blendsearch.trial_runner import OnlineTrialRunnerWithChampion
        from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, HyperBandScheduler
        from AML.blendsearch.scheduler.online_scheduler import OnlineDoublingScheduler 
        from AML.blendsearch.searcher.online_searcher import BaseFeatureSearcher, FeatureInteractionSearcher 
        self._min_resource_budget = min_resource_budget
        self._policy_budget = policy_budget
        self._fixed_hp_config = fixed_hp_config
        self._init_feature_set = set(namespace_feature_dim.keys())
        self._namespace_feature_dim = namespace_feature_dim
        self._learner_class = AutoVW.learner_class
        self._best_trial = None 
        self._all_trials = {} # trial_id -> trial
        self._metric = 'loss_ucb'
        self._mode = 'min'
        self._init_result = {
        'resource_used': 0,
        'resource_budget': self._min_resource_budget,
        'data_sample_count': 0,
        'loss_sum': np.inf,
        'loss_avg': np.inf, 
        'cb': 1.0,
        'loss_ucb': np.inf, 
        'loss_lcb': np.inf,
        'is_newly_promoted_champion': False,
        'is_champion': False,
        'priority_factor': 1.0,
        }
        self._loss_func = self._get_loss_func_from_config(self._fixed_hp_config)
        # the incumbent_learner is the incumbent vw when the AutoVW is called
        # the incumbent_learner is udpated everytime the learn function is called.
        self._incumbent_learner = AutoVW.learner_class(
            **self._fixed_hp_config)
        self._y_pred = None 
        self._sum_loss = 0.0
        self._learner_dic = {}
        # self._searcher = NaiveSearcher(
        #     metric='loss_ucb',
        #     mode= self._mode ,
        #     init_feature_set = init_feature_set,
        #     init_result = self._init_result,
        #     )
        self._iter = 0

        self._searcher = FeatureInteractionSearcher(
            metric='loss_ucb',
            mode='min',
            init_feature_set = self._init_feature_set,
            init_result = self._init_result,
            inter_order = inter_order,
            )    
    
        online_scheduler = OnlineDoublingScheduler(
                resource_used_attr = "resource_used",
                resource_budget_attr = "resource_budget",
                doubling_factor = 2,
                )
        asha_scheduler = ASHAScheduler(
            time_attr='resource_used',
            metric='loss_ucb',
            mode='min',
            max_t= AutoVW.MAX_NUM,
            grace_period=10,
            reduction_factor=2,
            brackets=1)

        hyperband_scheduler = HyperBandScheduler(
            time_attr='resource_used',
            reward_attr='loss_avg',
            metric='loss_ucb',
            mode='min',
            max_t= 10000,#AutoVW.MAX_NUM,
            reduction_factor=2,
            )
        if scheduler_name and 'hyperband' in scheduler_name:
            self._scheduler = hyperband_scheduler
        else:
            self._scheduler = online_scheduler
        
        self._trial_runner = OnlineTrialRunnerWithChampion(
            search_alg=self._searcher,
            scheduler= self._scheduler,
            running_budget=self._policy_budget,
            resource_used_attr="resource_used",
            resource_budget_attr="resource_budget",
            doubling_factor = 2,
            metric = 'loss_lcb',
            mode = 'min',
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

    def _get_loss_func_from_config(self, config):
        loss_func_name = config['loss_function']
        if 'squared' in loss_func_name:
            loss_func = mean_squared_error
        else:
            NotImplementedError
        return loss_func

    def _train_vw_trial(self, trial, data, y):
        # if trial does not exist, create a new model
        if trial.trial_id not in self._learner_dic:
            self._learner_dic[trial.trial_id] = TrainableVWTrial(trial.trial_id, 
                trial.config, self._namespace_feature_dim, self._fixed_hp_config, self._init_result)
        # get trial result
        self._learner_dic[trial.trial_id].update_res_budget_in_result(trial.result)
        # train the model for one step
        self._learner_dic[trial.trial_id].train_vw(data, y, trial.result)
        # update trial result
        #TODO: should not change the result of a trial out side the trial runner
        trial.result = self._learner_dic[trial.trial_id].get_result() 
        return self._learner_dic[trial.trial_id].get_result() 

    def predict(self, x):
        """ Predict on the input example
        """
        self._y_predict = self._incumbent_learner.predict(x)
        return self._y_predict

    def learn(self, x, y = None):
        """Perform an online update
        Args: 
            x (vw_example/str/list) : 
                examples on which the model gets updated
            y (float): label of the example (optional) 
            #TODO: label can be obtained from x
        """
        scheduled_trials = self._trial_runner.schedule_trials()
        # clean up old models from self._learner_dics
        self._clean_up_old_models(scheduled_trials) 
        for trial in scheduled_trials:
            result = self._train_vw_trial(trial, x, y)
            self._trial_runner.process_trial_result(trial, result)
        self._trial_runner.process_running_trials(scheduled_trials)
        self._best_trial = self._select_trial_for_prediction(scheduled_trials, self._metric, self._mode)
        self._incumbent_learner = self._learner_dic[self._best_trial.trial_id].trained_model
        self._iter +=1

    def _clean_up_old_models(self, scheduled_trials):

        last_running_trial_ids = list(self._learner_dic.keys())
        scheduled_trial_ids = [trial.trial_id for trial in scheduled_trials]
        removed_ones = [x for x in last_running_trial_ids if x not in scheduled_trial_ids]
        newly_added_ones = [x for x in scheduled_trial_ids if x not in last_running_trial_ids]
        if removed_ones:
            logger.debug('====clean up at iteration %s====', self._iter)
            logger.debug('    clean up %s models, including %s ', len(removed_ones), removed_ones)
            logger.debug('    add      %s models, including %s', len(newly_added_ones), newly_added_ones)
            logger.debug('now running  %s models, including %s', len(scheduled_trial_ids), scheduled_trial_ids)
            for trail_id in removed_ones:
                del self._learner_dic[trail_id]
        
    def _update_sum_loss(self, y):
        loss = self._loss_func([self._y_pred], y)
        # destory self._y_predict after calculating the loss
        self._y_pred = None
        self._sum_loss += loss

    def get_sum_loss(self):
        self._sum_loss
        # TODO: need to implement get_sum_loss

    def finished(self):
        return self._incumbent_learner.finished()

    @staticmethod
    def _select_trial_for_prediction(running_trials, metric, mode):
        best_score = float('+inf') if mode=='min' else float('-inf')
        best_trial = None
        if len(running_trials)!=0:
            for trial in running_trials:
                score = trial.result[metric]
                if 'min' == mode and  score < best_score or \
                    'max' == mode and score > best_score:
                    best_score = score
                    best_trial = trial
        logger.info('trial for prediction %s', best_trial.config)
        return best_trial
    
    


    