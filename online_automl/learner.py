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
from util import get_y_from_vw_example
import logging
logger = logging.getLogger(__name__)
LARGE_NUM = 1000000
MODEL_SELECTION_THRES = 10
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
    const = 0.1
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
        self._dim = self.get_dim(namespace_feature_dim, feature_set)/5.0
        self._resouce_used_attr = 'resource_used'
        self._result = {'config': self._feature_set,}
        self._data_sample_size = 0
        self._bound_of_loss = 1.0
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

    def train_vw(self, data, y, predicted=False):
        """ train vw model
        """
        if not predicted: y_pred = self.trained_model.predict(data)
        self.trained_model.learn(data)
        # update training related results accordingly
        loss_sum = self.trained_model.get_sum_loss()
        # TODO: remove the hard code part and assumeing the size of data sample each step is 1
        data_sample_size = 1
        # TODO: need to re-write the dim of feature
        self._result[self._resouce_used_attr] +=  TrainableVWTrial.cost_unit*data_sample_size
            #self._dim*TrainableVWTrial.cost_unit*data_sample_size
        self._result['data_sample_count'] += data_sample_size
        self._data_sample_size += data_sample_size
        new_loss = (loss_sum - self._result['loss_sum'])/data_sample_size if \
            self._result['loss_sum'] else loss_sum/data_sample_size
        self._result['loss_sum'] = loss_sum
        # TODO: what if the loss function in the final evaluation phase is not the same as the one in get_sum_loss?
        self._result['loss_avg'] =  self._result['loss_sum']/self._result['data_sample_count']
        self._bound_of_loss = max(self._bound_of_loss, new_loss)
        # logger.debug('bound of loss %s',self._bound_of_loss )
        self._result['cb'] =  self._bound_of_loss*TrainableVWTrial.const*math.sqrt(self._dim/self._result['data_sample_count']) 
        self._result['comp'] = self._bound_of_loss*TrainableVWTrial.const*math.sqrt(self._dim/self._result['data_sample_count']) 
        self._result['loss_ucb'] = min(self._result['loss_avg'] + self._result['cb'],
            TrainableVWTrial.LOSS_MAX)
        self._result['loss_lcb'] = max(self._result['loss_avg'] - self._result['cb'], 
            TrainableVWTrial.LOSS_MIN)
        # logger.debug('result %s', self._result)

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
        return vw_fs_config

        
class AutoVW:
    #TODO: do we need to rewrite every function of vw?
    """The AutoOnlineLearner object is auto online learning object.
    class variable: learner_class, which is set to be vw learner
    Args:
        min_resource_budget (int): the number of min resource
        concurrent_running_budget (int): the number of policy budget
        fixed_hp_config (dict): the default config of the non-changing hyperparameters
        init_feature_set (dict): the init config for the hyperparameters to tune
    """
    learner_class = pyvw.vw
    def __init__(self, 
        min_resource_budget: float, 
        concurrent_running_budget: int, 
        namespace_feature_dim: dict, 
        fixed_hp_config: dict, 
        champion_test_policy: str,
        trial_runner_name: str='SuccessiveDoubling',
        model_select_policy: str=None,
        keep_champion_running: int =0,
        keep_incumbent_running: int =0,
        ):
        from AML.blendsearch.tune.trial_runner import BaseOnlineTrialRunner, OnlineTrialRunnerwIncumbent
        from AML.blendsearch.tune.auto_cross_trial_runner import AutoCrossOnlineTrialRunner,AutoCrossOnlineTrialRunnerPlus
        from AML.blendsearch.scheduler.online_scheduler import OnlineSuccessiveDoublingScheduler, \
            SDwChampionScheduler, SDwBestChallengerAndChampionScheduler
        from AML.blendsearch.scheduler.online_successive_halving import OnlineSHAwChampion
        from AML.blendsearch.searcher.online_searcher import ChampionFrontierSearcher 
        self._concurrent_running_budget = concurrent_running_budget
        self._min_resource_budget = min_resource_budget
        self._trial_runner_name = trial_runner_name
        self._fixed_hp_config = fixed_hp_config
        self._namespace_feature_dim = namespace_feature_dim 
        self._model_select_policy = model_select_policy if model_select_policy else 'online_ucb'
        self._model_selection_metric = 'loss_ucb'
        self._model_selection_mode = 'min'
        self._init_result = {
            'resource_used': 0,
            'data_sample_count': 0,
            'loss_sum': np.inf,
            'loss_avg': np.inf, 
            'cb': 1.0,
            'loss_ucb': np.inf, 
            'loss_lcb': np.inf,}
        self._loss_func = self._get_loss_func_from_config(self._fixed_hp_config)
        # the incumbent_learner is the incumbent vw when the AutoVW is called
        # the incumbent_learner is udpated everytime the learn function is called.
        self._incumbent_trial_id = None
        self._incumbent_trial_model = None
        self._y_pred = None 
        self._sum_loss = 0.0
        self._learner_dic = {}
        self._scheduled_trials = []
        self._iter = 0
        # keep_champion_running = False
        progressive_searcher = ChampionFrontierSearcher(
            metric='loss_ucb',
            mode='min',
            init_feature_set = set(namespace_feature_dim.keys()),
            init_result = self._init_result.copy(),
            min_resource_budget = min_resource_budget)    

        scheduler_common_args = {
            'resource_used_attr': "resource_used",
            #  'doubling_factor': 2,
            }

        trial_runner_commmon_args ={
            'search_alg': progressive_searcher,
            "champion_test_policy": champion_test_policy,
            'min_resource_budget': min_resource_budget,
            'resource_used_attr': "resource_used",
            # 'keep_incumbent_running': keep_incumbent_running
            }

        ## schedulers
        online_sd_scheduler = OnlineSuccessiveDoublingScheduler(**scheduler_common_args)
        online_sd_w_champion_scheduler = SDwChampionScheduler(**scheduler_common_args)
        online_sd_w_bestchallenger_champion_scheduler = SDwBestChallengerAndChampionScheduler(**scheduler_common_args)
        
        online_sha_scheduler = OnlineSHAwChampion(**scheduler_common_args)   

        print(self._trial_runner_name )
        if self._trial_runner_name == 'autocross':
            self._trial_runner = AutoCrossOnlineTrialRunner(
                scheduler= online_sha_scheduler,
                **trial_runner_commmon_args,
                )
        elif self._trial_runner_name == 'autocross+':
            self._trial_runner = AutoCrossOnlineTrialRunnerPlus(
                scheduler= online_sha_scheduler,
                **trial_runner_commmon_args,
                )
        elif self._trial_runner_name == 'autocross+sd':
            self._trial_runner = AutoCrossOnlineTrialRunner(
                scheduler= online_sd_scheduler,
                **trial_runner_commmon_args,
                )
        elif self._trial_runner_name == 'SuccessiveDoublingsha':
            self._trial_runner = BaseOnlineTrialRunner(
                scheduler= online_sha_scheduler,
                **trial_runner_commmon_args
                )
        elif self._trial_runner_name == 'SuccessiveDoubling':
            if keep_champion_running and keep_incumbent_running:
                logger.info('using both, %s %s', keep_incumbent_running, keep_champion_running)
                my_scheduler = online_sd_w_bestchallenger_champion_scheduler
            elif keep_champion_running:
                logger.info('using champion, %s %s', keep_incumbent_running, keep_champion_running)
                my_scheduler = online_sd_w_champion_scheduler
                # my_scheduler = online_sha_scheduler
            else:
                my_scheduler = online_sd_scheduler
            self._trial_runner = OnlineTrialRunnerwIncumbent(
                scheduler=my_scheduler,
                **trial_runner_commmon_args
                )  
        else:
            NotImplementedError
            
    @property  
    def incumbent_vw(self):
        return self._incumbent_trial_model

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

    def _construct_vw_model_from_trial(self, trial):
        if trial.trial_id not in self._learner_dic:
            self._learner_dic[trial.trial_id] = TrainableVWTrial(trial.trial_id, 
                trial.config, self._namespace_feature_dim, self._fixed_hp_config, self._init_result)

    def predict(self, x):
        """ Predict on the input example
        """
        if not self._incumbent_trial_id:
            # do the initial scheduling
            print(self._trial_runner_name )
            self._scheduled_trials = self._trial_runner.schedule_trials_to_run(\
                self._concurrent_running_budget, self._incumbent_trial_id)
            assert len(self._learner_dic) == 0, 'initial prediction'
            self._cleanup_old_and_construct_new_model(self._scheduled_trials)
            assert self._trial_runner.champion_trial.trial_id in [t.trial_id for t in self._scheduled_trials]
            assert self._trial_runner.champion_trial.trial_id in self._learner_dic.keys()
            self._incumbent_trial_id = self._trial_runner.champion_trial.trial_id
            self._incumbent_trial_model = self._learner_dic[self._trial_runner.champion_trial.trial_id].trained_model
            assert len(self._learner_dic) <= self._concurrent_running_budget
        self._y_predict = self._learner_dic[self._incumbent_trial_id].trained_model.predict(x)
        # self._y_predict = 1.0
        return self._y_predict

    def learn(self, x):
        """Perform an online update
        Args: 
            x (vw_example/str/list) : 
                examples on which the model gets updated
            y (float): label of the example (optional) 
            #TODO: label can be obtained from x
        """
        y = get_y_from_vw_example(x)
        # clean up old models from self._learner_dics
        self._cleanup_old_and_construct_new_model(self._scheduled_trials) 
        # train the model that are scheduled in self._learner_dic.
        for trial_id in self._learner_dic.keys():
            if self._incumbent_trial_id and self._incumbent_trial_id == trial_id:
                predicted = True 
            else: predicted = False
            # train the model for one step
            self._learner_dic[trial_id].train_vw(x, y, predicted)
            result = self._learner_dic[trial_id].get_result() 
            # report result to the trial runner
            self._trial_runner.process_trial_result(trial_id, result)

        best_trial = self._select_trial_for_prediction()
        self._incumbent_trial_id = best_trial.trial_id
        self._incumbent_trial_model = best_trial.trained_model
        self._iter +=1
        # schedule which models to train 
        self._scheduled_trials = self._trial_runner.schedule_trials_to_run(\
            self._concurrent_running_budget, self._incumbent_trial_id)
        assert len(self._scheduled_trials) <= self._concurrent_running_budget
        
    def _cleanup_old_and_construct_new_model(self, scheduled_trials):
        if scheduled_trials:
            logger.debug('scheduled_trials %s', [trial.trial_id for trial in scheduled_trials])
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
                    # TODO: how can we ensure we do not easily remove the incumbent_trials
                    if trail_id == self._incumbent_trial_id:
                        logger.critical('cleaning up the incumbent trial')
                    del self._learner_dic[trail_id]
            for trial in scheduled_trials:
                # create a new model
                self._construct_vw_model_from_trial(trial)
        
    def _update_sum_loss(self, y):
        loss = self._loss_func([self._y_pred], y)
        # destory self._y_predict after calculating the loss
        self._y_pred = None
        self._sum_loss += loss

    def get_sum_loss(self):
        self._sum_loss
        # TODO: need to implement get_sum_loss

    def finished(self):
        return self._incumbent_trial_model.finished()

    def _select_trial_for_prediction(self,):
        # assert champion running or best_challenger running
        # select from best_challenger and champion
        if 'avg'  in self._model_select_policy:
            self._model_selection_metric = 'loss_avg'
        else:
            self._model_selection_metric = 'loss_ucb'

        best_score = float('+inf') if self._model_selection_mode=='min' else float('-inf')
        best_trial = None
        best_trial_id = None
        if len(self._learner_dic)!=0:
            for key, trial in self._learner_dic.items():
                score = trial._result[self._model_selection_metric]
                if 'min' == self._model_selection_mode and  score < best_score or \
                    'max' == self._model_selection_mode and score > best_score:
                    best_score = score
                    best_trial = trial
                    best_trial_id = key    

        if 'threshold' in self._model_select_policy:
            incumbent_selection_threshod = min(self._min_resource_budget, MODEL_SELECTION_THRES)
            
            if best_trial._data_sample_size >= incumbent_selection_threshod \
                or self._incumbent_trial_id not in self._learner_dic:
                logger.debug('selecting the best trial %s at iter %s', best_trial.trial_id, self._iter)
                return best_trial
            else: 
                logger.debug('%s, %s %s',  best_trial._data_sample_size,self._min_resource_budget , best_trial.trial_id)
                logger.debug('selecting the incumbent trial %s at iter %s',self._incumbent_trial_id, self._iter )
                return self._learner_dic[self._incumbent_trial_id]
        else:
            logger.info('trial for prediction %s %s', best_trial_id, best_trial._result)
            return best_trial