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

def trainable_vw(learner_class, fixed_hp_config, data_buffer, hp_config):
    print(hp_config)
    vw_learner = learner_class(**hp_config, **fixed_hp_config)
    data_sample_count = 0
    total_resource_consumed = 0
    const = 0.1
    # while True:
    for i in range(2):
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
            'trained_model': vw_learner,
            }
            tune.report(**result)


def trainable_vw_test(AutoML_instance):
    vw_learner = AutoML_instance._learner_class(**AutoML_instance._hp_config, **AutoML_instance._fixed_hp_config)
    data_sample_count = 0
    total_resource_consumed = 0
    const = 0.1
    # while True:
    for i in range(2):
        data = AutoML_instance.data_buffer.feed()
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
            # 'trained_model': vw_learner,
            }
            tune.report(**result)
class DataBuffer:

    def __init__(self, visit_limit=1):
        self.data_set = []
        self._counter = 0
        self._visit_limit = visit_limit

    def collect(self, new_data):
        self.data_set += new_data

    def feed(self, num = None):
        self._counter +=1
        data = self.data_set if not num else self.data_set[-num:]
        # if reach the limit for the counter, cleanup the list
        print('counter', self._counter)
        if self._counter > self._visit_limit:
            self.data_set = [] if not num else self.data_set.pop(num)
            self._counter = 0
        return data

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
        from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
        # from AML.scheduler.online_scheduler import OnlineFeatureSelectionScheduler 
        self._min_resource_budget = min_resource_budget
        self._policy_budget = policy_budget
        self._fixed_hp_config = fixed_hp_config
        self._hp2tune_init_config = hp2tune_init_config
        self._learner_class = AutoVW.learner_class
        self._data_buffer = DataBuffer(visit_limit = self._policy_budget)

        self._learner_dic = {} 
        # the incumbent_learner is the incumbent vw when the AutoVW is called
        # the incumbent_learner is udpated everytime the learn function is called.
        print(self._hp2tune_init_config, self._fixed_hp_config)
        self._incumbent_learner = AutoVW.learner_class(
            **self._fixed_hp_config, **self._hp2tune_init_config)
        # config_id = self._get_config_id(self._hp2tune_init_config['q'])
        # self._learner_dic[config_id] = self._incumbent_learner  
        self._scheduler = ASHAScheduler(
            time_attr='training_sample_size',
            metric='loss_ucb',
            mode='min',
            #TODO: how to specify the max resource in ASHA
            max_t=self._min_resource_budget*(2**6), 
            grace_period= self._min_resource_budget,
            reduction_factor=2,
            brackets=1)
        #TODO: do I need a scheduler or searcher?
        self._searcher = None
        # self._scheduler = OnlineFeatureSelectionScheduler(
        #    time_attr='training_sample_size',
        #     metric='loss_ucb',
        #     mode='min',
        #     min_t=self._min_resource_budget, 
        #     reduction_factor=2,
        #     init_config = hp2tune_init_config,
        #     generate_new_func = self._generate_new_sapce)

    @staticmethod
    def _get_config_id(config_value):
        return tuple(config_value)

    def _trainable_vw(self, config):
        print(config)
        config_id = self._get_config_id(self._hp2tune_init_config['q'])
        if config_id not in self._learner_dic:
            self._learner_dic[config_id] = self._learner_class(config, **self._fixed_hp_config) 
        # vw_learner = pyvw.vw(config, **self._fixed_hp_config)
        data_sample_count = 0
        total_resource_consumed = 0
        const = 0.1
        while True:
            # data = self._data_buffer.feed()
            data = [(1,3), (3,4)]
            if data:
                self._learner_dic[config_id].learn(data)
                data_sample_count += len(data)
                #TODO: where to get c
                c = 1.0
                resource_consumed = len(data)*c
                loss_sum = self._learner_dic[config_id].get_sum_loss

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
                }
                # self.learners[config_id] = vw_learner
            else:
                result = None
            tune.report(**result)
    
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
        #TODO: is this the right way to setup the data buffer?
        #TODO: do we need to decouple x and y as sometimes x and y come at different time
        # self._data_buffer.collect([(x,y)])
        self._data_buffer.collect([x])
        full_search_sapce = self._generate_full_space(list(self._hp2tune_init_config['q']))
        search_space = self._searcher.search_space if (
            self._searcher and self._searcher.search_space) else full_search_sapce
        print(tuple(search_space))
        tune_search_space ={'q': tune.choice(search_space)}
        tune_search_space ={'q': tune.choice(['ab', 'ac'])}
        # tune.utils.diagnose_serialization(self._trainable_vw)
        # tune.utils.diagnose_serialization(partial(AutoVW._trainable_vw, self))
        # tune.utils.diagnose_serialization(partial(trainable_vw_test, self))
        # tune.utils.diagnose_serialization(partial(trainable_vw, AutoVW.learner_class,
        #     self._fixed_hp_config, self._data_buffer, self._incumbent_learner))
       
        analysis = tune.run(
            partial(trainable_vw, AutoVW.learner_class, self._fixed_hp_config, self._data_buffer),
            # self._trainable_vw,
            config = tune_search_space,
            search_alg = self._searcher,
            scheduler = self._scheduler,
            # verbose=0, local_dir='logs/ray_results'
            )
        #TODO: check the API for finding the best trial
        # self._incumbent_learner = analysis.get_best_trial(metric ='loss_ucb').last_result['config_id']
    #TODO: need to wrap all the function calls for AutoVW
    
    def get_sum_loss(self):
        return self._incumbent_learner.get_sum_loss()

    def finished(self):
        return self._incumbent_learner.finished()
    
    @property  
    def incumbent_vw(self):
        return self._incumbent_learner

    @staticmethod
    def _generate_new_sapce(champion_config, order=2):
        assert len(champion_config) == 1
        hp_name = champion_config.keys()[0]
        seed = champion_config[hp_name]
        #TODO: construct the feature map set generators
        import itertools
        return list(itertools.product(* [seed, seed]))
    
    @staticmethod
    def _generate_full_space(org_feature_list, max_poly_degree = 4):
        # feature_list = config['q']
        if not max_poly_degree: max_poly_degree = len(org_feature_list)
        max_poly_comb = list(org_feature_list)
        for i in range(2, max_poly_degree):
            max_poly_comb = list(itertools.product(* [max_poly_comb, org_feature_list]))
        return max_poly_comb

    