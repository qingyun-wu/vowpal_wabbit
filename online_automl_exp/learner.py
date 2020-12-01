from vowpalwabbit import pyvw
import uuid
import numpy as np
import copy
import itertools
import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
from automl.search_space import FSSProblem
from automl.search_policy import LocalSearcher
from ray import tune

class VWLearner(pyvw.vw):
    """class for the online learner
    """

    unchanging_config = {}
    def __init__(self, **param):
        #TODO: initialize a learner based on the input config
        # initialize the total number of evaluation made by the learner
        self._evaluation_count = 0 
        self._sum_loss = 0
        super().__init__(**param)        

class ConfiguredVWLearner(VWLearner):
    """This is a custromized wrapper around the input base_learner.
        Comparing to pyvw.vw, it has additional information including
        - id
        - cb: the confidence bound of the learner average loss
        - loss_ub: the upper bound of the loss
        - loss_lb: the lower bound of the loss
        - cost: accumulated cost consumed
        - feature_map: feature map used, which is a list of namespaces and their interactions
    """
    const = 0.1
    base_learner = VWLearner
    def __init__(self, learner_config, id_ = None, **params):
        self.id = id_
        self.config = learner_config
        self.resource_used = 0
        self._sum_loss = 0
        self._eval_count = 0
        self.avg_loss = 0 # usually use empirical negative loss as the empirical score
        self.avg_loss_cb = 0 # confidence bound of the empirical score estimation
        super().__init__(learner_config)

    def eval_incremental(self, data_X, data_Y):
        #TODO: need to check base_learner.learn() function
        # perform learning and get the resource for the learning step
        self.learner.predict(data_X)

    def learn(self, data_X, data_Y):
        super().learn(x, y)
         # update total resource used
        resource = self.learner.learn(data_X,data_Y)
        # update performance scores
        self._sum_loss += (-self.learner.get_sum_loss())
        self._eval_count += len(data_Y)
        self.resource_used += resource

    def _get_learner_complexity(self):
        # returns the complexity of the configured learner
        # e.g if the input config is the feature space
        # TODO: need to specify the complexity
        # complexity = len(self.config)
        complexity = 1
        return complexity

    def get_avg_loss(self):
        return self._sum_loss/self._eval_count

    def _get_loss_cb(self):
        return ConfiguredLearner.const*np.sqrt(self._get_learner_complexity
            )/np.sqrt(self._eval_count)

    def get_loss_ub(self):
        return self.get_avg_loss() + self._get_loss_cb()

    def get_loss_lb(self):    
        return self.get_avg_loss() - self._get_loss_cb()

    @classmethod
    def generate_id(cls):
        return str(uuid.uuid1().hex)[:8]

class TrainableVWLearner(tune.Trainable):
    
    """
    setup function is invoked once training starts.

    step is invoked multiple times. Each time, the Trainable object 
    executes one logical iteration of training in the tuning process, 
    which may include one or more iterations of actual training.

    cleanup is invoked when training is finished.
    """
    const = 0.1
    trainable_class = pyvw.vw
    min_resource = 100
    def setup(self, **config):
        """setup function is invoked once training starts.
        """
        self.model = TrainableVW.trainable_class(**config)
        self.update_count = 0
        self.datasample_count = 0

        #info about resource consumption
        self.resource_budget = TrainableVW.min_resource
        self.resource_used = 0

        #info about performance
        self._sum_loss = 0
        self._eval_count = 0
        self.avg_loss = 0 # usually use empirical negative loss as the empirical score
        self.avg_loss_cb = 0 # confidence bound of the empirical score estimation

        self.config = config if config else {}
       
    def step(self, data_x, data_y):
        """step is invoked multiple times. Each time, the Trainable object 
        executes one logical iteration of training in the tuning process, which
        may include one or more iterations of actual training.
        """
        self.model.predict(data_x)
        self.model.learn(data_x, data_y)
        #update resource consumption 
        self.resource_used += 1.0
        if self.resource_used >= self.resource_budget:
            self.resource_budget *= 2
        self.update_count +=1.0
        self.datasample_count += len(data_y)
        
        # update performance measurements
        self._sum_loss = self.model.get_sum_loss()
        self._eval_count += len(data_y)
        self.avg_loss = self._sum_loss/self._eval_count

        #TODO: addd the calculation of loss_cb
        self.avg_loss_cb = TrainableVW.const*math.sqrt(1.0/self.datasample_count) 
        self.avg_loss_ucb = self.avg_loss + self.avg_loss_cb
        self.avg_loss_ucb = self.avg_loss + self.avg_loss_cb
        result = {
            'avg_loss': self.avg_loss,
            'avg_loss_cb': self.avg_loss_cb,
            'avg_loss_ucb': self.avg_loss_ucb,
            'avg_loss_lcb': self.avg_loss_lcb,
            'resource_used': self.resource_used,
            'resource_budget': self.resource_budget,
        }
        #in addition to the results, Tuen will automatically log many metrics,
        # such as config, trial_id, timesteps_tota and etc,
        # https://docs.ray.io/en/master/tune/user-guide.html#tune-autofilled-metrics
        return result
    
#TODO: what is the name of the algorithm: how to reflect the feature generation part?
class AutoVWLearner(VWLearner):
    """The AutoOnlineLearner object is auto online learning object.
    It has the same main API with pyvw.vw:
        - predict(example)
        - learn(example)
    
    Parameters
    ---------- 
    fixed_hp_config: the args dic used to build a basic learner
    totune_hp_initial_config: dict
    """
    #min_resource specifies the least amount of resources need to spend on a particular policy
    #if the policy is selected
    min_resource = 20
    #policy budget specifies how many polices can be evaluated at the same time
    policy_budget = 10
    #the base learner class
    learner_class = Learner
    def __init__(self, fixed_hp_config: dict, totune_hp_initial_config: dict, **kwargs):
        from AML.scheduler.online_scheduler import OnlineFeatureSelectionScheduler 
        #maintain a dictionary of learners. key: id, value: learner
        self.learner_dic = {} 
        self.fixed_hp_config = fixed_hp_config
        #currently only handling one type of config
        assert len(totune_hp_initial_config)==1
        self.totune_hp_name = totune_hp_initial_config.keys()[0]
        #seed config is the config based on which we expand the search space
        #TODO: maybe need to combinie seed_config with best_config?
        self.seed_config = totune_hp_initial_config.values()[0]
        self.best_config = copy.copy(self.seed_config)
        self.seed_config_id = 0 
        self.best_config_id = 0
        self.incumbent_trials = [] #store the ids of active learners, the max size of it is b
        self.seed = ('a', 'b', 'c',)
        self.space = { 'q': self.seed}
        self._update_search_space()

        self.scheduler = OnlineFeatureSelectionScheduler(
            config = self.space,
        )

    def _update_search_space(self):
        self.space['q'] = self.generate_new_sapce(self.seed, 2)

    def predict(self, x):
        """ Predict on the example
        Parameters
        ---------
        x: vw_example
        
        Returns
        -------
        predict label of input 
        """
        #TODO: check the API for finding the best trial
        best_trial = self.analysis.get_best_trial(metric ='loss_lcb')
        return best_trial.model.predict(x)

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
        #configure all the learners
        self.analysis = tune.run(TrainableVW, config = self.space, num_samples = 1)
        #update learners
        self._update_all_learners(x, y)
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

    

    @staticmethod
    def generate_new_sapce(seed, order=2):
        #TODO: construct the feature map set generators
        import itertools
        return list(itertools.product(* [seed, seed]))
        
    def _initialize_learner(self, interaction):
        return self.base_learner( q = interaction, **self.fixed_hp_config)

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
