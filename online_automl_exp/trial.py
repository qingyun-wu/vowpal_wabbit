from vowpalwabbit import pyvw
import uuid

class Learner(pyvw.vw):
    """class for the online learner
    """

    unchanging_config = {}
    def __init__(self, **param):
        #TODO: initialize a learner based on the input config
        # initialize the total number of evaluation made by the learner
        self._evaluation_count = 0 
        self._sum_loss = 0
        super().__init__(**param)        

class ConfiguredLearner(Learner):
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
    base_learner = Learner
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

# class Trial:
#     """A trial object holds the state for one model training run.
#     The concept is similar with the Trial concept in Ray Tune: 
#     https://github.com/ray-project/ray/blob/d9c4dea7cf57f653eb24833aec97e57b5a829a66/python/ray/tune/trial.py
#     Attributes:
#         trainable_name (str): Name of the trainable object to be executed.
#         config (dict): Provided configuration dictionary with evaluated params.
#         trial_id (str): Unique identifier for the trial.
#         resources (Resources): Amount of resources that this trial will use.
#         status (str): One of PENDING, RUNNING, PAUSED, TERMINATED, ERROR/
#     """

#     @classmethod
#     def generate_id(cls):
#         return str(uuid.uuid1().hex)[:8]