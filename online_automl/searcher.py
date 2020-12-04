def config2id(config):
    config_id = tuple(config)
    return config_id

class ConfigProfile:

    def __init__(self, config, config_id):
        
        self._config = config
        self._id = config_id
        self._score = None
        self._resource_used = 0
        self._loss = 0

    def update(self, result):
        new_score = result['loss_ucb']
        self._score = new_score

    
    @property
    def score(self):
        return self._score
    
    @property
    def config(self):
        return self._config
    
    @property
    def id(self):
        return self._id


class FeatureSearcher:

    def __init__(self, 
            running_budget: int = None,
            min_resource: int = None,
            init_config: dict = None,
            metric: Optional[str] = None,
            mode: Optional[str] = None,
            ):

        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."
        else: mode = "min"

        self.champion = init_config
        space = self.generator(self.champion['q'])
        self._config_space = {}
        for config in space: 
            config_id = config_id(config)
            self._config_space[config_id] = ConfigProfile(config, config_id)
        self._running_configs = []
        NotImplementedError

    def set_search_properties(self, metric: Optional[str], mode: Optional[str],
                              config: Dict) -> bool:
        NotImplementedError
    def suggest(self, trial_id: str) -> Optional[Dict]:
        NotImplementedError

    def on_trial_result(self, trial_id: str, result: Dict):
        NotImplementedError

    def on_trial_complete(self,
                          trial_id: str,
                          result: Optional[Dict] = None,
                          error: bool = False):
        NotImplementedError
    def choose_trial_to_run(self):
        resource_budget = sorted([c.resource_budget for c in self.config_space])
        suggested = resource_budget[0]
        return suggested

    def next_trial(self) -> Trial:
        NotImplementedError
        return trial 

    def update(self, result: dict):
        #update the following
        for c in self.config_space:
            c.update(result)
        better, new_champion = self._champion_test(self.champion, self.config_space)
        if better: self.champion = new_champion

    def _champion_test(self, champion, candidates):
        for c in candidates:
            if c.test_score > champion.test_score:
                return True, c        
        return False, champion
    
    @staticmethod
    def generator(champion):
        return {champion}

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
    



from ray.tune.schedulers.trial_scheduler import FIFOScheduler
from ray.tune.trial import Trial


class OnlineDoublingScheduler(FIFOScheduler):

    # TODO: check the input info
    # TODO: check what is needed in the constructor
    def __init__(self,
                resource_used_attr: str = "resource_used",
                resource_budget_attr: str = "resource_budget",
                reward_attr: str = None,
                 metric: str = None,
                 mode: str = None,
                 max_t: int = 900,
                 doubling_factor: float = 2):
        
        self._all_trial_added = []
        self._live_trial = set()
        self._doubling_factor = doubling_factor
        self._resource_used_attr = resource_used_attr
        self._resource_budget_attr = resource_budget_attr

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial, result: Dict):
        
        self._live_trial.add(trial)

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        self._all_trial_added.append(trial)

    def choose_trial_to_run(self, trial_runner: "trial_runner.TrialRunner"):
        if len(self._live_trial) > 0:
            for trial in self._live_trial:
                    if trial.result[self._resource_used_attr
                        ] >= trial.result[self._resource_budget_attr]:
                        trial.result[self._resource_budget_attr] *= self._doubling_factor
                        trial = Trial.PAUSED
                        self._live_trial.remove(trial)

        if len(self._live_trial) < trial_runner.running_budget():
            # if there are free slots, use the top priorized in the queue
            i = 1
            trial = self._all_trial_added[-i]
            while self._all_trial_added[-i].status == Trial.RUNNING:
                i +=1
                trial = self._all_trial_added[-i]
            trial.status = Trial.RUNNING
            return trial
