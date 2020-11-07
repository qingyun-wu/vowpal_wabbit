class LocalSearch:
    """Class for randomized local search
    """

    def __init__(self, best_learner,):
        self.candiate_dic = {}
        self.best_learner = best_learner
        
    
    def get_best_learner(self, candidate_list):
        for trial in candidate_list:
            if trial.id == self.best_learner_id: 
                y_pred =self.learner_dic[fm_id].predict(x)
