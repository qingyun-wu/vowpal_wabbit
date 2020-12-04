def _better_champion_found(self, best_learner, seed_learner):
        # print(best_learner.loss_ub, best_learner.loss_lb, seed_learner.loss_lb)
        return best_learner.loss_ub + 0.1*(seed_learner.loss_ub - seed_learner.loss_lb
                ) < seed_learner.loss_lb
    
    def _update_all_learners(self, x, y):
        for _, learner in self.learner_dic.items():
            learner.learn(x, y)

    def _rank_feature_map_set(self,):
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
                    self.learner_dic[b_id] = self._initialize_learner(inter_arg)
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
                self.learner_dic[fm_id] = self._initialize_learner(str(inter_arg))
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
                        self.learner_dic[fm_id]= self._initialize_learner(str(inter_arg))
                        # self.learner_dic[fm_id]= Learner(fm, fm_id, self.loss_func, " --quiet " + str(inter_arg))
                #TODO: need to create learner for all fm_set
            #generate a ranked list of the feature map set (if policy_budget is not None and 
            # is smaller than the total size of the feature map set, it will return the top ranked
            # feature map)
            # print('feature map set', self.feature_map_set)
            self.feature_map_ordered_list = self.fm_generator.rank_feature_map_set(self.feature_map_set, 
                self.policy_budget)
        self.seed_fm_id = self.fm_id_dic[self.seed_fm]