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


from blendsearch import Problem
from sklearn.preprocessing import PolynomialFeatures
import itertools

class FSSProblem(Problem):
    """class for online learning problem
    """

    def __init__(self, input_fs, **param):
        self.incremental = True
        self.name = 'FeatureSpaceSearch'
        super().__init__(**param)
        space = self.search_space(input_fs)
        self.update_search_space_info(space)

    def search_space(self, input_fs, interaction_order=2):
        space = {}
        poly_fs_list = FSSProblem.generate_all_poly(input_fs, interaction_order)
        #TODO: how to construct the search space
        space['fs'] = ConfigSearchInfo(name = 'fs', type = str, lower = 3, 
                init = None, upper = None, log=False, choices = poly_fs_list ) 
        d = len(input_fs)
        interaction_num = d choose order
        K = d*(d-1)

        for i in range(K):
            space[i] = ConfigSearchInfo(name = 'fs', type = str, lower = 3, 
                init = None, upper = None, log=False, choices = poly_fs_list ) 

        combination  2^{all_inter}
        return space
    
    def compute_with_config(self):
        pass 

    @staticmethod
    def generate_all_poly(self, input_fs, max_poly_degree):
        poly_comb_dic = {}
        poly_comb_dic[1] = input_fs
        for i in range(2, max_poly_degree+1):
            # poly_comb_dic[i] = list(itertools.product(* [poly_comb_dic[i-1], poly_comb_dic[1]]))
            poly_comb_dic[i] = self._combine_list_of_tuples(poly_comb_dic[i-1], poly_comb_dic[1])
        all_poly_list = []
        for i in range(1, max_poly_degree+1):
            poly_i_list = list(poly_comb_dic[i])
            all_poly_list.extend(poly_i_list)
        # get all possible interactions up to max_poly_degree 
        all_poly_list = FSSProblem._convert_tuple_list(all_poly_list)
        return all_poly_list
   
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
 
    @staticmethod
    def generate_intreactions(FS, max_poly_inter_order=None):
        if not max_poly_inter_order: max_poly_inter_order = len(FS)
        poly_comb_dic = {}
        poly_comb_dic[1] = FS
        all_poly_interaction_list = list(FS)
        for i in range(2, max_poly_inter_order+1):
            poly_comb_dic[i] = list(itertools.product(* [poly_comb_dic[i-1], poly_comb_dic[1]]))
            # poly_comb_dic[i] = FSSProblem._combine_list_of_tuples(poly_comb_dic[i-1], poly_comb_dic[1])
            all_poly_interaction_list.extend(list(poly_comb_dic[i]))
        # get all possible interactions up to max_poly_inter_order 
        all_poly_interactions = FSSProblem.convert_tuple_list(all_poly_interaction_list)
        return all_poly_interactions

    @staticmethod
    def _convert_tuple_list(tuple_list):
        converted_tuple_list = []
        for element in tuple_list:
            tuple_ = FSSProblem._squeeze_nested_tuple(element)
            converted_tuple_list.append(tuple_)
        return converted_tuple_list

    @staticmethod
    def _squeeze_nested_tuple(test_tuple): 
        res = tuple() 
        if type(test_tuple) is int: return (test_tuple,)
        else:
            for ele in test_tuple: 
                if isinstance(ele, tuple): 
                    res += FSSProblem._squeeze_nested_tuple(ele)
                else:
                    res += (ele,)
            return res 

    @staticmethod
    def _combine_list_of_tuples(list1, list2):
        # get combinatorial list of tuples
        new_list = []
        for i in list1:
            for j in list2:
                new_tuple = (i,) + (j,)
                #TODO: is there a more efficient way to do this?
                if new_tuple not in new_list and ((j,) + (i,)) not in new_list:
                    new_list.append(new_tuple)
        return new_list

