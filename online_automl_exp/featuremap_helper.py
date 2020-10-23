from sklearn.preprocessing import PolynomialFeatures
import itertools
def convert_tuple_list(tuple_list):
    converted_tuple_list = []
    for element in tuple_list:
        tuple_ = ele_from_nested_tuple(element)
        converted_tuple_list.append(tuple_)
    return converted_tuple_list

def ele_from_nested_tuple(test_tuple): 
    res = tuple() 
    if type(test_tuple) is int: return (test_tuple,)
    else:
        for ele in test_tuple: 
            if isinstance(ele, tuple): 
                res += ele_from_nested_tuple(ele)
            else:
                res += (ele,)
        # print('res', res, len(res))
        return res 

def cob_list_of_tuple(list1, list2):
    # get combinatorial list of tuples
    new_list = []
    for i in list1:
        for j in list2:
            new_tuple = (i,) + (j,)
            #TODO: is there a more efficient way to do this?
            if new_tuple not in new_list and ((j,) + (i,)) not in new_list:
                new_list.append(new_tuple)
    return new_list

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

