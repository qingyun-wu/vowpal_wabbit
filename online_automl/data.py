
import numpy as np
import argparse
import itertools
import pandas as pd
from sklearn.metrics import mean_squared_error
from vowpalwabbit import pyvw
import pylibvw

from sklearn.preprocessing import PolynomialFeatures
import itertools
import logging
import random 
from config import VW_DS_DIR
from config import OPENML_REGRESSION_LIST_inst_larger_than_5k, \
    OPENML_REGRESSION_LIST_inst_larger_than_10k, OPENML_REGRESSION_LIST_inst_larger_than_100k
RANDOMSEED = 8888

logger = logging.getLogger(__name__)
class DataSimulator:
    def __init__(self, iter_num, parameter=None):
        self.Y = None 
        self.raw_ns = ['a', 'b', 'c', 'd', 'e']
        #key is namespace id, and value is the dim of the namespace
        self.raw_ns_dic = {'a':3, 'b':3, 'c':3, 'd':3, 'e':3}
        self.ground_truth_ns = ['a', 'b', 'c', 'd', 'e','ab', 'ac', 'cd',]  #'ac', 'cd'
        self._random_feature = np.random.RandomState(RANDOMSEED)
        self._random_param = np.random.RandomState(RANDOMSEED+100)
        self._random_noise = np.random.RandomState(RANDOMSEED+1234)
        self._generate_raw_X(iter_num)
        self._generate_parameter()
        self._generate_reward(self.vw_x_dic_list)
        self._construct_vw_example()

    def _generate_raw_X(self, iter_num):
        self.vw_x_dic_list = []
        for i in range(iter_num):
            vw_x_dic = {}
            count = 0
            for raw_ns, ns_dim in self.raw_ns_dic.items():
                feature = self._random_feature.uniform(0,1,ns_dim)
                # feature = feature/np.linalg.norm(feature)
                vw_x_dic[raw_ns] = feature
                count +=1
            for ns in self.ground_truth_ns:
                if ns not in vw_x_dic and len(ns) >1:
                    inter_feature = vw_x_dic[str(ns)[0]]
                    for i in range(1, len(str(ns))):
                        inter_feature = np.outer(inter_feature, vw_x_dic[str(ns)[i]]).flatten()
                    vw_x_dic[ns] = inter_feature
            self.vw_x_dic_list.append(vw_x_dic)
        
    def _generate_parameter(self):
        #generate reward paramter dictionary
        vw_parameter_dic = {}
        count =0
        for ns_int in self.ground_truth_ns:
            ns_dim = 1
            for ns in ns_int:
                ns_dim *= self.raw_ns_dic[ns]
            parameter = self._random_param.uniform(0,1,ns_dim)
            # parameter = parameter/np.linalg.norm(parameter)
            vw_parameter_dic[ns_int] = parameter

            count +=1
        self.vw_parameter_dic = vw_parameter_dic

    def _generate_reward(self, vw_x_list):
        # input is a dictionary of feature in vw format
        y_list = []
        for vw_x_dic in vw_x_list:
            reward_fs = 0
            for ns, fs in vw_x_dic.items():
                if ns in self.vw_parameter_dic:
                    # print(ns, fs, self.vw_parameter_dic[ns])
                    reward_fs += np.dot(vw_x_dic[ns], self.vw_parameter_dic[ns])
            noise = self._random_noise.normal(0, 0.1, 1)[0]
            r = reward_fs  + noise
            log_r = 1/(1 + np.exp(-r))
            if log_r >0.5:
                label = 1
            else: label = 0
            y_list.append(r)
            # y_list.append(label)
        self.Y = y_list

    def _construct_vw_example(self):
        # construct a list of vw example
        self.vw_examples = []
        for i, x_dic in enumerate(self.vw_x_dic_list):
            raw_vw_example = str(self.Y[i]) + ' '
            for ns, ns_x in x_dic.items():
                raw_vw_example = raw_vw_example + '|' + str(ns) + ' ' + ' '.join([str(s) for s in ns_x]) + ' '
            # pyvw_example = pyvw.example(raw_vw_example)
            # print(raw_vw_example)
            self.vw_examples.append(raw_vw_example)


def get_data(iter_num=None, data_source = 'simulation', vw_format=True, max_ns_num=10, shuffle=False, use_log=True):
    logging.info('generating data')
    #get data from simulation
    vw_examples = None
    if 'simu' in data_source:
        # get simulation data
        data = DataSimulator(iter_num)
    else:
        # get openml data
        from openml_data_helper import OpenML2VWData
        data_id = int(data_source)
        data = OpenML2VWData(data_id, max_ns_num, 'regression') 
    Y = data.Y
    if vw_format: vw_examples = data.vw_examples
    logger.debug('first data %s', vw_examples[0])
    if 'simu' in data_source:
        return vw_examples, Y
    # do data shuffling or log transformation for oml data when needed
    if shuffle:
        random.seed(54321)
        random.shuffle(vw_examples)
    
    # do log transformation
    unique_y = set(Y)
    all_y_positive = all(i>0 for i in unique_y)
    d_list = [1191, 1196, 41539, 42496]
    if (all_y_positive and (max(unique_y)>=100) and use_log) or (int(data_id) in d_list and use_log):
        log_vw_examples = []
        for v in vw_examples:
            org_y = v.split('|')[0]
            y = float(v.split('|')[0])
            if y<=0: y=1.0  #convert nonpositive value to 0
            log_y = np.log(y)
            log_vw = v.replace(org_y + '|', str(log_y) + ' |')
            log_vw_examples.append(log_vw)
        if log_vw_examples: return log_vw_examples, Y
    return vw_examples, Y

