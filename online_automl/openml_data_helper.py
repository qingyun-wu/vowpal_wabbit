import logging
RANDOM_SEED = 20201234
from config import QW_OML_API_KEY
from config import VW_DS_DIR
from config import OPENML_REGRESSION_LIST_inst_larger_than_5k, \
    OPENML_REGRESSION_LIST_inst_larger_than_10k, OPENML_REGRESSION_LIST_inst_larger_than_100k, \
        OPENML_REGRESSION_LIST_larger_than_1k

class OpenML2VWData:
    VW_DS_DIR = VW_DS_DIR
    def __init__(self, did, task_type):
        self._did = did
        self._task_type = task_type
        self._is_regression = False
        self.vw_x_dic_list = []
        self.Y = []
        if 'regression' in self._task_type:
            self._is_regression = True        
        self.vw_examples = self.load_vw_dataset(did, OpenML2VWData.VW_DS_DIR, self._is_regression)
        print( 'number of samples', len(self.vw_examples))
        for i, e in enumerate(self.vw_examples):
            self.Y.append(float(e.split('|')[0]))
        print(i, self.Y[0:5])
        logging.info('y label%s', self.Y[0:5])

    @staticmethod
    def load_vw_dataset(did, ds_dir, is_regression):
        import os
        data_list = []
        if is_regression:
            fname = 'ds_{}_{}.vw'.format(did, 0)
            with open(os.path.join(ds_dir, fname), 'r') as f:
                # vw_content = f.readlines()
                vw_content = f.read().splitlines()
                print(type(vw_content), len(vw_content))
                # data_list.append(vw_content)
        return vw_content

import argparse
# from config import OML_API_KEY
import gzip
import openml
import os
import numpy as np
import string
import pandas as pd
import scipy
ns_list = list(string.ascii_lowercase)
# target # of ns: 10-26.
# TODO: split features into 10-26 ns:(1) look at the prefix (10<# of unique prefix< 26); (2) sequentially.

# convert openml dataset to vw example
def save_vw_dataset_w_ns(X, y, did, ds_dir, is_regression):
    
    if is_regression:
        fname = 'ds_{}_{}.vw'.format(did, 0)
        print('dataset size', X.shape[0])
        print('saving data', did, ds_dir, fname)
        from os import path
        if not path.exists(os.path.join(ds_dir, fname)):
            with open(os.path.join(ds_dir, fname), 'w') as f:
                if isinstance(X, pd.DataFrame):
                    for i in range(len(X)):
                        ns_line =  '{} |{}'.format(str(y[i]), '|'.join('{} {}:{:.6f}'.format(ns_list[j], j, val) for 
                            j, val in enumerate(X.iloc[i].to_list()) ))
                        f.write(ns_line)
                        f.write('\n')
                elif isinstance(X, np.ndarray):
                    for i in range(len(X)):
                        ns_line =  '{} |{}'.format(str(y[i]), '|'.join('{} {}:{:.6f}'.format(ns_list[j], j, val) for 
                                j, val in enumerate(X[i]) ))
                        f.write(ns_line)
                        f.write('\n')
                elif isinstance(X, scipy.sparse.csr_matrix):
                    print('sparce')
                    NotImplementedError
        

def shuffle_data(X, y, seed):
    try:
        n = len(X)
    except:
        n = X.getnnz()
    perm = np.random.RandomState(seed=seed).permutation(n)
    X_shuf = X[perm, :]
    y_shuf = y[perm]
    return X_shuf, y_shuf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openML to vw converter')
    parser.add_argument('-min_sample_size', type=int, default=1000, help='minimum sample size')
    parser.add_argument('-max_sample_size', type=int, default=None, help='maximum sample size')
    args = parser.parse_args()
    openml.config.apikey =  QW_OML_API_KEY
    openml.config.set_cache_directory('./data/omlcache/')

    print('loaded openML')
    if not os.path.exists(VW_DS_DIR): os.makedirs(VW_DS_DIR)
    if args.min_sample_size >=1000 and args.max_sample_size is None:
        dids = OPENML_REGRESSION_LIST_larger_than_1k
    dids = OPENML_REGRESSION_LIST_larger_than_1k
    failed_datasets = []
    for did in sorted(dids):
        print('processing did', did)
        print('getting data,', did)
        try:
            ds = openml.datasets.get_dataset(did)
            data = ds.get_data(target=ds.default_target_attribute, dataset_format='array')
            X, y = data[0], data[1] # return X: pd DataFrame, y: pd series
            if data and isinstance(X, np.ndarray):
                save_vw_dataset_w_ns(X, y, did, VW_DS_DIR, is_regression = True)
            else:
                print('no data')
        except:
            failed_datasets.append(did)
            print('-------------failing to save dataset!!', did)
    print('-----------failed datasets', failed_datasets)
## command line:
# python openml_data_helper.py -min_sample_size 1000
# failed datasets [1414, 5572, 40753, 41463, 42080, 42092, 42125, 42130, 42131, 42160, 42183, 42207, 
# 42208, 42362, 42367, 42464, 42559, 42635, 42672, 42673, 42677, 42688, 42720, 42721, 42726, 42728, 42729, 42731]