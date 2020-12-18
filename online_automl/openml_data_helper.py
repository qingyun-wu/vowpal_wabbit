import logging
RANDOM_SEED = 20201234
from config import VW_DS_DIR, QW_OML_API_KEY


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
        print( 'lssine', len(self.vw_examples))
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


def save_vw_dataset_w_ns(X, y, did, ds_dir, is_regression):
    
    if is_regression:
        fname = 'ds_{}_{}.vw'.format(did, 0)
        print(X.shape[0])
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
    parser.add_argument('-min_did', type=int, default=0, help='minimum dataset id to process')
    parser.add_argument('-max_did', type=int, default=None, help='maximum dataset id to process')
    args = parser.parse_args()
    print(args.min_did, ' to ', args.max_did)

    openml.config.apikey =  QW_OML_API_KEY
    openml.config.set_cache_directory('./data/omlcache/')

    print('loaded openML')
    if not os.path.exists(VW_DS_DIR):
        os.makedirs(VW_DS_DIR)
    did_list_5k_to_10k = [i for i in OPENML_REGRESSION_LIST_inst_larger_than_5k if i not in OPENML_REGRESSION_LIST_inst_larger_than_10k] 
    did_list_10k_to_100k = [i for i in OPENML_REGRESSION_LIST_inst_larger_than_10k if i not in OPENML_REGRESSION_LIST_inst_larger_than_100k] 
    print(len(did_list_5k_to_10k), did_list_5k_to_10k)
    dids = OPENML_REGRESSION_LIST_inst_larger_than_100k + [1595, 218]
    for did in sorted(dids):
        # if did < args.min_did:
        #     continue
        # if args.max_did is not None and did >= args.max_did:
        #     break
        print('processing did', did)
        print('getting data,', did)
        ds = openml.datasets.get_dataset(did)
        data = ds.get_data(target=ds.default_target_attribute, dataset_format='array')
        # try:
        #     print('getting data,', did)
        #     ds = openml.datasets.get_dataset(did)
        #     data = ds.get_data(target=ds.default_target_attribute, dataset_format='array')
        # except:
        #     data = None
        #     print('get data error')
        X, y = data[0], data[1] # return X: pd DataFrame, y: pd series
        if data and isinstance(X, np.ndarray):
            shuffled_X, shuffled_y = shuffle_data(X, y, seed = RANDOM_SEED)
            save_vw_dataset_w_ns(shuffled_X, shuffled_y, did, VW_DS_DIR, is_regression = True)
            # vw_line = load_vw_dataset(did, VW_DS_DIR, is_regression=True)
            # print(len(vw_line), vw_line)
        else:
            print('no data')
        

## command line:
# python data_openml.py  -max_did 219 -min_did 218
# python data_openml.py  -max_did 573 -min_did 572