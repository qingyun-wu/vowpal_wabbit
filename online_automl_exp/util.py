
import numpy as np
import matplotlib.pyplot as plt

def squared_error(y, y_pred):
    #use l2 norm as the loss function
    #TODO: may need to generalize this loss function
    # loss = np.linalg.norm(y-y_pred, 2)**2
    loss = (y-y_pred)**2
    # print(y, y_pred)
    return loss

def plot_obj(obj_list, alias='reward', vertical_list=None):
    plt.plot(range(len(obj_list)), obj_list, label = alias)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('cumulative loss', fontsize=14)
    plt.legend()
    if vertical_list:
        for v in vertical_list:
            plt.axvline(x=v)
    # plt.ylim([0,1])
    # fig_name = './results/' + alias + '.pdf'
    # plt.savefig(fig_name)

import re
import pandas as pd
import string

def to_vw_format(line):
    chars = re.escape(string.punctuation)
    res = f'{int(line.y)} |'
    for idx, value in line.drop(['y']).iteritems():
        feature_name = re.sub(r'(['+chars+']|\s)+', '_', idx)
        res += f' {feature_name}:{value}'
    return res

def fm2inter_arg(fm):
    for l in fm:
        if len(l)>1: 
            inter_arg = '--interactions  '
            for jj in l: inter_arg += str(jj)
        else: inter_arg = ''
    return inter_arg