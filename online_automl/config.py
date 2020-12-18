
VW_DS_DIR = './data/openml_vwdatasets/'
RANDOM_SEED = 20201234
QW_OML_API_KEY = '8c4eebcda506ae1065902c2b224369b9'

## using the following filter: 
# qualities.NumberOfInstances:>5000 qualities.NumberOfFeatures:<15 qualities.NumberOfClasses:0
# total num 54
OPENML_REGRESSION_LIST_inst_larger_than_5k = [572, 545, 1595, 1433, 1414, 1202, 1196, 23515, 23395, 23397, 5889, 
5648, 5587, 41539, 41540, 42225, 42559, 42545, 42496, 42125, 42130, 42677, 42669, 42688, 225, 227, 42207, 42131, 
218, 189, 198, 287, 564, 562, 344, 42721, 42728, 42712, 42713, 42720, 688, 215, 41463, 1213, 1203, 1194, 1201, 1197, 
1193, 1200, 1204, 1188, 1199, 537]

# qualities.NumberOfInstances:>10000 qualities.NumberOfFeatures:<26 qualities.NumberOfClasses:0
# total num: 42
OPENML_REGRESSION_LIST_inst_larger_than_10k = [11595, 1414, 1202, 1196, 23515, 23395, 23397, 5889, 5648, 5587, 41539, 41540, 
42225, 42559, 42496, 42130, 42677, 42669, 42688, 42207, 42131, 218, 564, 344, 42721, 42728, 42712, 42713, 42720, 215, 41463, 
1213, 1203, 1194, 1201, 1197, 1193, 1200, 1204, 1188, 1199, 537]

# qualities.NumberOfInstances:>100000 qualities.NumberOfFeatures:<26 qualities.NumberOfClasses:0
# total num: 21
OPENML_REGRESSION_LIST_inst_larger_than_100k = [1595, 1202, 1196, 23397, 5889, 5648, 5587, 41540, 42559, 42130, 42207, 
42131, 42721, 42728, 42720, 1203, 1194, 1201, 1197, 1204, 1188]