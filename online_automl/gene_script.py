

# -m naiveVW oracleVW fixed notest autocross ours 
# -m naiveVW oracleVW fixed notest
# -m autocross ours 
# python gene_script.py  -i 1000000 -min_resource 50 -policy_budget 5   -filename run_openml.sh
import argparse
from config import STDOUT_DIR, VW_DS_DIR
def get_full_ds_list(DIR):
    import os
    dids = []
    for root, dirs, files in os.walk(DIR):
        for filename in files:
            if 'vw' in filename:
                d_id = str(filename.split('_')[1])
                dids.append(d_id)
    return dids
    
if __name__=='__main__':
    # dataset_list = [189, 198, 215, 218, 227, 287, 344, 562, 572, 688, 1193, 1196, 1199, 1200, 1201, 1204, 1213, 1595, 'simulation']
    dataset_list = [189, 198, 215, 218, 227, 287, 344, 562, 572, 688, 1193, 1196, 1199, 1200, 1201, 1204, 1213, 1595, 'simulation']
    # dataset_list = [189, 215,  572, 1196, 1199, 1200, 1201, 1204,  'simulation']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', metavar='iter', type = int, 
        default=1000, help="iteration number")
    parser.add_argument('-min_resource', '--min_resource', metavar='min_resource', type = int, 
        default=20, help="min_resource")
    parser.add_argument('-policy_budget', '--policy_budget', metavar='policy_budget', type = int, 
        default=5, help="policy_budget")
    # parser.add_argument('-problem', '--problem', dest='problems', nargs='*', default=['simulation'])
    parser.add_argument('-datasets', '--dataset_list', dest='dataset_list', nargs='*' , 
        default= [], help="The dataset list")
    parser.add_argument('-m', '--method_list', dest='method_list', nargs='*' , 
        default= [], help="The method list")
    parser.add_argument('-no_redirect', '--no_redirect', action='store_true',
                        help='whether to redirect std output.')
    parser.add_argument('-rerun', '--force_rerun', action='store_true',
                        help='whether to force rerun.') 
    parser.add_argument('-filename', '--file_name', metavar='file_name',  
        default='run_exp.sh', help="result filename")
    args = parser.parse_args()

    iter_budget = args.iter # iterations, for synthetic study only
    alias_time = '-i ' + str(iter_budget)
    alias_resource = '-min_resource ' + str(args.min_resource)
    alias_policy_num = '-policy_budget ' + str(args.policy_budget)
    additional_argument = ''
    argument_list = []

    # filename = 'run_flaml_bs.sh'
    filename = args.file_name
    f = open(filename,'w')
    # f.write('export SCREENDIR=$HOME/.screen')
    no_redirect = args.no_redirect
    if len(args.dataset_list)==0:
        dataset_list = get_full_ds_list(VW_DS_DIR)
    for d in dataset_list:
        alias_prob =  '-dataset ' + str(d) 
        alias_rerun = '-rerun' if args.force_rerun else ''
        alias_m_list = '-m ' + str(args.method_list) if len(args.method_list)!=0  else ''
        alias_screen =  str(d)+ '_' + str(iter_budget) + '_' +str(args.min_resource)
        argument_list = [alias_time, alias_resource, alias_policy_num, alias_prob, alias_m_list, alias_rerun, additional_argument]
        line_part1 = '\n'+ 'screen -Sdm ' + alias_screen + ' ' + 'bash -c '
        line_part2 = '"' + 'python tester.py ' + ' '.join(argument_list) \
                    + '>' + STDOUT_DIR + 'out_' + alias_screen \
                    + ' ' + '2>' + STDOUT_DIR + 'err_' + alias_screen + '"'
        if not no_redirect:
            line = line_part1 + line_part2 
        else:
            line = '\n'+ 'screen -Sdm ' + alias_screen \
                + ' ' + ' python -m test.test' + ' '.join(argument_list) 
        f.write(line) 
        f.write('\n' + 'sleep 10s')
