

# python gene_script.py  -i 10000 -min_resource 20 -policy_budget 5   -filename run_openml.sh
# python gene_script.py  -i 10000 -min_resource 10 -policy_budget 5  -filename run_openml_10.sh
# python gene_script.py  -i 10000 -min_resource 10 -policy_budget 5 -inter_order 2 -filename run_openml_10.sh
import argparse

if __name__=='__main__':
    ###************************Datasets for ICLR21**************************************
    ##====================xgb_cat/lgbm datasets (35 in total): =========================
    dataset_list = [189, 198, 215, 218, 227, 287, 344, 562, 572, 688, 1193, 1196, 1199, 1200, 1201, 1204, 1213, 1595, 'simulation']


   
    # dataset_list = ['guillermo', 'volkert', 'MiniBooNE', 'Jannis', 'mfeat', 'jungle', 'jasmine']
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iter', metavar='iter', type = int, 
        default=1000, help="iteration number")
    parser.add_argument('-min_resource', '--min_resource', metavar='min_resource', type = int, 
        default=20, help="min_resource")
    parser.add_argument('-policy_budget', '--policy_budget', metavar='policy_budget', type = int, 
        default=5, help="policy_budget")
    parser.add_argument('-inter_order', '--inter_order', metavar='inter_order', type = int, 
        default=3, help="inter_order")
    # parser.add_argument('-problem', '--problem', dest='problems', nargs='*', default=['simulation'])
    parser.add_argument('-datasets', '--dataset_list', dest='dataset_list', nargs='*' , 
        default= [], help="The dataset list")
    # parser.add_argument('-m', '--method_list', dest='method_list', nargs='*' , 
    #     default= [], help="The method list")
    parser.add_argument('-no_redirect', '--no_redirect', action='store_true',
                        help='whether to redirect std output.')
    parser.add_argument('-filename', '--file_name', metavar='file_name',  
        default='run_exp.sh', help="result filename")
    args = parser.parse_args()

    iter_budget = args.iter # iterations, for synthetic study only
    alias_time = '-i ' + str(iter_budget)
    alias_resource = '-min_resource ' + str(args.min_resource)
    alias_policy_num = '-policy_budget ' + str(args.policy_budget)
    alias_inter_order = '-inter_order ' + str(args.inter_order)
    additional_argument = ''
    argument_list = []

    # filename = 'run_flaml_bs.sh'
    filename = args.file_name
    f = open(filename,'w')
    # f.write('export SCREENDIR=$HOME/.screen')
    no_redirect = args.no_redirect
    for d in dataset_list:
        alias_prob =  '-dataset ' + str(d) 
        alias_screen =  str(d)+ '_' + str(iter_budget) + '_' +str(args.min_resource)
        argument_list = [alias_time, alias_resource, alias_policy_num, alias_inter_order, alias_prob, additional_argument]
        line_part1 = '\n'+ 'screen -Sdm ' + alias_screen + ' ' + 'bash -c '
        line_part2 = '"' + 'python tester.py ' + ' '.join(argument_list) \
                    + '>./stdout/out_' + alias_screen \
                    + ' ' + '2>./stdout/err_' + alias_screen + '"'
        if not no_redirect:
            line = line_part1 + line_part2 
        else:
            line = '\n'+ 'screen -Sdm ' + alias_screen \
                + ' ' + ' python -m test.test' + ' '.join(argument_list) 
        f.write(line) 
        f.write('\n' + 'sleep 10s')
