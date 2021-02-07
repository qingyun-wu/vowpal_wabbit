####ICML submission exp command lines

####ICML visualization command lines
python tester.py -i 100000 -policy_budget 5 -d 201  1191  215  344 537 564 1196 1199 1203 1206 5648 23515 41506 41539 42729 42496  -m Chambent-Hybrid naiveVW fixed-50 fixed-5-  Chambent-Van-lcb-top1 Chambent-Van-lcb-tophalf Chambent-Van-ucb-top1 Chambent-Van-ucb-tophalf Chambent-Van-ucb-champion-tophalf  -log -no_rerun
python tester.py -i 100000 -policy_budget 5 -d 201  1191  -m Chambent-Hybrid naiveVW fixed-50 fixed-5-  Chambent-Van-lcb-top1 Chambent-Van-lcb-tophalf Chambent-Van-ucb-top1 Chambent-Van-ucb-tophalf Chambent-Van-ucb-champion-tophalf  -log -no_rerun
## Chacha
# python tester.py -i 100000 -policy_budget 5 -d 201  1191  215  344 537 564 1196 1199 1203 1206 5648 23515 41506 41539 42729 42496  -m Chambent-Hybrid ChaCha naiveVW fixed-50 fixed-5-  -log -no_rerun
python tester.py -i 100000 -policy_budget 5 -d 201  344 564  -m Chambent-Hybrid ChaCha-CB naiveVW fixed-50 fixed-5-  -log -no_rerun
python tester.py -i 100000 -policy_budget 5 -d  344  -m  ChaCha   -log 
python tester.py -i 100000 -policy_budget 5 -d 201  1191  215  344 537 564 1196 1199 1203 1206 5648 23515 41506 41539 42729 42496  -m Chambent-Hybrid ChaCha-CB naiveVW fixed-50 fixed-5-  -log -no_rerun
python tester.py -i 100000 -policy_budget 5 -d 5648 -m Chambent-Hybrid fixed-50 fixed-5- ChaCha-Org naiveVW  -log -no_rerun

python tester.py -i 100000 -policy_budget 5 -d 201  1191  -m Chambent-Hybrid naiveVW fixed-50 fixed-5-  Chambent-Van-lcb-top1 Chambent-Van-lcb-tophalf Chambent-Van-ucb-top1 Chambent-Van-ucb-tophalf Chambent-Van-ucb-champion-tophalf  -log -no_rerun

python tester.py -i 100000 -policy_budget 5 -d 201  1191  215  344 537 564 1196 1199 1203 1206 5648 23515 41506 41539 42729 42496  -m Chambent-Hybrid ChaCha-CB naiveVW fixed-50 fixed-5-  ChaCha-nochampion -log -no_rerun

python tester.py -i 5000 -policy_budget 5 -d 5648 -m ChaCha-Demo fixed-50 fixed-5- naiveVW -log

# command to show demo
python tester.py -i 5000 -policy_budget 5 -d 5648 -m  naiveVW ChaCha-Demo -log -demo #-rerun 
# command to get performance over time on 5648
python get_plots.py -i 100000 -policy_budget 5 -d 5648 -m naiveVW  fixed-5- fixed-50 Chambent-Hybrid -log -no_rerun
# command to get bar plots 
python get_plots.py -i 100000 -policy_budget 5 -d 201  1191  215  344 537 564 1196 1199 1203 1206 5648 23515 41506 41539 42729 42496 -m Chambent-Hybrid naiveVW fixed-50 fixed-5- ChaCha-Org ChaCha-ucb-top0 ChaCha-nochampion-top0  -log -no_rerun -barplot


# python get_plots.py -i 500000 -policy_budget 5 -d 1206 42729 1203 5648 -m Chambent-Hybrid naiveVW fixed-50 fixed-5- ChaCha-Org ChaCha-ucb-top0 ChaCha-nochampion-top0  -log -no_rerun -barplot
# # python get_plots.py -i 500000 -policy_budget 5 -d 1206 42729 5648  -m Chambent-Hybrid naiveVW fixed-50 fixed-5- ChaCha-Org ChaCha-ucb-top0 ChaCha-nochampion-top0  -log -no_rerun -barplot


python get_plots.py -i 1000000 -policy_budget 5 -d 1206 42729 5648  -m naiveVW fixed-50 fixed-5- ChaCha-Org -log -no_rerun -barplot
python get_plots.py -i 500000 -policy_budget 5 -d 1206 42729 5648  -m naiveVW fixed-50 fixed-5- ChaCha-Org -log -no_rerun -barplot

## directly get figures
python get_figures.py