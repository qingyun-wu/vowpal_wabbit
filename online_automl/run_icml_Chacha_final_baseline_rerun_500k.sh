screen -Sdm 1203_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1203  -rerun -m  ChaCha-Final  fixed-50 fixed-5-  naive  -log  >./result/stdout/out_1203_10000_50 2>./result/stdout/err_1203_10000_50"
sleep 10s
screen -Sdm 41539_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 41539  -rerun -m  ChaCha-Final  fixed-50 fixed-5-  naive  -log  >./result/stdout/out_41539_10000_50 2>./result/stdout/err_41539_10000_50"
sleep 10s
screen -Sdm 42496_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 42496  -rerun -m  ChaCha-Final  fixed-50 fixed-5-  naive -log  >./result/stdout/out_42496_10000_50 2>./result/stdout/err_42496_10000_50"
sleep 10s
screen -Sdm 1196_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 1196  -rerun -m  ChaCha-Final  fixed-50 fixed-5-  naive -log  >./result/stdout/out_1196_10000_50 2>./result/stdout/err_1196_10000_50"
sleep 10s
screen -Sdm 1191_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1191  -rerun -m  ChaCha-Final fixed-50 fixed-5-  naive  -log  >./result/stdout/out_1191_10000_50 2>./result/stdout/err_1191_10000_50"
sleep 10s


screen -Sdm 1203_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1203  -rerun -m  ChaCha-Final   fixed-5  -log -seed 2468  >./result/stdout/out_1203_10000_50 2>./result/stdout/err_1203_10000_50"
sleep 10s
screen -Sdm 41539_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 41539  -rerun -m  ChaCha-Final   fixed-5  -log -seed 2468  >./result/stdout/out_41539_10000_50 2>./result/stdout/err_41539_10000_50"
sleep 10s
screen -Sdm 42496_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 42496  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 2468  >./result/stdout/out_42496_10000_50 2>./result/stdout/err_42496_10000_50"
sleep 10s
screen -Sdm 1196_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 1196  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 2468  >./result/stdout/out_1196_10000_50 2>./result/stdout/err_1196_10000_50"
sleep 10s
screen -Sdm 1191_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1191  -rerun -m  ChaCha-Final  fixed-5-    -log -seed 2468 >./result/stdout/out_1191_10000_50 2>./result/stdout/err_1191_10000_50"
sleep 10s

screen -Sdm 1203_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1203  -rerun -m  ChaCha-Final   fixed-5  -log -seed 4567  >./result/stdout/out_1203_10000_50 2>./result/stdout/err_1203_10000_50"
sleep 10s
screen -Sdm 41539_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 41539  -rerun -m  ChaCha-Final   fixed-5  -log -seed 4567  >./result/stdout/out_41539_10000_50 2>./result/stdout/err_41539_10000_50"
sleep 10s
screen -Sdm 42496_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 42496  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 4567  >./result/stdout/out_42496_10000_50 2>./result/stdout/err_42496_10000_50"
sleep 10s
screen -Sdm 1196_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 1196  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 4567  >./result/stdout/out_1196_10000_50 2>./result/stdout/err_1196_10000_50"
sleep 10s

screen -Sdm 1191_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1191  -rerun -m  ChaCha-Final  fixed-5-    -log -seed 4567 >./result/stdout/out_1191_10000_50 2>./result/stdout/err_1191_10000_50"
sleep 10s


screen -Sdm 1203_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1203  -rerun -m  ChaCha-Final   fixed-5  -log -seed 9999  >./result/stdout/out_1203_10000_50 2>./result/stdout/err_1203_10000_50"
sleep 10s
screen -Sdm 41539_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 41539  -rerun -m  ChaCha-Final   fixed-5  -log -seed 9999  >./result/stdout/out_41539_10000_50 2>./result/stdout/err_41539_10000_50"
sleep 10s
screen -Sdm 42496_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 42496  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 9999  >./result/stdout/out_42496_10000_50 2>./result/stdout/err_42496_10000_50"
sleep 10s
screen -Sdm 1196_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 1196  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 9999  >./result/stdout/out_1196_10000_50 2>./result/stdout/err_1196_10000_50"
sleep 10s
screen -Sdm 1191_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1191  -rerun -m  ChaCha-Final  fixed-5-    -log -seed 9999  >./result/stdout/out_1191_10000_50 2>./result/stdout/err_1191_10000_50"
sleep 10s

screen -Sdm 1203_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1203  -rerun -m  ChaCha-Final   fixed-5  -log -seed 8666  >./result/stdout/out_1203_10000_50 2>./result/stdout/err_1203_10000_50"
sleep 10s
screen -Sdm 41539_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 41539  -rerun -m  ChaCha-Final   fixed-5  -log -seed 8666  >./result/stdout/out_41539_10000_50 2>./result/stdout/err_41539_10000_50"
sleep 10s
screen -Sdm 42496_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 42496  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 8666  >./result/stdout/out_42496_10000_50 2>./result/stdout/err_42496_10000_50"
sleep 10s
screen -Sdm 1196_10000_50 bash -c "python tester.py -i 100000 -policy_budget 5 -d 1196  -rerun -m  ChaCha-Final   fixed-5-  -log -seed 8666  >./result/stdout/out_1196_10000_50 2>./result/stdout/err_1196_10000_50"
sleep 10s
screen -Sdm 1191_10000_50 bash -c "python tester.py -i 100000  -policy_budget 5 -d 1191  -rerun -m  ChaCha-Final  fixed-5-    -log -seed 8666   >./result/stdout/out_1191_10000_50 2>./result/stdout/err_1191_10000_50"
sleep 10s


