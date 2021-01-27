# Packages to intall 

pip install vowpalwabbit

pip install ray[tune]

git rm -f AML
git submodule add https://github.com/sonichi/AML.git
(https://github.com/sonichi/AML/tree/async)

pip install openml

pip install matplotlib

cd online_automl/
git submodule add https://github.com/sonichi/AML.git
(switch to this branch: https://github.com/sonichi/AML/tree/async)


# Experiment running

## Command lines

Example commmand line to run exp on simluation
``` 
python tester.py -i 2000 -policy_budget 5 -dataset simulation  -min_resource 10   -rerun
```
in which  
```-i 2000 ``` specifies the number of iterations to run. Default value: 2000

``` -policy_budget 5 ``` specifies the number of models to run at each iter. Default value: 5.

``` -dataset simulation ``` specifies the name of the dataset. The dataset can be a synthetic dataset using  `-dataset simulation` or a particular openml dataset id, e.g., ``` -dataset 344``` runs exp on an openml dataset whose dataset id is 344).

```-min_resource 10``` specifies the min resource budget that will be used in Chambent (our method).  Default value: 50.

Other options:

```-m``` specifies a particular method to run, e.g., ``` -m Chambent``` runs our method only. If not specified, it will run all methods. The list of all possible methods ``` -m naiveVW oracleVW fixed-5- Chambent```

```-rerun``` specifies whether to rerun an exp or load results from disk if the exp result already exist. 

```-shuffle```  specifies whether to shuffle data.

```-log``` specifies whether to do log transformation.


## Result saving and visualization
After running the exp, several files will be genearted
- debugging log in .log: saved in ```./result/log/```
- prediction result in .json: save in  ```./result/result_log/```
- plot in .pdf: saved in ```./result/plots/``` which visualize the average loss of all methods over time. The plot is generated based on the results saved in .json file. 

The name of both the .json and .pdf files are generated based on an alias which reflects the id of the exp. When running an exp that has already ran. It will directly load result from existing file unless the ```-rerun``` option is enabled.


#  Result visualization
- Step 1: download the result to  ./result/result_log/ (note: since some of the results files are quite large, I only included 4 datasets as examples, which should be enough to get the scripts running). 

- Step 2: interpretation of the result .json file
1. ‘y_pred’ y prediction
2. ‘y’: ground truth y
3.  ‘loss’ loss field stores the instantaneous loss (mean square error)

So if we want to visualize the running average loss (mean square error), we just need to go through the results and get the running average over the ‘loss’ filed. 


If we want to see other types of loss metric, may need to calculate from ‘y_predict’ and ‘y’

- Step 3: methods to compare with.

File signatures of different methods:  naiveVW     fixed-5     Chambent-Hybrid

Step 4: scripts to parse the names and settings of experiment (based on which corresponding results will be loaded)

This command will generate a figure that shows average loss over time for data set 344

```
python tester.py -i 20000 -log -min_resource 100 -policy_budget 5 -dataset 344  -m fixed-5 Chambent-Hybrid naiveVW  -plot_only

```

