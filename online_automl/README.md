# Packages to intall 
pip install vowpalwabbit

pip install ray[tune]

git rm -f AML
git submodule add https://github.com/sonichi/AML.git
(https://github.com/sonichi/AML/tree/async)

pip install openml


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

# TODO
- Some part of the code is in an private repo, need to add Marco such that Macro can add the repo as submodule

- VW name space related changes: 
- Better visualization
- ConfigOracle: in file: AML/BlendSearch/online_searcher.py


```
self._champion_frontier_config_list = self._generate_new_space(self._champion_trial.config, \
                1, order=2) + [self._champion_trial.config] # config_id -> config
```

```
def _generate_new_space(self, champion_config, interaction_num_to_add, order=2) -> list 
```