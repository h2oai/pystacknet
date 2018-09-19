## About

`pystacknet` is a light python version of [StackNet](https://github.com/kaz-Anova/StackNet) which was originally made in Java.

It supports many of the original features, with some new elements. 


## Installation

```
git clone https://github.com/h2oai/pystacknet
cd pystacknet
python setup.py install
```

## New features

`pystacknet`'s main object is a 2-dimensional list of sklearn type of models. This list defines the StackNet structure. This is the equivalent of [parameters](https://github.com/kaz-Anova/StackNet#parameters-file) in the Java version. A representative example could be:

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

    models=[ 
            ######## First level ########
            [RandomForestClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
             ExtraTreesClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
             GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
             LogisticRegression(random_state=1)
             ],
            ######## Second level ########
            [RandomForestClassifier (n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)]
            ]
```

`pystacknet` is not as strict as in the `Java` version and can allow `Regressors`, `Classifiers` or even `Transformers` at any level of StackNet. In other words the following could work just fine:

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import PCA
    models=[ 
            
            [RandomForestClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
             ExtraTreesRegressor (n_estimators=100, max_depth=5, max_features=0.5, random_state=1),
             GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
             LogisticRegression(random_state=1),
             PCA(n_components=4,random_state=1)
             ],
            
            [RandomForestClassifier (n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)]
            
            
            ]
```

**Note** that not all transformers are meaningful in this context and you should use it at your own risk. 


## Parameters

A typical usage for classification could be : 

```python
from pystacknet.pystacknet import StackNetClassifier

model=StackNetClassifier(models, metric="auc", folds=4,
	restacking=False,use_retraining=True, use_proba=True, 
	random_state=12345,n_jobs=1, verbose=1)

model.fit(x,y)
preds=model.predict_proba(x_test)


```
Where :


Command | Explanation
--- | ---
models  |  List of models. This should be a 2-dimensional list . The first level hould defice the stacking level and each entry is the model. 
metric  | Can be "auc","logloss","accuracy","f1","matthews" or your own custom metric as long as it implements (ytrue,ypred,sample_weight=)
folds   |  This can be either integer to define the number of folds used in `StackNet` or an iterable yielding train/test splits.
restacking   |  True for [restacking](https://github.com/kaz-Anova/StackNet#restacking-mode) else False
use_proba   |  When evaluating the metric, it will use probabilities instead of class predictions if `use_proba==True`
use_retraining   |  If `True` it does one model based on the whole training data in order to score the test data. Otherwise it takes the average of all models used in the folds ( however this takes more memory and there is no guarantee that it will work better.) 
random_state   |  Integer for randomised procedures
n_jobs   |   Number of models to run in parallel. This is independent of any extra threads allocated
 n_jobs   |   Number of models to run in parallel. This is independent of any extra threads allocated from the selected algorithms. e.g. it is possible to run 4 models in parallel where one is a randomforest that runs on 10 threads (it selected).
 verbose   |   Integer value higher than zero to allow printing at the console. 
