import numpy as np

from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest

pipeline = Pipeline([
#    ("reduction", PCA(n_components=0.95)),
    ("random_survival_forest", RandomSurvivalForest(random_state=123))
])

GRIDSEARCH_PARAMETERS = {
    #"n_estimators":
    #    [int(x) for x in np.linspace(start=200, stop=2000, num=100)], # Number of trees in random forest
    #"max_features":
    #    ["auto", "sqrt", "log2"], # Number of features to consider at every split
    #"max_depth":
    #    [int(x) for x in np.linspace(10, 110, num=11)], # Maximum number of levels in tree
    "min_samples_split":
        [2, 5, 10], # Minimum number of samples required to split a node
    "min_samples_leaf":
        [1, 2, 4], # Minimum number of samples required at each leaf node
    "bootstrap":
        [True, False] # Method of selecting samples for training each tree
}
