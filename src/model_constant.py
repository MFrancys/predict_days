import numpy as np

from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest

pipeline = Pipeline([
#    ("reduction", PCA(n_components=0.95)),
    ("random_survival_forest", RandomSurvivalForest(random_state=123))
])

gridsearch_parameters = {
#     "reduction__n_components": [2,3,4,5,10],
#     "reduction__kernel": ["linear", "poly", "rbf", "sigmoid", "cosine"],
#     "reduction__degree": [1,2,3,4,5,10],
#     "reduction__gamma": [0.001, 0.01, 0.1, 0.5, 0.7],
    # "cluster__eps": [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
     #"cluster__min_samples": [5, 10, 20, 30, 40, 50, 70, 100],
     #"cluster__metric": ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
     #"cluster__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']
    #"cluster__min_cluster_size": [5, 10, 20]
    #"n_estimators":
    #    [int(x) for x in np.linspace(start=200, stop=2000, num=100)], # Number of trees in random forest
    #"max_features":
    #    ["auto", "sqrt", "log2"], # Number of features to consider at every split
    #"max_depth":
    #    [int(x) for x in np.linspace(10, 110, num=11)], # Maximum number of levels in tree
    #"min_samples_split":
    #    [2, 5, 10], # Minimum number of samples required to split a node
    "min_samples_leaf":
        [1, 2, 4], # Minimum number of samples required at each leaf node
    "bootstrap":
        [True, False] # Method of selecting samples for training each tree
}
