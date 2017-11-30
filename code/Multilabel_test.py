from DatabaseExtractor import *
from OutputGenerator import *

from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

import numpy as np

# Read in the data of the match and of the players
"""
  input will be; for both Home and Away
  -Build-up-speed
  -Chance-creating-passing
  -Chance-creating-crossing
  -Chance-creating-shooting
  -defence-pressure
  -defence-Aggression
  -defence-pressure
  -defence-width
"""

x2 = read_data("Match", "home_player_1")[:5000]
p1_rating = stats_from_id(x2, "overall_rating", [70])

y = process_output()
print(p1_rating)

# split into training and test
x2_train = x2[:4700]
x_train = p1_rating[:4700]
y_train = y[:4700]

x2_test = x2[4700:5000]
x_test = p1_rating[4700:5000]
y_test = y[4700:5000]


forest = RandomForestClassifier(n_estimators=10000, max_features=None, max_depth= None, min_samples_split= 2, min_samples_leaf=80)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=1)
result = multi_target_forest.fit(x_train, y_train).predict(x_test)
print(y_test)
print(result)

