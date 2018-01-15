from __future__ import print_function, division
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from OutputGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split

# This file was used to find the optimal inputs for each model. The first part is the same as the main file, after that some code is ran to analyze which inputs to select.

x2 = read_data("Match_DEF", "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
                            "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
                            "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
                            "H_buildUpPlaySpeed, H_buildUpPlayPassing, H_chanceCreationPassing, H_chanceCreationCrossing, H_chanceCreationShooting, H_defencePressure, H_defenceAggression, H_defenceTeamWidth, "
                            "A_buildUpPlaySpeed, A_buildUpPlayPassing, A_chanceCreationPassing, A_chanceCreationCrossing, A_chanceCreationShooting, A_defencePressure, A_defenceAggression, A_defenceTeamWidth, "
                            "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA" )
y = process_output()
x = []
odds = []
y1 = []

for i in range(21000):
    none_present = False
    for j in range(len(x2[i])):
        if x2[i][j] is None:
            none_present = True
    if not none_present:
        x.append(x2[i])
        y1.append(y[i])
        odds.append((x2[i][-3:]))
print(len(x))

# splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.3, shuffle=True )

#analyse the classes distribution
length = len(y1)
count0_perc = (y1.count(0)/length)
count1_perc = (y1.count(1)/length)
count2_perc = (y1.count(2)/length)
class_weight = {0:count0_perc,1:count1_perc,2:count2_perc}

svm = LogisticRegression()
# create the RFE model for the svm classifier and select attributes
rfe = RFECV(svm, 1)
rfe = rfe.fit(x_test, y_test)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)

# Create the RFE object and compute a cross-validated score.
svc = LogisticRegression()
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=svc, step=1,
              scoring='accuracy')
rfecv.fit(x, y1)

print("Optimal number of features : %d" % rfecv.n_features_)
print(rfecv.support_)
print( rfecv.ranking_)
# Plot number of features vs cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

