from DatabaseExtractor import *
from sklearn.multiclass import OneVsRestClassifier
from OutputGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import *
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Read in the data of the match and of the players
#                             "H_buildUpPlaySpeed, H_buildUpPlayPassing, H_chanceCreationPassing, H_chanceCreationCrossing, H_chanceCreationShooting, H_defencePressure, H_defenceAggression, H_defenceTeamWidth, "
#                             "A_buildUpPlaySpeed, A_buildUpPlayPassing, A_chanceCreationPassing, A_chanceCreationCrossing, A_chanceCreationShooting, A_defencePressure, A_defenceAggression, A_defenceTeamWidth
x2 = read_data("Match_DEF", "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
                            "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
                            "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
                            "B365H, B365D, B365A")

# x2 = read_data("Match_DEF", "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA")
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

#x = StandardScaler().fit_transform(x)
# split into training and test
x_train = x[:16000]
y_train = y1[:16000]
x_test = x[16000:]
y_test = y1[16000:]

# bayes, seems good, 2nd gives constant and 2nd best result
gnb = GaussianNB()
gnbfit = gnb.fit(x_train, y_train)
mnb = MultinomialNB().fit(x_train, y_train)
bnb = BernoulliNB().fit(x_train, y_train)
print("bayes")
print(gnbfit.score(x_test, y_test))
print(mnb.score(x_test, y_test))
print(bnb.score(x_test, y_test))
print(mnb.predict_proba(x_test[:10]))

clf = AdaBoostClassifier().fit(x_train, y_train)
result1 = clf.score(x_test, y_test)
print("some1")
print(result1)

# seems good, gives constant result, best atm
otherclf = LogisticRegression().fit(x_train,y_train)
result2 = otherclf.score(x_test, y_test)
print("logisticregr")
print(result2)

# print(otherclf.get_params(deep=True))
#
# # seems good, gives close to constant result
# clf4 = DecisionTreeClassifier()
# model4 = clf4.fit(x_train, y_train)
# result3 = model4.predict(x_test)
# print("decisiontree")
# print(model4.score(x_test, y_test))
#
# # seems good, gives changing results
# clf3 = svm.LinearSVC()
# clf3p = clf3.fit(x_train, y_train)
# print("linearsvc")
# print(clf3p.score(x_test, y_test))
#
# # seems medium, gives changing results
# mlp = MLPClassifier()
# print("mlp")
# print(mlp.fit(x_train, y_train).score(x_test, y_test))
# mlp = MLPClassifier()
# print("mlp")
# print(mlp.fit(x_train, y_train).score(x_test, y_test))
# mlp = MLPClassifier()
# print("mlp")
# print(mlp.fit(x_train, y_train).score(x_test, y_test))


# weirdly enough, theoretically only these matter:
# "H_rating4, H_rating6, H_rating7, H_rating8, H_rating9,  "
#                             "A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating11, "
#                             "H_chanceCreationCrossing, A_chanceCreationCrossing, A_defenceAggression")

# Algorithm to calculate if there is profit to be made
####
array = otherclf.predict_proba(x_test[:3000])
odds = odds[16000:19000]
result = y_test[:3000]
calcodds = []

for i in range(len(array)):
    calcodds.append(list(array[i]))
matches = 0
profit = 0
wrong = 0
correct = 0
for i in range(len(calcodds)):
    for j in range(len(calcodds[i])):
        if 1.03 < calcodds[i][j]*odds[i][j] < 1.15 and j == 0:
            matches += 1

            if j == result[i]:
                profit += (odds[i][j] - 1)
                correct += 1

            if j != result[i]:
                profit -= 1
                wrong += 1

print(profit, "in ", matches, "matches")
print("profit per match:" + str(profit/matches))
print(wrong, "vs", correct)
####
# Above is basically this
# if odds*bwinodds>100 percent:
#     if correct: result+=1*bwinodds
#     if not correct: result-=1
