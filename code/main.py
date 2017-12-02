from DatabaseExtractor import *
from sklearn.multiclass import OneVsRestClassifier
from OutputGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import *


# Read in the data of the match and of the players
#                             "H_buildUpPlaySpeed, H_buildUpPlayPassing, H_chanceCreationPassing, H_chanceCreationCrossing, H_chanceCreationShooting, H_defencePressure, H_defenceAggression, H_defenceTeamWidth, "
#                             "A_buildUpPlaySpeed, A_buildUpPlayPassing, A_chanceCreationPassing, A_chanceCreationCrossing, A_chanceCreationShooting, A_defencePressure, A_defenceAggression, A_defenceTeamWidth
x2 = read_data("Match_DEF", "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
                            "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
                            "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
                            "B365H, B365D, B365A")

#x2 = read_data("Match_DEF", "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA")
y = process_output()

x = []
odds = []
y1 = []

for i in range(21000):
    truth = True
    for j in range(len(x2[i])):
        if x2[i][j] is None:
            truth = False
    if truth:
        x.append(x2[i])
        y1.append(y[i])
        odds.append((x2[i][-3:]))
print(len(x))

# split into training and test
x_train = x[:16000]
y_train = y1[:16000]
x_test = x[16000:]
y_test = y1[16000:]

# bayes, seems good, give 2nd gives constant result
gnb = GaussianNB()
gnbfit = gnb.fit(x_train, y_train)
mnb = MultinomialNB().fit(x_train, y_train)
bnb = BernoulliNB().fit(x_train, y_train)
print("bayes")
print(gnbfit.score(x_test, y_test))
print(mnb.score(x_test, y_test))
print(bnb.score(x_test, y_test))
print(mnb.predict_proba(x_test[:10]))


# seems good, gives changing results
clf = OneVsRestClassifier(LinearSVC())
result1 = clf.fit(x_train, y_train).score(x_test, y_test)
print("ovr")
print(result1)

# seems good, gives constant result, best atm
otherclf= LogisticRegression().fit(x_train,y_train)
result2 = otherclf.score(x_test, y_test)
print("logisticregr")
print(result2)
print(otherclf.predict_proba(x_test[:10]))
print(odds[16000:16010])
print(y_test[:10])


# seems good, gives close to constant result
clf4 = DecisionTreeClassifier()
model4 = clf4.fit(x_train, y_train)
result3 = model4.predict(x_test)
print("decisiontree")
print(model4.score(x_test, y_test))

# seems good, gives changing results
clf3 = svm.LinearSVC()
clf3p = clf3.fit(x_train, y_train)
print("linearsvc")
print(clf3p.score(x_test, y_test))

# seems medium, gives changing results
mlp = MLPClassifier()
print("mlp")
print(mlp.fit(x_train, y_train).score(x_test, y_test))
mlp = MLPClassifier()
print("mlp")
print(mlp.fit(x_train, y_train).score(x_test, y_test))
mlp = MLPClassifier()
print("mlp")
print(mlp.fit(x_train, y_train).score(x_test, y_test))


# weirdly enough, theoretically only these matter:
# "H_rating4, H_rating6, H_rating7, H_rating8, H_rating9,  "
#                             "A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating11, "
#                             "H_chanceCreationCrossing, A_chanceCreationCrossing, A_defenceAggression")
