from OutputGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# This file is basically a copy of the main file, but with the class weight added into the prediction algorithms. This file is basically for comparison with the main file. See main file for further explanation.

x2 = read_data("Match_DEF", "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
                            "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
                            "H_buildUpPlaySpeed, H_buildUpPlayPassing, H_chanceCreationPassing, H_chanceCreationCrossing, H_chanceCreationShooting, H_defencePressure, H_defenceAggression, H_defenceTeamWidth, "
                            "A_buildUpPlaySpeed, A_buildUpPlayPassing, A_chanceCreationPassing, A_chanceCreationCrossing, A_chanceCreationShooting, A_defencePressure, A_defenceAggression, A_defenceTeamWidth, "
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
print(x)

# splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, shuffle=True )

#analyse the classes distribution
length = len(y1)
count0_perc = (y1.count(0)/length)
count1_perc = (y1.count(1)/length)
count2_perc = (y1.count(2)/length)
class_weight = {0:count0_perc,1:count1_perc,2:count2_perc}


#2nd gives constant and 2nd best result
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes_fit = gaussian_naive_bayes.fit(x_train, y_train)
multi_naive_bayes = MultinomialNB().fit(x_train, y_train)
bernoulli_naive_bayes = BernoulliNB().fit(x_train, y_train)


print("Bayes")
print(gaussian_naive_bayes_fit.score(x_test, y_test))
print(multi_naive_bayes.score(x_test, y_test))
print(bernoulli_naive_bayes.score(x_test, y_test))


gradient_boosting_fit = GradientBoostingClassifier().fit(x_train,y_train)
print("Gradient Boosting")
print(gradient_boosting_fit.score(x_test, y_test))

ada_classifier = AdaBoostClassifier().fit(x_train, y_train)
ada_classifier_score = ada_classifier.score(x_test, y_test)
print("AdaBoost")
print(ada_classifier_score)

# seems good, gives constant result, best atm
logistic_regression = LogisticRegression(class_weight=class_weight)
logistic_regression_fit = logistic_regression.fit(x_train,y_train)
print("Logistic Regression")
print(logistic_regression_fit.score(x_test, y_test))

decision_tree = DecisionTreeClassifier(class_weight=class_weight)
decision_tree_fit = decision_tree.fit(x_train, y_train)
decision_tree_prediction = decision_tree.predict(x_test)
print(confusion_matrix(y_test, decision_tree_prediction))
print("Decision Tree")
print(decision_tree_fit.score(x_test, y_test))

# seems good, gives changing results
linear_svc = svm.LinearSVC(class_weight=class_weight)
linear_svc_fit = linear_svc.fit(x_train, y_train)
print("Linear SVC")
print(linear_svc_fit.score(x_test, y_test))

# seems good
random_forest = RandomForestClassifier(class_weight=class_weight)
random_forest_fit = random_forest.fit(x_train, y_train)
print("Random Forest")
print(random_forest_fit.score(x_test, y_test))
print(random_forest.predict(x_test))

# seems medium, gives changing results
mlp = MLPClassifier()
print("mlp")
print(mlp.fit(x_train, y_train).score(x_test, y_test))
mlp = MLPClassifier()
print(mlp.fit(x_train, y_train).score(x_test, y_test))
mlp = MLPClassifier()
print(mlp.fit(x_train, y_train).score(x_test, y_test))


calcodds = ada_classifier.predict_proba(x_test[:3000])
odds = []
for i in range(3000):
    odds.append(x_test[i][-3:])
result = y_test[:3000]

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

