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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp



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


# splits the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.3, shuffle=True )

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


print("bayes")
print(gaussian_naive_bayes_fit.score(x_test, y_test))
print(multi_naive_bayes.score(x_test, y_test))
print(bernoulli_naive_bayes.score(x_test, y_test))
print(multi_naive_bayes.predict_proba(x_test[:10]))


#seems good, gives changing results
#clf = OneVsRestClassifier(LinearSVC(class_weight=class_weight))
#clf_fit = clf.fit(x_train, y_train).score(x_test, y_test)
#print("One vs. Rest")
#print(clf_fit)

#gradient_boosting = GradientBoostingClassifier()
#gradient_boosting_fit = GradientBoostingClassifier.fit(x_train,y_train)
#print("Gradient Boosting")
#print(gradient_boosting_fit.score(x_test, y_test))

# seems good, gives constant result, best atm
# = LogisticRegression()
#logistic_regression_fit = logistic_regression.fit(x_train,y_train)
#print("logisticregr")
#print(logistic_regression_fit.score(x_test, y_test))

# seems good, gives close to constant result
decision_tree = DecisionTreeClassifier(class_weight=class_weight)
decision_tree_fit = decision_tree.fit(x_train, y_train)
decision_tree_prediction = decision_tree.predict(x_test)
print(confusion_matrix(y_test, decision_tree_prediction))
print("decisiontree")
print(decision_tree_fit.score(x_test, y_test))

# seems good, gives changing results
linear_svc = svm.LinearSVC(class_weight=class_weight)
linear_svc_fit = linear_svc.fit(x_train, y_train)
print("linearsvc")
print(linear_svc_fit.score(x_test, y_test))

# seems good
random_forest = RandomForestClassifier(class_weight=class_weight)
random_forest_fit = random_forest.fit(x_train, y_train)
print("randomforest")
print(random_forest_fit.score(x_test, y_test))
print(random_forest.predict(x_test))

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

# Algorithm to calculate if there is profit to be made
####
array = decision_tree.predict_proba(x_test[:3000])
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
