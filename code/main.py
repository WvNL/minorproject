from DatabaseExtractor import *
from Profit_optimization import *
from Scraper import *
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
from sklearn.model_selection import KFold


# Read in the data of the match and of the players
#                             "H_buildUpPlaySpeed, H_buildUpPlayPassing, H_chanceCreationPassing, H_chanceCreationCrossing, H_chanceCreationShooting, H_defencePressure, H_defenceAggression, H_defenceTeamWidth, "
#                             "A_buildUpPlaySpeed, A_buildUpPlayPassing, A_chanceCreationPassing, A_chanceCreationCrossing, A_chanceCreationShooting, A_defencePressure, A_defenceAggression, A_defenceTeamWidth
x2 = read_data("Match_DEF", "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
                            "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
                            "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
                            "B365H, B365D, B365A" )

# workin bttin sits = "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA")
y = process_output()

x = []
y1 = []

for i in range(21000):
    none_present = False
    for j in range(len(x2[i])):
        if x2[i][j] is None:
            none_present = True
    if not none_present:
        x.append(x2[i])
        y1.append(y[i])
print(len(x))

total_matches = 0
total_profit = 0


kf = KFold(n_splits=10)

for train_index, test_index in kf.split(x):
    # x = StandardScaler().fit_transform(x)
    # split into training and test
    # x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.15, shuffle=True)

    ## k-fold split into train and test to test the profit over the whole training set
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for index in train_index:
        x_train.append(x[index])
        y_train.append(y1[index])

    for index in test_index:
        x_test.append(x[index])
        y_test.append(y1[index])
    # bayes, seems good, 2nd gives constant and 2nd best result
    # gnb = GaussianNB()
    # gnbfit = gnb.fit(x_train, y_train)
    # mnb = MultinomialNB().fit(x_train, y_train)
    # bnb = BernoulliNB().fit(x_train, y_train)

    #seems good, gives changing results
    #clf = OneVsRestClassifier(LinearSVC(class_weight=class_weight))
    #clf_fit = clf.fit(x_train, y_train).score(x_test, y_test)
    #print("One vs. Rest")
    #print(clf_fit)


    # gradient_boosting_fit = GradientBoostingClassifier().fit(x_train, y_train)
    # print("Gradient Boosting")
    # print(gradient_boosting_fit.score(x_test, y_test))


    # adaboosting_fit = AdaBoostClassifier(algorithm= 'SAMME', base_estimator= RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=None, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=None, verbose=0,warm_start=False), learning_rate= 0.5, n_estimators= 50).fit(x_train, y_train)
    # result1 = adaboosting_fit.score(x_test, y_test)
    # print("Adaboost")
    # print(result1)

    # seems good, gives constant result, best atm
    logisticreg_fit = LogisticRegression().fit(x_train,y_train)
    result2 = logisticreg_fit.score(x_test, y_test)
    print("logisticregr")
    print(result2)

    # # seems good, gives close to constant result
    # clf4 = DecisionTreeClassifier()
    # decisiontree_fit = clf4.fit(x_train, y_train)
    # print("decisiontree")
    # print(decisiontree_fit.score(x_test, y_test))
    # #
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
    ###
    calcodds = logisticreg_fit.predict_proba(x_test[:len(x_test)])
    result = y_test[:2800]

    return_multiplier = []
    for i in range(len(x_test)):
        return_multiplier.append(x_test[i][-3:])



    # profit visualization
    # expected_return_pm(calcodds, return_multiplier, result)
    # expected_return_total(calcodds, return_multiplier, result)
    # multiplier_return_average(calcodds, return_multiplier, result)
    # multiplier_return_total(calcodds, return_multiplier, result)

    profit, matches = profit_calculator(calcodds, return_multiplier, result)
    total_profit += profit
    total_matches += matches
    print(profit)


print(total_profit/total_matches)
