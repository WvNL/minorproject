from Profit_optimization import *
from sklearn.multiclass import OneVsRestClassifier
from OutputGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold

# Main file. From here the profit of a model over the whole data set is calculated.

# Read in the data, and process the results
x2 = read_data("Match_DEF",
               "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
               "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
               "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
               "B365H, B365D, B365A")
y = process_output()

x = []
y1 = []

# Filter out all the matches that are None in the data.
for i in range(21000):
    none_present = False
    for j in range(len(x2[i])):
        if x2[i][j] is None:
            none_present = True
    if not none_present:
        x.append(x2[i])
        y1.append(y[i])
print(len(x))

# initialize the total match count, the total profit count and the stats for plotting the profit optimizations plots
total_matches = 0
total_profit = 0
all_calcodds = []
all_returnmultiplier = []
all_results = []

# k-fold split into train and test to test the profit over the whole data set
kf = KFold(n_splits=10)

# run over the whole data set
for train_index, test_index in kf.split(x):

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # Split into train and test for this fold of the k-fold cross validation
    for index in train_index:
        x_train.append(x[index])
        y_train.append(y1[index])

    for index in test_index:
        x_test.append(x[index])
        y_test.append(y1[index])

    # Most of the models we have tried (some have been removed again because they didn't give a decent output:

    # bayes
    gnb = GaussianNB()
    gnbfit = gnb.fit(x_train, y_train)
    mnb = MultinomialNB().fit(x_train, y_train)
    bnb = BernoulliNB().fit(x_train, y_train)

    # OVR
    clf = OneVsRestClassifier(LinearSVC())
    clf_fit = clf.fit(x_train, y_train).score(x_test, y_test)
    print("One vs. Rest")
    print(clf_fit)

    # Gradientboosting
    gradient_boosting_fit = GradientBoostingClassifier().fit(x_train, y_train)
    print("Gradient Boosting")
    print(gradient_boosting_fit.score(x_test, y_test))

    # Adaboost
    adaboosting_fit = AdaBoostClassifier().fit(x_train, y_train)
    result1 = adaboosting_fit.score(x_test, y_test)
    print("Adaboost")
    print(result1)

    # Logistic regression
    logisticreg_fit = LogisticRegression().fit(x_train, y_train)
    result2 = logisticreg_fit.score(x_test, y_test)
    print("logisticregr")
    print(result2)

    # DecisionTree
    decisiontree_fit = DecisionTreeClassifier().fit(x_train, y_train)
    print("decisiontree")
    print(decisiontree_fit.score(x_test, y_test))

    # Random forest
    randomforest_fit = RandomForestClassifier().fit(x_train, y_train)
    print("Random forest")
    print(randomforest_fit.score(x_test, y_test))

    # MLP
    mlp = MLPClassifier()
    print("mlp")
    print(mlp.fit(x_train, y_train).score(x_test, y_test))

    # Predict the test set with a model of choice.
    calcodds = logisticreg_fit.predict_proba(x_test[:len(x_test)])
    result = y_test

    # Put the betting return's into a list of return multipliers.
    return_multiplier = []
    for i in range(len(x_test)):
        return_multiplier.append(x_test[i][-3:])

    # Calculate the profit of this k-cross fold and add them to the totals
    profit, matches = profit_calculator(calcodds, return_multiplier, result)
    total_profit += profit
    total_matches += matches
    print(profit)
    all_calcodds.extend(calcodds)
    all_returnmultiplier.extend(return_multiplier)
    all_results.extend(result)

# profit visualizations of the total profit
expected_return_pm(all_calcodds, all_returnmultiplier, all_results)
expected_return_total(all_calcodds, all_returnmultiplier, all_results)
multiplier_return_average(all_calcodds, all_returnmultiplier, all_results)
multiplier_return_total(all_calcodds, all_returnmultiplier, all_results)
print("total profit", total_profit / total_matches)
