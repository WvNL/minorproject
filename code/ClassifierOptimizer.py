from OutputGenerator import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# This file is used for optimizing the parameters for every model used. Running this takes a very long time and there is probably more efficient ways of doing this.

x2 = read_data("Match_DEF",
               "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
               "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
               "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
               "H_buildUpPlaySpeed, H_buildUpPlayDribbling, H_buildUpPlayPassing, H_chanceCreationPassing, H_chanceCreationCrossing, H_chanceCreationShooting, H_defencePressure, H_defenceAggression, H_defenceTeamWidth, "
               "A_buildUpPlaySpeed, A_buildUpPlayDribbling, A_buildUpPlayPassing, A_chanceCreationPassing, A_chanceCreationCrossing, A_chanceCreationShooting, A_defencePressure, A_defenceAggression, A_defenceTeamWidth, "
               "B365H, B365D, B365A")
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
x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.3, shuffle=True)

# analyse the classes distribution
length = len(y1)
count0_perc = (y1.count(0) / length)
count1_perc = (y1.count(1) / length)
count2_perc = (y1.count(2) / length)
class_weight = {0: count0_perc, 1: count1_perc, 2: count2_perc}

# 2nd gives constant and 2nd best result
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes_fit = gaussian_naive_bayes.fit(x_train, y_train)
multi_naive_bayes = MultinomialNB().fit(x_train, y_train)
bernoulli_naive_bayes = BernoulliNB().fit(x_train, y_train)

# Parameters to be tuned
tuned_parameters = [{'n_estimators': [100, 50, 150, 200],
                     # 'penalty':                 ['ls'],
                     # 'hidden_layer_sizes':      [(100,)],
                     # 'activation':              ['relu', 'identity', 'logistic', 'tanh'],
                     # 'solver':                  ['adam', 'lbfgs', 'sgd'],
                     # 'power_t':                 [0.5, 0.3, 0.7],
                     # 'learning_rate_init':      [0.001, 0.01, 0.0001],
                     # 'learning_rate':           ['constant', 'invscaling', 'adaptive'],
                     'learning_rate': [1., 1.5, 2.5, 0.8, 0.5],
                     'algorithm': ['SAMME', 'SAMME.R'],
                     # 'batch_size':              ['auto'],
                     # 'alpha':                   [0.00001, 0.000001, 0.0001],
                     # 'loss':                    ['deviance', 'exponential'],
                     # 'dual':                    [True, False],
                     # 'tol':                     [0.0001, 0.001, 0.00001],
                     # 'C':                       [1.0],
                     # 'multi_class':             ['ovr'],
                     # 'fit_intercept':           [True, False],
                     # 'intercept_scaling':       [1],
                     # 'solver':                  ['liblinear'],
                     # 'shuffle':                 [True, False],
                     # 'max_iter':                [200, 400, 600],
                     # 'criterion':               ['gini'],
                     # 'max_features':            [None],
                     # 'max_depth':               [3, 2, 5],
                     # 'min_samples_split':       [2, 3, 5, 8],
                     # 'min_samples_leaf':        [1, 2, 3, 5],
                     # 'subsample':               [1.0, 0.8, 1.5, 2.5],
                     # 'min_weight_fraction_leaf':[0, 0.1, 0.3, 0.5],
                     # 'max_leaf_nodes':          [None], LinearSVC(),
                     'base_estimator': [RandomForestClassifier(), DecisionTreeClassifier()],
                     # 'min_impurity_decrease':   [0],
                     # 'bootstrap':               [True],
                     # 'oob_score':               [True],
                     # 'n_jobs':                  [1],
                     # 'verbose':                 [5],
                     # 'momentum':                [0.9, 0.7, 0.8],
                     # 'early_stopping':          [False, True],
                     # 'beta_1':                  [0.9, 0.7, 0.8],
                     # 'beta_2':                  [0.999, 0.9, 0.95],
                     # 'epsilon':                 [0.00000001],
                     # 'validation_fraction':     [0.1],
                     # 'nesterovs_momentum':      [True, False],
                     # 'warm_start':              [False],
                     # 'class_weight':            [None, 'balanced'],
                     # 'splitter':                ['best'],
                     # 'presort':                 [False, True]
                     }]

# Classifier optimazation
score = 'precision'
print("# Tuning hyper-parameters for precision")
print()

clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5,
                   scoring='%s_macro' % score)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(x_test)
print(classification_report(y_true, y_pred))
print()
