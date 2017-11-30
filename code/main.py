from DatabaseExtractor import *
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from OutputGenerator import *
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

y = process_output()
# Read in the data of the match and of the players
x2 = read_data("Match", "home_player_1")[:1000]
<<<<<<< HEAD
p1_rating = stats_from_id(x2, "overall_rating", [100])

print(p1_rating)
=======
x3 = read_data("Match", "home_player_2")[:1000]
x4 = read_data("Match", "home_player_3")[:1000]
x5 = read_data("Match", "home_player_4")[:1000]
x6 = read_data("Match", "home_player_5")[:1000]
x7 = read_data("Match", "home_player_6")[:1000]
x8 = read_data("Match", "home_player_7")[:1000]
x9 = read_data("Match", "home_player_8")[:1000]
x10 = read_data("Match", "home_player_9")[:1000]
x11 = read_data("Match", "home_player_10")[:1000]
x12 = read_data("Match", "home_player_11")[:1000]

p1_rating = stats_from_id(x2, "overall_rating", [67])
p2_rating = stats_from_id(x3, "overall_rating", [70])
p3_rating = stats_from_id(x4, "overall_rating", [70])
p4_rating = stats_from_id(x5, "overall_rating", [70])
p5_rating = stats_from_id(x6, "overall_rating", [70])
p6_rating = stats_from_id(x7, "overall_rating", [70])
p7_rating = stats_from_id(x8, "overall_rating", [70])
p8_rating = stats_from_id(x9, "overall_rating", [70])
p9_rating = stats_from_id(x10, "overall_rating", [70])
p10_rating = stats_from_id(x11, "overall_rating", [70])
p11_rating = stats_from_id(x12, "overall_rating", [70])


rating=[]
for i in range(len(x2)):
    rating.append([p1_rating[i][0], p2_rating[i][0], p3_rating[i][0], p4_rating[i][0], p5_rating[i][0], p6_rating[i][0], p7_rating[i][0], p8_rating[i][0], p9_rating[i][0], p10_rating[i][0], p11_rating[i][0]])

print(rating)
y = process_output()
>>>>>>> c0761ef1630450fa89e04affda211feef3488a50

# split into training and test
x_train = rating[:800]
y_train = y[:800]
x_test = rating[800:1000]
y_test = y[800:1000]

<<<<<<< HEAD
# Apply linear regression models [MODELS ARE VERY BAD ATM]
#algorithm_parameters = "solver='lbfgs', multi_class='multinomial'"
#clf = OneVsRestClassifier(LinearSVC())
#result1=clf.fit(x_train, y_train).predict(x_test)

#otherclf= LogisticRegression()
#result2 = otherclf.fit(x_train,y_train).predict(x_test)
=======
# # Apply linear regression models [MODELS ARE VERY BAD ATM]
# algorithm_parameters = "solver='lbfgs', multi_class='multinomial'"
# clf = OneVsRestClassifier(LinearSVC())
# result1 = clf.fit(x_train, y_train).predict(x_test)
#
# otherclf= LogisticRegression()
# result2 = otherclf.fit(x_train,y_train).predict(x_test)

clf4 = DecisionTreeClassifier()
result3 = clf4.fit(x_train, y_train).predict(x_test)
print(result3)
# print(y_test)
>>>>>>> c0761ef1630450fa89e04affda211feef3488a50

# print(x_train)
# clf3 = svm.LinearSVC(multi_class="crammer_singer")
# clf3p = clf3.fit(x_train, y_train)
# print(clf3p.predict(x_test))
# print(x_test[-7:])
<<<<<<< HEAD
#print(result1)
=======
# print(result1)
print(y_test)
>>>>>>> c0761ef1630450fa89e04affda211feef3488a50
#print(result2)

# mlp = MLPClassifier(solver="lbfgs")
# print(mlp.fit(x_train, y_train).predict(x_test))
