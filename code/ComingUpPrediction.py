from Scraper import *

from OutputGenerator import *
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

# Read in the data of the match and of the players
#                             "H_buildUpPlaySpeed, H_buildUpPlayPassing, H_chanceCreationPassing, H_chanceCreationCrossing, H_chanceCreationShooting, H_defencePressure, H_defenceAggression, H_defenceTeamWidth, "
#                             "A_buildUpPlaySpeed, A_buildUpPlayPassing, A_chanceCreationPassing, A_chanceCreationCrossing, A_chanceCreationShooting, A_defencePressure, A_defenceAggression, A_defenceTeamWidth
x2 = read_data("Match_DEF",
               "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
               "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
               "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
               "B365H, B365D, B365A")

# workin bttin sits = "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA")
y = process_output()

x = []
return_multiplier = []
y1 = []

for i in range(21000):
    none_present = False
    for j in range(len(x2[i])):
        if x2[i][j] is None:
            none_present = True
    if not none_present:
        x.append(x2[i])
        y1.append(y[i])
        return_multiplier.append((x2[i][-3:]))
print(len(x))

# x = StandardScaler().fit_transform(x)
# split into training and test
x_train = x
y_train = y1

gradient_boosting_fit = GradientBoostingClassifier().fit(x_train, y_train)

adaboosting_fit = AdaBoostClassifier().fit(x_train, y_train)

logisticreg_fit = LogisticRegression().fit(x_train, y_train)

# new matches calculation
#####

new_matches = coming_matches()
print(new_matches)
new_matches_stats = []
return_multiplier = []
match_names = []
for match in new_matches:
    new_matches_stats.append(match[:-1])
    return_multiplier.append(match[-4:-1])
    match_names.append(match[-1:])

calcodds1 = gradient_boosting_fit.predict_proba(new_matches_stats)
calcodds2 = logisticreg_fit.predict_proba(new_matches_stats)

print(match_names)

print(calcodds2)

for i in range(len(calcodds1)):
    for j in range(len(calcodds1[i])):
        print(new_matches[-1], j)

        if 1 < calcodds2[i][j] * return_multiplier[i][j]:
            print(match_names[i], j)
