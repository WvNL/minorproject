from Scraper import *
from OutputGenerator import *
from sklearn.linear_model import LogisticRegression

# This file predicts which future matches to bet on, using the only statistic model we could manage a profit with, logistic regression. The upper part is the same as the main file.
x2 = read_data("Match_DEF",
               "H_rating1, H_rating2, H_rating3, H_rating4, H_rating5, H_rating6, H_rating7, H_rating8, H_rating9, H_rating10, H_rating11, "
               "A_rating1, A_rating2, A_rating3, A_rating4, A_rating5, A_rating6, A_rating7, A_rating8, A_rating9, A_rating10, A_rating11, "
               "H_chanceCreationCrossing, H_defencePressure, H_defenceAggression, A_chanceCreationCrossing, A_defencePressure, A_defenceAggression, "
               "B365H, B365D, B365A")

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

# Train on the whole dataset
x_train = x
y_train = y1
logisticreg_fit = LogisticRegression().fit(x_train, y_train)

# Load in the data of the upcoming matches from the scraper and devide it accordingly over arrays
new_matches = coming_matches()
print(new_matches)
new_matches_stats = []
return_multiplier = []
match_names = []
for match in new_matches:
    new_matches_stats.append(match[:-1])
    return_multiplier.append(match[-4:-1])
    match_names.append(match[-1:])

# Predict the odds
calcodds = logisticreg_fit.predict_proba(new_matches_stats)

print(match_names)
print(calcodds)

# Print which matches and teams to bet on.
for i in range(len(calcodds)):
    for j in range(len(calcodds[i])):
        if 1.05 < calcodds[i][j] * return_multiplier[i][j]:
            print(match_names[i], j)
