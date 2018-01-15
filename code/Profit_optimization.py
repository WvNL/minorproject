import numpy as np
import matplotlib.pyplot as plt

# This document contains the methods for profit calculation and optimization.

# Method for profit calculation. Pretty straight forward, if the expected return (predicted_odds1[i][j] * return_multiplier[i][j]) is bigger than some percentage (atleast 100%), then we bet on that team/game.
# There are different kinds of requisites possible to add to the criteria for whether or not to bet on a game. In this case we have j==0 and j==2, meaning that we only bet on home(j==0) or away(j==2) team wins,
# not draw(j==1)
def profit_calculator(predicted_odds1, return_multiplier, result):
    matches = 0
    profit = 0
    wrong = 0
    correct = 0
    for i in range(len(predicted_odds1)):
        for j in range(len(predicted_odds1[i])):

            if 1.10 < predicted_odds1[i][j] * return_multiplier[i][j] and (j == 0 or j == 2):
                matches += 1

                if j == result[i]:
                    profit += (return_multiplier[i][j] - 1)
                    correct += 1

                if j != result[i]:
                    profit -= 1
                    wrong += 1

    print(profit, "in ", matches, "matches")
    print("average profit per match:" + str(profit / matches))
    print(correct, "vs", wrong)
    return profit, matches

# The following 4 functions are functions that plot the profit per range. In the first 2 this is the expected return range, one showing the total profit, and the other the relative profit per match.
# In the second 2 this is the return_multiplier range, and again one total and one relative.

# Average profit per expected return range (so relative profit)
def expected_return_pm(predicted_odds, return_multiplier, result):
    lower_bound = 0.7
    upper_bound = 0.75
    profit_ranges_array = []
    y_axis = []
    htw_profits = []
    d_profits = []
    atw_profits = []

    for k in range(40):
        range_profit = 0
        match_count = 0
        htw_prof = 0
        d_prof = 0
        atw_prof = 0
        ht_count = 0
        d_count = 0
        at_count = 0
        for i in range(len(predicted_odds)):
            for j in range(len(predicted_odds[i])):
                if lower_bound <= predicted_odds[i][j] * return_multiplier[i][j] < upper_bound:
                    match_count += 1
                    if j == result[i]:
                        range_profit += (return_multiplier[i][j] - 1)
                        if j == 0:
                            htw_prof += (return_multiplier[i][j] - 1)
                        if j == 1:
                            d_prof += (return_multiplier[i][j] - 1)
                        else:
                            atw_prof += (return_multiplier[i][j] - 1)
                        ht_count += 1
                        d_count += 1
                        at_count += 1

                    if j != result[i]:
                        range_profit -= 1
                        if j == 0:
                            htw_prof -= 1
                        if j == 1:
                            d_prof -= 1
                        else:
                            atw_prof -= 1
                        ht_count += 1
                        d_count += 1
                        at_count += 1

        if match_count > 10:
            profit_ranges_array.append(round(range_profit / match_count, 5))
            y_axis.append(str(round(lower_bound, 3)))
            if ht_count != 0:
                htw_profits.append(round(htw_prof / ht_count, 5))
            else:
                htw_profits.append(0)
            if d_count != 0:
                d_profits.append(round(d_prof / d_count, 5))
            else:
                d_profits.append(0)
            if at_count != 0:
                atw_profits.append(round(atw_prof / at_count, 5))
            else:
                atw_profits.append(0)
        lower_bound += 0.05
        upper_bound += 0.05
    y_pos = np.arange(len(y_axis))
    plt.bar(y_pos, profit_ranges_array, 0.2, align='edge', label='overall')
    plt.bar(y_pos + 0.25, htw_profits, 0.2, align='edge', label='Profit when home team wins')
    plt.bar(y_pos + 0.5, d_profits, 0.2, align='edge', label='Profit when draw')
    plt.bar(y_pos + 0.75, atw_profits, 0.2, align='edge', label='Profit when away team wins')
    plt.ylabel('Profit')
    plt.title(''''Average return per match for every per range''')
    plt.legend()
    plt.xticks(y_pos, y_axis)
    plt.grid()

    plt.show()


# Total Profit per expected return range
def expected_return_total(predicted_odds, return_multiplier, result):
    lower_bound = 0.7
    upper_bound = 0.75
    profit_ranges_array = []
    y_axis = []
    htw_profits = []
    d_profits = []
    atw_profits = []

    for k in range(30):
        range_profit = 0
        match_count = 0
        htw_prof = 0
        d_prof = 0
        atw_prof = 0
        ht_count = 0
        d_count = 0
        at_count = 0
        for i in range(len(predicted_odds)):
            for j in range(len(predicted_odds[i])):
                if lower_bound <= predicted_odds[i][j] * return_multiplier[i][j] < upper_bound:
                    match_count += 1
                    if j == result[i]:
                        range_profit += (return_multiplier[i][j] - 1)
                        if j == 0:
                            htw_prof += (return_multiplier[i][j] - 1)
                        if j == 1:
                            d_prof += (return_multiplier[i][j] - 1)
                        else:
                            atw_prof += (return_multiplier[i][j] - 1)
                        ht_count += 1
                        d_count += 1
                        at_count += 1

                    if j != result[i]:
                        range_profit -= 1
                        if j == 0:
                            htw_prof -= 1
                        if j == 1:
                            d_prof -= 1
                        else:
                            atw_prof -= 1
                        ht_count += 1
                        d_count += 1
                        at_count += 1

        profit_ranges_array.append(round(range_profit, 5))
        y_axis.append(str(round(lower_bound, 3)))
        htw_profits.append(round(htw_prof, 5))
        d_profits.append(round(d_prof, 5))
        atw_profits.append(round(atw_prof, 5))

        lower_bound += 0.05
        upper_bound += 0.05
    y_pos = np.arange(len(y_axis))
    plt.bar(y_pos, profit_ranges_array, 0.2, align='edge', label='overall')
    plt.bar(y_pos + 0.25, htw_profits, 0.2, align='edge', label='Profit when home team wins')
    plt.bar(y_pos + 0.5, d_profits, 0.2, align='edge', label='Profit when draw')
    plt.bar(y_pos + 0.75, atw_profits, 0.2, align='edge', label='Profit when away team wins')
    plt.ylabel('Profit')
    plt.title('Total Profit per range')
    plt.legend()
    plt.xticks(y_pos, y_axis)
    plt.grid()

    plt.show()


# Total profit per multiplier range
def multiplier_return_total(predicted_odds, return_multiplier, result):
    lower_bound = 1
    upper_bound = 1.5
    profit_ranges_array = []
    y_axis = []
    htw_profits = []
    d_profits = []
    atw_profits = []

    for k in range(36):
        range_profit = 0
        match_count = 0
        htw_prof = 0
        d_prof = 0
        atw_prof = 0
        ht_count = 0
        d_count = 0
        at_count = 0
        for i in range(len(predicted_odds)):
            for j in range(len(predicted_odds[i])):
                if lower_bound <= return_multiplier[i][j] < upper_bound and 1 < predicted_odds[i][j] * \
                        return_multiplier[i][j] < 1.2:

                    match_count += 1
                    if j == result[i]:
                        range_profit += (return_multiplier[i][j] - 1)
                        if j == 0:
                            htw_prof += (return_multiplier[i][j] - 1)
                        if j == 1:
                            d_prof += (return_multiplier[i][j] - 1)
                        else:
                            atw_prof += (return_multiplier[i][j] - 1)
                        ht_count += 1
                        d_count += 1
                        at_count += 1

                    if j != result[i]:
                        range_profit -= 1
                        if j == 0:
                            htw_prof -= 1
                        if j == 1:
                            d_prof -= 1
                        else:
                            atw_prof -= 1
                        ht_count += 1
                        d_count += 1
                        at_count += 1

        profit_ranges_array.append(round(range_profit, 5))
        y_axis.append(str(round(lower_bound, 3)))
        htw_profits.append(round(htw_prof, 5))

        d_profits.append(round(d_prof, 5))
        atw_profits.append(round(atw_prof, 5))

        lower_bound += 0.5
        upper_bound += 0.5
    y_pos = np.arange(len(y_axis))
    plt.bar(y_pos, profit_ranges_array, 0.2, align='edge', label='overall')
    plt.bar(y_pos + 0.25, htw_profits, 0.2, align='edge', label='Profit when home team wins')
    plt.bar(y_pos + 0.5, d_profits, 0.2, align='edge', label='Profit when draw')
    plt.bar(y_pos + 0.75, atw_profits, 0.2, align='edge', label='Profit when away team wins')
    plt.ylabel('Profit')
    plt.title('Total return per multiplier range')
    plt.legend()
    plt.xticks(y_pos, y_axis)
    plt.grid()

    plt.show()


# Average return per match per multiplier range
def multiplier_return_average(predicted_odds, return_multiplier, result):
    lower_bound = 1
    upper_bound = 1.2
    profit_ranges_array = []
    y_axis = []
    htw_profits = []
    d_profits = []
    atw_profits = []

    for k in range(24):
        range_profit = 0
        match_count = 0
        htw_prof = 0
        d_prof = 0
        atw_prof = 0
        ht_count = 0
        d_count = 0
        at_count = 0
        if k == 10:
            upper_bound += 0.3
        for i in range(len(predicted_odds)):
            for j in range(len(predicted_odds[i])):
                if lower_bound <= return_multiplier[i][j] < upper_bound and 1 < predicted_odds[i][j] * \
                        return_multiplier[i][j]:
                    match_count += 1
                    if j == result[i]:
                        range_profit += (return_multiplier[i][j] - 1)
                        if j == 0:
                            htw_prof += (return_multiplier[i][j] - 1)
                        if j == 1:
                            d_prof += (return_multiplier[i][j] - 1)
                        else:
                            atw_prof += (return_multiplier[i][j] - 1)
                        ht_count += 1
                        d_count += 1
                        at_count += 1

                    if j != result[i]:
                        range_profit -= 1
                        if j == 0:
                            htw_prof -= 1
                        if j == 1:
                            d_prof -= 1
                        else:
                            atw_prof -= 1
                        ht_count += 1
                        d_count += 1
                        at_count += 1

        if match_count > 10:
            profit_ranges_array.append(round(range_profit / match_count, 5))
            y_axis.append(str(round(lower_bound, 3)))
            if ht_count != 0:
                htw_profits.append(round(htw_prof / ht_count, 5))
            else:
                htw_profits.append(0)
            if d_count != 0:
                d_profits.append(round(d_prof / d_count, 5))
            else:
                d_profits.append(0)
            if at_count != 0:
                atw_profits.append(round(atw_prof / at_count, 5))
            else:
                atw_profits.append(0)
        if lower_bound >= 4:
            lower_bound += 0.5
            upper_bound += 0.5
        else:
            lower_bound += 0.2
            upper_bound += 0.2
    y_pos = np.arange(len(y_axis))
    plt.bar(y_pos, profit_ranges_array, 0.2, align='edge', label='overall')
    plt.bar(y_pos + 0.25, htw_profits, 0.2, align='edge', label='Profit when home team wins')
    plt.bar(y_pos + 0.5, d_profits, 0.2, align='edge', label='Profit when draw')
    plt.bar(y_pos + 0.75, atw_profits, 0.2, align='edge', label='Profit when away team wins')
    plt.ylabel('Profit')
    plt.title('Average return per match per multiplier range')
    plt.legend()
    plt.xticks(y_pos, y_axis)
    plt.grid()

    plt.show()
