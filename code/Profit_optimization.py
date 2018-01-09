import numpy as np
import matplotlib.pyplot as plt


# Average profit per range (so relative profit)
def expected_return_pm(predicted_odds, return_multiplier, result):
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
                if lower_bound < predicted_odds[i][j] * return_multiplier[i][j] < upper_bound:
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

        if match_count > 5:
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
    print(profit_ranges_array)
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


# Total Profit per range
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
                if lower_bound < predicted_odds[i][j] * return_multiplier[i][j] < upper_bound:
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

        if match_count > 5:
            profit_ranges_array.append(round(range_profit, 5))
            y_axis.append(str(round(lower_bound, 3)))
            if ht_count != 0:
                htw_profits.append(round(htw_prof, 5))
            else:
                htw_profits.append(0)
            if d_count != 0:
                d_profits.append(round(d_prof, 5))
            else:
                d_profits.append(0)
            if at_count != 0:
                atw_profits.append(round(atw_prof, 5))
            else:
                atw_profits.append(0)
            lower_bound += 0.05
            upper_bound += 0.05
    print(profit_ranges_array)
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
def multiplier_return(predicted_odds, return_multiplier, result):
    lower_bound = 1
    upper_bound = 1.2
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
                if lower_bound < return_multiplier[i][j] < upper_bound:
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

        if match_count > 5:
            profit_ranges_array.append(round(range_profit, 5))
            y_axis.append(str(round(lower_bound, 3)))
            if ht_count != 0:
                htw_profits.append(round(htw_prof, 5))
            else:
                htw_profits.append(0)
            if d_count != 0:
                d_profits.append(round(d_prof, 5))
            else:
                d_profits.append(0)
            if at_count != 0:
                atw_profits.append(round(atw_prof, 5))
            else:
                atw_profits.append(0)
            lower_bound += 0.2
            upper_bound += 0.2
    print(profit_ranges_array)
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
def multiplier_return(predicted_odds, return_multiplier, result):
    lower_bound = 1
    upper_bound = 1.2
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
                if lower_bound < return_multiplier[i][j] < upper_bound:
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

        if match_count > 5:
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
            lower_bound += 0.2
            upper_bound += 0.2
    print(profit_ranges_array)
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
