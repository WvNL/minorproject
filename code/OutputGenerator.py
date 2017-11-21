from DatabaseExtractor import read_data


def process_output():
    data = read_data("match", "home_team_goal, away_team_goal")
    new_output = []
    for i in data:
        if i[0] > i[1]:
            new_output.append((1, 0, 0))
        elif i[0] < i[1]:
            new_output.append((0, 0, 1))
        else:
            new_output.append((0, 1, 0))
    return new_output


def process_betting_odds(betting_company):
    data = read_data("match", betting_company)
    output = []

    for i in data:
        output.append((i[0], i[1], i[2]))
    return output
