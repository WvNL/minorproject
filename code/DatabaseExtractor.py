import sqlite3

# This file contains the method used to read in the data from the SQLite file.

# Read in Data from table, row
def read_data(table, row):
    db = sqlite3.connect('database.sqlite')
    cursor = db.cursor()
    cursor.execute("SELECT " + row + " FROM " + table)

    return cursor.fetchall()


# Reads in Data from table,row, with an id
def read_id(table, row, id):
    db = sqlite3.connect('database.sqlite')
    cursor = db.cursor()
    cursor.execute("SELECT " + row + " FROM " + table + " WHERE player_api_id= " + id + " ")

    return cursor.fetchall()


def stats_from_id(player_IDs, needed_stats, standard_values):
    """
    # Reads in the stats of the players
    player_ID needs to be an list of player_ID's, for example by calling "read_data("Match", "home_player_1")[:1000]"
    needed_stats are the player stats that you want to extract
    standard_values are the standard values that will be used if the ID equals None, and needs to be in brackets []
    """
    api_id = read_data("Player_Attributes", "player_api_id, " + needed_stats)

    p1_ratings = []
    # For every player id
    for item in player_IDs:
        # Parse in some value if None, else parse in the player stat retrieved from Player_attributes
        if item[0] is None:
            p1_ratings.append(standard_values)
        else:
            for player in api_id:
                if item[0] == player[0]:
                    p1_ratings.append(list(player[1:]))

                    break
    return p1_ratings


