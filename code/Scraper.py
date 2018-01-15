import requests
import bs4
import time


# Function for scraping. Basically one scrape copy-pasted 9 times, because automating it wasn't possible in this case.
def coming_matches():
    all_stats = []
    # home
    stats = []
    url = "https://sofifa.com/team/245?v=18&e=158967&set=true"

    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]

    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/246?v=18&e=158967&set=true"
    away = requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(1.6)
    stats.append(4)
    stats.append(5.25)
    stats.append("Ajax-Feyenoord")

    all_stats.append(stats)

    time.sleep(0.5)
    # Fc utrecht -az
    stats = []
    url = "https://sofifa.com/team/1903"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/1906"
    away = requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))
    time.sleep(0.5)
    # betting site odds

    stats.append(2.51)
    stats.append(3.58)
    stats.append(2.62)
    stats.append("Utrecht-AZ")

    all_stats.append(stats)

    # Roda-Twente
    stats = []
    url = "https://sofifa.com/team/1902"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/1908"
    away = requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(3.16)
    stats.append(3.45)
    stats.append(2.2)
    stats.append("Roda-Twente")

    all_stats.append(stats)
    time.sleep(0.5)
    # Zwolle-NAC
    stats = []
    url = "https://sofifa.com/team/1914"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/1904"
    away = requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(1.6)
    stats.append(4.1)
    stats.append(5.21)
    stats.append("zwolle-nac")

    all_stats.append(stats)
    time.sleep(0.5)
    # ADO-VVV
    stats = []
    url = "https://sofifa.com/team/650"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/100651"
    requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(2.11)
    stats.append(3.44)
    stats.append(3.38)
    stats.append("ADO-Venlo")

    all_stats.append(stats)
    time.sleep(0.5)
    # Vitesse-Heerenveen
    stats = []
    url = "https://sofifa.com/team/1909"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/1913"
    away = requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(1.85)
    stats.append(3.73)
    stats.append(3.96)
    stats.append("Vitesse-heerenveen")

    all_stats.append(stats)
    time.sleep(0.5)
    # sparta-excelsior
    stats = []
    url = "https://sofifa.com/team/100646"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/1971"
    away = requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(2.09)
    stats.append(3.49)
    stats.append(3.41)
    stats.append("Sparta-Excelsior")

    all_stats.append(stats)
    time.sleep(0.5)
    # willem2-Groningen
    stats = []
    url = "https://sofifa.com/team/1907"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/1915"
    requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(2.29)
    stats.append(3.53)
    stats.append(2.93)
    stats.append("Willem-Groningen")

    all_stats.append(stats)
    time.sleep(0.5)
    # Heracles-psv
    stats = []
    url = "https://sofifa.com/team/100634"
    home = requests.get(url)
    soup = bs4.BeautifulSoup(home.text, 'html.parser')
    home_team_attributes = soup.find_all(class_="float-right")

    player_stats = soup.find_all(class_="col-digit col-oa")
    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:11]
    time.sleep(0.5)
    # away
    url = "https://sofifa.com/team/247"
    away = requests.get(url)

    soup = bs4.BeautifulSoup(away.text, 'html.parser')
    away_team_attributes = soup.find_all(class_="float-right")
    player_stats = soup.find_all(class_="col-digit col-oa")

    for player_stat in player_stats:
        stats.append(int(str(player_stat.span).split("label p")[1].split("""">""")[0]))
        stats = stats[:22]

    stats.append(int(str(home_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(home_team_attributes[12]).split("""float-right">""")[1][0:2]))

    stats.append(int(str(away_team_attributes[8]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[11]).split("""float-right">""")[1][0:2]))
    stats.append(int(str(away_team_attributes[12]).split("""float-right">""")[1][0:2]))

    # betting site odds

    stats.append(7.73)
    stats.append(5.26)
    stats.append(1.35)
    stats.append("Heracles-psv")

    all_stats.append(stats)
    return all_stats
