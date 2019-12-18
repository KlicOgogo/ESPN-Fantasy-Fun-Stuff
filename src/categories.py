def get_week_results(scoreboard_html_source):
    matchups = scoreboard_html_source.findAll('div', {'Scoreboard__Row'})
    results = []
    for match in matchups:
        opponents = match.findAll('li', 'ScoreboardScoreCell__Item')
        team_names = []
        for opp in opponents:
            team_names.append(opp.findAll('div', {'class': 'ScoreCell__TeamName'})[0].text)

        rows = match.findAll('tr', {'Table2__tr'})
        categories = [header.text for header in rows[0].findAll('th', {'Table2__th'})[1:]]
        first_player_stats = [data.text for data in rows[1].findAll('td', {'Table2__td'})[1:]]
        second_player_stats = [data.text for data in rows[2].findAll('td', {'Table2__td'})[1:]]

        results.append(
            ((team_names[0], [(cat, float(stat)) for cat, stat in zip(categories, first_player_stats)]),
             (team_names[1], [(cat, float(stat)) for cat, stat in zip(categories, second_player_stats)])))
    return results, categories
