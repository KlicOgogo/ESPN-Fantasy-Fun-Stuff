from selenium import webdriver


def get_league_name(scoreboard_html_source):
    return scoreboard_html_source.findAll('h3')[0].text


def get_places(sorted_scores):
    places = {}
    i = 1
    while i <= len(sorted_scores):
        j = i + 1
        while j <= len(sorted_scores) and sorted_scores[j-1][1] == sorted_scores[i-1][1]:
            j += 1
        place = (i + j - 1) / 2
        for k in range(i, j):
            places[sorted_scores[k-1][0]] = place
        i = j
    return places


def get_week_stats(scoreboard_html_source, league='NHL'):
    matchup_results = []
    matchups = scoreboard_html_source.findAll('div', {'Scoreboard__Row'})
    for matchup in matchups:
        opponents = matchup.findAll('li', 'ScoreboardScoreCell__Item')
        res = []
        for opp in opponents:
            team = opp.findAll('div', {'class': 'ScoreCell__TeamName'})[0].text
            if league == 'NHL':
                score = float(opp.findAll('div', {'class': 'ScoreCell__Score'})[0].text)
            else:
                score_nba = opp.findAll('div', {'class': 'ScoreCell__Score'})[0].text.split('-')
                score = '-'.join(map(lambda x: str(x) if x % 1.0 > 1e-7 else str(int(x)), map(float, score_nba)))
            res.append((team, score))
        matchup_results.append(res)
    return matchup_results


def get_espn_fantasy_hockey_scoreboard_stats(league_id, n_weeks):
    espn_scoreboard_url = 'https://fantasy.espn.com/hockey/league/scoreboard'
    urls = [f'{espn_scoreboard_url}?leagueId={league_id}&matchupPeriodId={i+1}' for i in range(n_weeks)]
    all_matchups = []
    browser = webdriver.Chrome()
    for item in urls:
        browser.get(item)
        time.sleep(8)
        html_soup = BeautifulSoup(browser.page_source)
        all_matchups.append(get_week_stats(html_soup))
    return all_matchups