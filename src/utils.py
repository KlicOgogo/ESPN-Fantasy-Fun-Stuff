from selenium.webdriver import Chrome


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


def get_scoreboard_stats(league_id, n_weeks, type='hockey', scoring='points'):
    espn_scoreboard_url = f'https://fantasy.espn.com/{type}/league/scoreboard'
    urls = [f'{espn_scoreboard_url}?leagueId={league_id}&matchupPeriodId={i+1}' for i in range(n_weeks)]
    all_matchups = []
    BROWSER = Chrome()
    for u in urls:
        browser.get(u)
        time.sleep(8)
        html_soup = BeautifulSoup(BROWSER.page_source)
        all_matchups.append(get_week_scores(html_soup, scoring))
    return all_matchups


def get_week_scores(scoreboard_html_source, scoring='points'):
    if scoring not in ['points', 'categories']:
        raise Exception('Wrong scoring parameter!')
    matchups = []
    matchups_html = scoreboard_html_source.findAll('div', {'Scoreboard__Row'})
    for m in matchups_html:
        opponents = m.findAll('li', 'ScoreboardScoreCell__Item')
        res = []
        for o in opponents:
            team = o.findAll('div', {'class': 'ScoreCell__TeamName'})[0].text
            score_str = o.findAll('div', {'class': 'ScoreCell__Score'})[0].text
            score = float(score_str) if scoring == 'points' else score_str
            res.append((team, score))
        matchups.append(res)
    return matchups
