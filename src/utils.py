import time

from bs4 import BeautifulSoup
from selenium.webdriver import Chrome


ZERO = 1e-7

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


def get_scoreboard_stats(league_id, league_type, n_weeks, sleep_timeout, scoring='points'):
    espn_scoreboard_url = f'https://fantasy.espn.com/{league_type}/league/scoreboard'
    urls = [f'{espn_scoreboard_url}?leagueId={league_id}&matchupPeriodId={i+1}' for i in range(n_weeks)]
    browser = Chrome()
    all_matchups = []
    for u in urls:
        browser.get(u)
        time.sleep(sleep_timeout)
        html_soup = BeautifulSoup(browser.page_source, features='html.parser')
        all_matchups.append(get_week_scores(html_soup, scoring))
    return all_matchups, get_league_name(html_soup)


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
