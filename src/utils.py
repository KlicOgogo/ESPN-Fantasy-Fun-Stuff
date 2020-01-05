import os
from pathlib import Path
import time

from bs4 import BeautifulSoup
from jinja2 import Template
from selenium.webdriver import Chrome


_BROWSER = Chrome()
ATTRS = 'style="border-collapse: collapse; border: 1px solid black;" align= "center"'
REPO_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
STYLES = [
    dict(selector='td', props=[('border-left', '1px solid black'), ('border-right', '1px solid black'),
                               ('text-align', 'right'), ('padding-left', '4px'), ('padding-right', '4px')]),
    dict(selector='td:first-child', props=[('border-left', 'none')]),
    dict(selector='td:last-child', props=[('border-right', 'none')]),
    dict(selector='th', props=[('border-left', '1px solid black'), ('border-right', '1px solid black'),
                               ('border-bottom', '1px solid black'), ('background', '#FFFFFF'),
                               ('padding-left', '6px')]),
    dict(selector='tr:nth-child(odd)', props=[('background', '#F0F0F0')]),
]
ZERO = 1e-7


def _get_league_name(scoreboard_html):
    return scoreboard_html.findAll('h3')[0].text


def _get_matchup_scores(scoreboard_html, scoring='points'):
    if scoring not in ['points', 'categories']:
        raise Exception('Wrong scoring parameter!')
    matchups = []
    matchups_html = scoreboard_html.findAll('div', {'Scoreboard__Row'})
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


def export_tables_to_html(sport, leagues_tables, total_tables, league_id, season, matchup):
    with open(os.path.join(REPO_ROOT_DIR, 'templates/matchup_report.html'), 'r') as template_fp:
        template = Template(template_fp.read())
    html_str = template.render({
        'sport': sport,
        'leagues': leagues_tables,
        'total_tables': total_tables
    })
    index_html_path = os.path.join(REPO_ROOT_DIR, 'reports', sport, str(league_id), season, 'index.html')
    matchup_html_path = os.path.join(REPO_ROOT_DIR, 'reports', sport, str(league_id), season, f'matchup_{matchup}.html')
    Path(os.path.dirname(index_html_path)).mkdir(parents=True, exist_ok=True)
    with open(index_html_path, 'w') as html_fp:
        html_fp.write(html_str)
    with open(matchup_html_path, 'w') as html_fp:
        html_fp.write(html_str)


def get_minutes(league, matchup, n_teams, sleep_timeout=10):
    espn_fantasy_url = 'https://fantasy.espn.com/basketball'
    urls = [(f'{espn_fantasy_url}/boxscore?leagueId={league}&matchupPeriodId={matchup}'
             f'&scoringPeriodId={1 + (matchup - 1) * 7}' # fix with cronable publisher
             f'&seasonId=2020&teamId={t}&view=matchup') for t in range(1, n_teams+1)] # fix season
    minutes_dict = {}
    for u in urls:
        _BROWSER.get(u)
        time.sleep(sleep_timeout)
        html_soup = BeautifulSoup(_BROWSER.page_source, features='html.parser')
        tables_html = html_soup.findAll('div', {'class': 'players-table__sortable'})
        teams_html = html_soup.findAll('span', {'class': "team-name truncate"})
        for team_html, table_html in zip(teams_html, tables_html):
            team = team_html.text.replace(' Box Score', '')
            minutes = int(table_html.findAll('tr')[-1].findAll('td')[0].text)
            minutes_dict[team] = minutes
    return minutes_dict


def get_places(sorted_scores):
    places = {}
    i = 1
    while i <= len(sorted_scores):
        j = i + 1
        while j <= len(sorted_scores) and sorted_scores[j-1][1] == sorted_scores[i-1][1]:
            j += 1
        place = (i + j - 1) / 2
        for k in range(i, j):
            places[sorted_scores[k - 1][0]] = place
        i = j
    return places


def get_scoreboard_stats(league_id, sport, matchup, sleep_timeout=10, scoring='points'):
    espn_scoreboard_url = f'https://fantasy.espn.com/{sport}/league/scoreboard'
    urls = [f'{espn_scoreboard_url}?leagueId={league_id}&matchupPeriodId={m}' for m in range(1, matchup + 1)]
    all_matchups = []
    soups = []
    for u in urls:
        _BROWSER.get(u)
        time.sleep(sleep_timeout)
        html_soup = BeautifulSoup(_BROWSER.page_source, features='html.parser')
        soups.append(html_soup)
        all_matchups.append(_get_matchup_scores(html_soup, scoring))
    return all_matchups, soups, _get_league_name(html_soup)


def make_data_row(dict_item):
    return [dict_item[0], *dict_item[1]]
