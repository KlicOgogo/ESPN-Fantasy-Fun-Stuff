import datetime
import os
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
from jinja2 import Template
import pandas as pd
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


def _get_matchup_date(matchup_text, season_start_year):
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    start_end_str = re.findall(r'\((.+)\)', matchup_text)[0]
    start_str, end_str = map(lambda x: x.strip().lstrip(), start_end_str.split('-'))
    start_components = start_str.split(' ')
    start_month = months[start_components[0].lower()]
    start_day = int(start_components[1])
    end_components = end_str.split(' ')
    if len(end_components) == 1:
        end_month = start_month
        end_day = int(end_components[0])
    else:
        end_month = months[end_components[0].lower()]
        end_day = int(end_components[1])
    start_year = season_start_year if start_month > 6 else season_start_year + 1
    end_year = season_start_year if end_month > 6 else season_start_year + 1
    get_day_str = lambda year, month, day: datetime.datetime(year=year, month=month, day=day).date()
    return (
        get_day_str(start_year, start_month, start_day),
        get_day_str(end_year, end_month, end_day)
    )


def _get_matchup_number(matchup_text):
    return int(re.findall(r'Matchup (\d+)', matchup_text)[0])


def _get_matchup_schedule(matchup_text, season_start_year):
    matchup_number = _get_matchup_number(matchup_text)
    matchup_date = _get_matchup_date(matchup_text, season_start_year)
    return [(matchup_number, matchup_date)]


def _get_matchup_scores(scoreboard_html, league_id, scoring='points'):
    if scoring not in ['points', 'categories']:
        raise Exception('Wrong scoring parameter!')
    league_name = _get_league_name(scoreboard_html)
    matchups = []
    matchups_html = scoreboard_html.findAll('div', {'Scoreboard__Row'})
    for m in matchups_html:
        opponents = m.findAll('li', 'ScoreboardScoreCell__Item')
        res = []
        for o in opponents:
            team_id = re.findall(r'teamId=(\d+)', o.findAll('a', {'class': 'truncate'})[0]['href'])[0]
            team_name = o.findAll('div', {'class': 'ScoreCell__TeamName'})[0].text
            team = (team_name, team_id, league_name, league_id)
            score_str = o.findAll('div', {'class': 'ScoreCell__Score'})[0].text
            score = float(score_str) if scoring == 'points' else score_str
            res.append((team, score))
        matchups.append(res)
    return matchups


def add_position_column(df):
    position = {index: i + 1 for i, index in enumerate(df.index)}
    df_position = pd.DataFrame(data=list(position.values()), index=position.keys(), columns=['Pos'])
    result_df = df_position.merge(df, how='outer', left_index=True, right_index=True)
    return result_df


def export_tables_to_html(sport, leagues_tables, total_tables, league_id, season, matchup, test_mode_on=False):
    with open(os.path.join(REPO_ROOT_DIR, 'templates/matchup_report.html'), 'r') as template_fp:
        template = Template(template_fp.read())
    html_str = template.render({
        'sport': sport,
        'leagues': leagues_tables,
        'total_tables': total_tables
    })
    index_path = os.path.join(REPO_ROOT_DIR, 'reports', sport, str(league_id), season, 'index.html')
    Path(os.path.dirname(index_path)).mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as html_fp:
        html_fp.write(html_str)
    if not test_mode_on:
        matchup_path = os.path.join(REPO_ROOT_DIR, 'reports', sport, str(league_id), season, f'matchup_{matchup}.html')
        with open(matchup_path, 'w') as html_fp:
            html_fp.write(html_str)


def get_league_schedule(league_id, sport, season_start_year, sleep_timeout=10):
    espn_scoreboard_url = f'https://fantasy.espn.com/{sport}/league/scoreboard'
    url = f'{espn_scoreboard_url}?leagueId={league_id}&matchupPeriodId=1'
    _BROWSER.get(url)
    time.sleep(sleep_timeout)
    scoreboard_html = BeautifulSoup(_BROWSER.page_source, features='html.parser')

    matchups_dropdown = scoreboard_html.findAll('div', {'class': 'dropdown'})[0]
    matchups_html_list = matchups_dropdown.findAll('option')
    schedule = {}
    for matchup_html in matchups_html_list:
        schedule.update(_get_matchup_schedule(matchup_html.text, season_start_year))
    return schedule


def get_minutes(league, matchup, teams, scoring_period_id, season_id, sleep_timeout=10):
    espn_fantasy_url = 'https://fantasy.espn.com/basketball'
    minutes_dict = {}
    for pair in teams:
        url = (f'{espn_fantasy_url}/boxscore?leagueId={league}&matchupPeriodId={matchup}'
               f'&scoringPeriodId={scoring_period_id}'
               f'&seasonId={season_id}&teamId={pair[0][1]}&view=matchup')
        _BROWSER.get(url)
        time.sleep(sleep_timeout)
        html_soup = BeautifulSoup(_BROWSER.page_source, features='html.parser')
        tables_html = html_soup.findAll('div', {'class': 'players-table__sortable'})
        for index, table_html in enumerate(tables_html):
            minutes = int(table_html.findAll('tr')[-1].findAll('td')[0].text)
            minutes_dict[pair[index]] = minutes
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
        all_matchups.append(_get_matchup_scores(html_soup, league_id, scoring))
    return all_matchups, soups, _get_league_name(html_soup)
