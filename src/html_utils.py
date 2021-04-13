import datetime
import os
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
from selenium.webdriver import Chrome

from utils import REPO_ROOT_DIR

_BROWSER = Chrome()


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
    start_year = season_start_year if start_month > 6 else season_start_year + 1
    end_components = end_str.split(' ')
    end_month = start_month if len(end_components) == 1 else months[end_components[0].lower()]
    end_day = int(end_components[0]) if len(end_components) == 1 else int(end_components[1])
    end_year = season_start_year if end_month > 6 else season_start_year + 1
    get_day = lambda year, month, day: datetime.datetime(year=year, month=month, day=day).date()
    return (get_day(start_year, start_month, start_day), get_day(end_year, end_month, end_day))


def _get_matchup_number(matchup_text):
    matches = re.findall(r'Matchup (\d+)', matchup_text)
    return int(matches[0]) if len(matches) == 1 else None


def _get_matchup_schedule(matchup_text, season_start_year):
    matchup_number = _get_matchup_number(matchup_text)
    if matchup_number is None:
        return []
    matchup_date = _get_matchup_date(matchup_text, season_start_year)
    return [(matchup_number, matchup_date)]


def _get_matchup_scores(scoreboard_html, league_id):
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
            score = float(score_str) if len(score_str.split('-')) == 1 else score_str
            res.append((team, score))
        matchups.append(res)
    return matchups


def get_league_schedule(league_id, sport, season_start_year, sleep_timeout=10):
    today = datetime.datetime.today().date()
    season_start_year = today.year if today.month > 6 else today.year - 1    
    season_str = f'{season_start_year}-{str(season_start_year + 1)[-2:]}'
    offline_schedule_path = os.path.join(REPO_ROOT_DIR, 'data', sport, str(league_id), season_str, 'matchup_1.html')
    
    if os.path.exists(offline_schedule_path):
        scoreboard_html = BeautifulSoup(open(offline_schedule_path, 'r', encoding='utf-8'), features='html.parser')
    else:    
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
        exit = False
        while not exit:
            _BROWSER.get(url)
            time.sleep(sleep_timeout)
            html_soup = BeautifulSoup(_BROWSER.page_source, features='html.parser')
            tables_html = html_soup.findAll('div', {'class': 'players-table__sortable'})
            for player, table_html in zip(pair, tables_html):
                minutes = int(table_html.findAll('tr')[-1].findAll('td')[0].findAll('div')[0].text)
                if minutes == minutes:
                    minutes_dict[player] = minutes
                    exit = True
    return minutes_dict


def get_scoreboard_stats(league_id, sport, matchup, sleep_timeout=10):
    espn_scoreboard_url = f'https://fantasy.espn.com/{sport}/league/scoreboard'
    urls = [f'{espn_scoreboard_url}?leagueId={league_id}&matchupPeriodId={m}' for m in range(1, matchup + 1)]
    all_matchups = []
    soups = []
    today = datetime.datetime.today().date()
    season_start_year = today.year if today.month > 6 else today.year - 1    
    season_str = f'{season_start_year}-{str(season_start_year + 1)[-2:]}'
    
    offline_html_dir = os.path.join(REPO_ROOT_DIR, 'data', sport, str(league_id), season_str)
    Path(offline_html_dir).mkdir(parents=True, exist_ok=True)

    for index, u in enumerate(urls):
        matchup_html_path = os.path.join(offline_html_dir, f'matchup_{index + 1}.html')
        if not os.path.exists(matchup_html_path) or index + 4 > matchup:
            _BROWSER.get(u)
            time.sleep(sleep_timeout)
            html_soup = BeautifulSoup(_BROWSER.page_source, features='html.parser')
            with open(matchup_html_path, 'w', encoding='utf-8') as html_fp:
                html_fp.write(str(html_soup))
        else:
            html_soup = BeautifulSoup(open(matchup_html_path, 'r', encoding='utf-8'), features='html.parser')
        soups.append(html_soup)
        all_matchups.append(_get_matchup_scores(html_soup, league_id))
    return all_matchups, soups, _get_league_name(html_soup)
