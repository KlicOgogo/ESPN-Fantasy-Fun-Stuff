from collections import defaultdict
from operator import itemgetter

import numpy as np

from src.utils import get_league_name, get_places, get_week_scores


NUMBERED_VALUE_COLS = {'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS', 'TP'}
ZERO_RES = 1e-7


def format_value(x):
    return str(x) if x % 1.0 > ZERO_RES else str(int(x))


def _get_best_and_worst_values(table, col):
    if col in NUMBERED_VALUE_COLS:
        if col == 'TO':
            return table[col].min(), table[col].max()
        else:
            return table[col].max(), table[col].min()
    else:
        scores_for_sort = []
        for sc in table[col]:
            sc_values = list(map(float, sc.split('-')))
            scores_for_sort.append([sc_values[i] for i in [0, 2, 1]])
        max_val = max(scores_for_sort)
        min_val = min(scores_for_sort)
        format_score_lambda = lambda x: '-'.join(map(format_value, [x[i] for i in [0, 2, 1]]))
        return format_score_lambda(max_val), format_score_lambda(min_val)


def get_best_and_worst_rows(table):
    empty_value_cols = {'ER', 'SUM', 'W', 'L', 'D', 'WD', 'LD', 'DD'} | {f'{col} ' for col in NUMBERED_VALUE_COLS}
    best = {}
    worst = {}
    for col in table.columns:
        if col in empty_value_cols:
            best[col], worst[col] = ('', '')
        else:
            best[col], worst[col] = _get_best_and_worst_values(table, col)
    return best, worst


def get_diff(expected, real):
    return list(map(lambda x: float(x[0]) - float(x[1]), zip(expected.split('-'), real.split('-'))))


def _get_expected_category_stats(score_pairs, category):
    scores = np.array([score for _, score in score_pairs])
    result = {}
    for team, sc in score_pairs:
        greater_count = np.sum(scores < sc)
        equal_count = np.sum(scores == sc) - 1
        less_count = np.sum(scores > sc)
        if category == 'TO':
            result[team] = np.array([less_count, greater_count, equal_count]) / (len(scores) - 1)
        else:
            result[team] = np.array([greater_count, less_count, equal_count]) / (len(scores) - 1)
    return result


def get_expected_score_and_result(results, categories):
    res = {}
    opponents_dict = {}
    for matchup in results:
        opponents_dict[matchup[0][0]] = matchup[1][0]
        opponents_dict[matchup[1][0]] = matchup[0][0]
        res[matchup[0][0]] = np.array([0.0, 0.0, 0.0])
        res[matchup[1][0]] = np.array([0.0, 0.0, 0.0])
    scores_info = get_scores_info(results)

    for i, cat in enumerate(categories):
        pairs = [(team, scores_info[team][i]) for team in scores_info]
        expected_stats = _get_expected_category_stats(pairs, cat)
        for team in expected_stats:
            concatted = np.vstack((expected_stats[team], res[team]))
            res[team] = concatted.sum(axis=0)
    matchup_results = {}
    for team in opponents_dict:
        if list(res[team][[0, 2, 1]]) > list(res[opponents_dict[team]][[0, 2, 1]]):
            matchup_results[team] = 'W'
        elif list(res[team][[0, 2, 1]]) < list(res[opponents_dict[team]][[0, 2, 1]]):
            matchup_results[team] = 'L'
        else:
            matchup_results[team] = 'D'
    exp_scores = {team: '-'.join(map(lambda x: format_value(np.round(x, 1)), res[team])) for team in res}
    return exp_scores, matchup_results


def get_matchup_result(team_stat, opp_stat, categories):
    win_count = 0
    lose_count = 0
    for index, cat in enumerate(categories):
        if team_stat[index] > opp_stat[index]:
            lose_count += (cat == 'TO')
            win_count += (cat != 'TO')
        elif team_stat[index] < opp_stat[index]:
            lose_count += (cat != 'TO')
            win_count += (cat == 'TO')
    result = 'D' if win_count == lose_count else 'W' if win_count > lose_count else 'L'
    return result


def get_places_data(table):
    places_data = defaultdict(list)
    for col in table.columns:
        pairs = [(team, table[col][team]) for team in table[col].index]
        pairs_sorted = sorted(pairs, key=itemgetter(1), reverse=False if col == 'TO' else True)
        places = get_places(pairs_sorted)
        for team in places:
            places_data[team].append(places[team])
    for team in places_data:
        places_data[team].append(np.sum(places_data[team]))
    return places_data


def get_scores_info(results):
    scores_info = {}
    for matchup in results:
        for team, total_score in matchup:
            scores_info[team] = [score if score % 1.0 > ZERO_RES else int(score) for _, score in total_score]
    return scores_info


def get_team_win_stat(team_stat):
    return '-'.join(map(format_value, [team_stat.count('W'), team_stat.count('L'), team_stat.count('D')]))


def get_week_matchups(scoreboard_html_source):
    matchups_html = scoreboard_html_source.findAll('div', {'Scoreboard__Row'})
    matchups = []
    for m in matchups_html:
        opponents = m.findAll('li', 'ScoreboardScoreCell__Item')
        team_names = [o.findAll('div', {'class': 'ScoreCell__TeamName'})[0].text for o in opponents]

        rows = m.findAll('tr', {'Table2__tr'})
        categories = [header.text for header in rows[0].findAll('th', {'Table2__th'})[1:]]
        first_team_stats = [data.text for data in rows[1].findAll('td', {'Table2__td'})[1:]]
        second_team_stats = [data.text for data in rows[2].findAll('td', {'Table2__td'})[1:]]

        matchups.append(
            ((team_names[0], [(cat, float(stat)) for cat, stat in zip(categories, first_team_stats)]),
             (team_names[1], [(cat, float(stat)) for cat, stat in zip(categories, second_team_stats)])))
    return matchups, categories
