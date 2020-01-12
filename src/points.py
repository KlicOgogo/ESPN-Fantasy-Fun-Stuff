from collections import defaultdict, Counter
import copy
import datetime
from operator import itemgetter

import pandas as pd
import numpy as np

import html_utils
import styling
import utils


def _export_league_stats(league, sport, test_mode_on, sleep_timeout):
    tables = {}
    scores = defaultdict(list)
    today = datetime.datetime.today().date()
    season_start_year = today.year if today.month > 6 else today.year - 1
    schedule = html_utils.get_league_schedule(league, sport, season_start_year, sleep_timeout)
    matchup = 1 if test_mode_on else utils.find_proper_matchup(schedule)
    if matchup == -1:
        return {}, {}, '', matchup, {}

    luck = defaultdict(list)
    opp_luck = defaultdict(list)
    places = defaultdict(list)
    opp_places = defaultdict(list)
    h2h_comparisons = defaultdict(lambda: defaultdict(Counter))
    all_pairs, _, name = html_utils.get_scoreboard_stats(league, sport, matchup, sleep_timeout)
    for matchup_results in all_pairs:
        for sc in matchup_results:
            scores[sc[0][0]].append(sc[0][1])
            scores[sc[1][0]].append(sc[1][1])

        format_lambda = lambda x: x if x % 1.0 > utils.ZERO else int(x)
        matchup_scores = sorted([s for pair in matchup_results for s in pair], key=itemgetter(1), reverse=True)
        matchup_places = utils.get_places(matchup_scores)
        opp_dict = utils.get_opponent_dict(matchup_results)
        for team in matchup_places:
            places[team].append(format_lambda(matchup_places[team]))
            opp_places[team].append(format_lambda(matchup_places[opp_dict[team]]))
        matchup_luck = _get_luck(matchup_results, matchup_places)
        for team in matchup_luck:
            luck[team].append(format_lambda(matchup_luck[team]))
            opp_luck[team].append(format_lambda(matchup_luck[opp_dict[team]]))
            
        for team, score in matchup_scores:
            for opp, opp_score in matchup_scores:
                if team == opp:
                    continue
                h2h_res = 'W' if score > opp_score else 'D' if score == opp_score else 'L'
                h2h_comparisons[team][opp][h2h_res] += 1
    matchups = np.arange(1, matchup + 1)
    tables['Luck scores'] = _render_luck_table(luck, matchups, False)
    tables['Opponent luck scores'] = _render_luck_table(opp_luck, matchups, True)
    tables['Places'] = _render_places_table(places, matchups, False)
    tables['Opponent places'] = _render_places_table(opp_places, matchups, True)
    tables['Pairwise comparisons h2h'] = utils.render_h2h_table(h2h_comparisons)
    return tables, scores, name, matchup, schedule


def _get_luck(pairs, places):
    luck = {}
    for player1, player2 in pairs:
        if player1[1] > player2[1]:
            luck[player1[0]] = max(0, places[player1[0]] - len(places) / 2)
            luck[player2[0]] = min(0, places[player2[0]] - len(places) / 2 - 1)
        elif player1[1] < player2[1]:
            luck[player1[0]] = min(0, places[player1[0]] - len(places) / 2 - 1)
            luck[player2[0]] = max(0, places[player2[0]] - len(places) / 2)
        else:
            if places[player1[0]] > len(places) / 2:
                luck[player1[0]] = (places[player1[0]] - len(places) / 2) / 2
                luck[player2[0]] = (places[player1[0]] - len(places) / 2) / 2
            else:
                luck[player1[0]] = (places[player1[0]] - len(places) / 2 - 1) / 2
                luck[player2[0]] = (places[player1[0]] - len(places) / 2 - 1) / 2
    return luck


def _render_luck_table(luck, matchups, opp_flag):
    df_data = copy.deepcopy(luck)
    for team in df_data:
        n_positive = np.sum(np.array(df_data[team]) > 0)
        n_negative = np.sum(np.array(df_data[team]) < 0)
        df_data[team].append(n_positive if opp_flag else n_negative)
        df_data[team].append(n_negative if opp_flag else n_positive)
        df_data[team].append(np.sum(luck[team]))

    df_teams = pd.DataFrame(list(map(itemgetter(0), df_data.keys())), index=df_data.keys(), columns=['Team'])
    cols = [*matchups, '&#128532;', '&#128526;', 'SUM']
    df = pd.DataFrame(list(df_data.values()), index=df_data.keys(), columns=cols)
    df = df_teams.merge(df, how='outer', left_index=True, right_index=True)
    sort_cols = ['&#128532;', '&#128526;', 'SUM'] if opp_flag else ['&#128526;', '&#128532;', 'SUM']
    sort_indexes = np.lexsort([df[col] * coeff for col, coeff in zip(sort_cols, [1.0, -1.0, 1.0])])
    df = df.iloc[sort_indexes]
    df = utils.add_position_column(df)
    styler = df.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
        applymap(styling.color_opponent_value if opp_flag else styling.color_value, subset=matchups)
    return styler.render()


def _render_places_table(places, matchups, opp_flag, is_overall=False):
    df_data = copy.deepcopy(places)
    for team in df_data:
        n_top = np.sum(styling.is_top(np.array(df_data[team]), df_data))
        n_bottom = np.sum(styling.is_bottom(np.array(df_data[team]), df_data))
        df_data[team].append(n_top if opp_flag else n_bottom)
        df_data[team].append(n_bottom if opp_flag else n_top)
        df_data[team].append(np.sum(places[team]))
        
    df_teams = pd.DataFrame(list(map(itemgetter(2, 0) if is_overall else itemgetter(0), df_data.keys())),
                            index=df_data.keys(), columns=['League', 'Team'] if is_overall else ['Team'])
    cols = [*matchups, '&#128532;', '&#128526;', 'SUM']
    df = pd.DataFrame(list(df_data.values()), index=df_data.keys(), columns=cols)
    df = df_teams.merge(df, how='outer', left_index=True, right_index=True)
    sort_cols = ['&#128526;', '&#128532;', 'SUM'] if opp_flag else ['&#128532;', '&#128526;', 'SUM']
    sort_indexes = np.lexsort([df[col] * coeff for col, coeff in zip(sort_cols, [1.0, -1.0, 1.0])])
    df = df.iloc[sort_indexes]
    df = utils.add_position_column(df)
    styler = df.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
        apply(styling.color_opponent_place_column if opp_flag else styling.color_place_column, subset=matchups)
    return styler.render()


def _render_top(data, n_top, cols):
    df_data = sorted(data, key=itemgetter(1), reverse=True)[:n_top]
    df = pd.DataFrame(df_data, index=np.arange(1, 1 + n_top), columns=cols)
    df = utils.add_position_column(df)
    return df.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().render()


def export_matchup_stats(leagues, sport, github_login, test_mode_on=False, sleep_timeout=10):
    overall_scores = defaultdict(list)
    leagues_tables = defaultdict(dict)
    for league in leagues:
        tables, scores, name, matchup, schedule = _export_league_stats(league, sport, test_mode_on, sleep_timeout)
        if matchup == -1:
            return
        overall_scores.update(scores)
        leagues_tables[name] = tables

    overall_tables = {}
    if len(leagues) > 1:
        overall_places = defaultdict(list)
        for i in range(matchup):
            matchup_overall_scores = [(team, overall_scores[team][i]) for team in overall_scores]
            matchup_overall_scores = sorted(matchup_overall_scores, key=itemgetter(1), reverse=True)
            matchup_overall_places = utils.get_places(matchup_overall_scores)
            for team in matchup_overall_places:
                overall_places[team].append(matchup_overall_places[team])
        overall_tables['Overall places'] = _render_places_table(overall_places, np.arange(1, matchup + 1), False, True)

    n_top = int(len(overall_scores) / len(leagues))
    top_common_cols = ['Team', 'Score', 'League']
    last_matchup_scores = [(team[0], overall_scores[team][-1], team[2]) for team in overall_scores]
    overall_tables['Best scores this matchup'] = _render_top(last_matchup_scores, n_top, top_common_cols)
    each_matchup_scores = []
    for team in overall_scores:
        team_scores = [(team[0], score, team[2], index + 1) for index, score in enumerate(overall_scores[team])]
        each_matchup_scores.extend(team_scores)
    overall_tables['Best scores this season'] = _render_top(each_matchup_scores, n_top, top_common_cols + ['Matchup'])
    totals = [(team[0], np.sum(scores), team[2], scores[-1]) for team, scores in overall_scores.items()]
    overall_tables['Best total scores this season'] = _render_top(totals, n_top, top_common_cols + ['Last matchup'])

    today = datetime.datetime.today().date()
    season_start_year = today.year if today.month > 6 else today.year - 1
    season_str = f'{season_start_year}-{str(season_start_year + 1)[-2:]}'
    utils.export_tables_to_html(sport, leagues_tables, overall_tables,
                                leagues[0], season_str, matchup, github_login, schedule, test_mode_on)
