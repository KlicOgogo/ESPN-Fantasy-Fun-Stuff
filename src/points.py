import datetime
from collections import defaultdict
from operator import itemgetter as _itemgetter

import pandas as pd
import numpy as np

import styling
import utils

def _get_luck(pairs, places):
    luck = {}
    for player1, player2 in pairs:
        if player1[1] > player2[1]:
            place1 = places[player1[0]]
            luck[player1[0]] = max(0, place1 - len(places) / 2)

            place2 = places[player2[0]]
            luck[player2[0]] = min(0, place2 - len(places) / 2 - 1)
        elif player1[1] < player2[1]:
            place1 = places[player1[0]]
            luck[player1[0]] = min(0, place1 - len(places) / 2 - 1)

            place2 = places[player2[0]]
            luck[player2[0]] = max(0, place2 - len(places) / 2)
        else:
            place = places[player1[0]]
            if place > len(places) / 2:
                luck[player1[0]] = (place - len(places) / 2) / 2
                luck[player2[0]] = (place - len(places) / 2) / 2
            else:
                luck[player1[0]] = (place - len(places) / 2 - 1) / 2
                luck[player2[0]] = (place - len(places) / 2 - 1) / 2
    return luck


def _get_sorted_matchup_scores(matchup_pairs):
    scores = []
    for pair in matchup_pairs:
        scores.extend(pair)
    return sorted(scores, key=_itemgetter(1), reverse=True)


def export_matchup_stats(leagues, sport, test_mode_on=False, sleep_timeout=10):
    overall_scores = defaultdict(list)
    leagues_tables = defaultdict(dict)
    for league in leagues:
        luck = defaultdict(list)
        opp_luck = defaultdict(list)
        places = defaultdict(list)
        opp_places = defaultdict(list)

        today = datetime.datetime.today().date()
        this_season_begin_year = today.year if today.month > 6 else today.year - 1
        matchup = 1
        if not test_mode_on:
            matchup = -1
            schedule, = utils.get_league_schedule(league, sport, this_season_begin_year, sleep_timeout)
            yesterday = today - datetime.timedelta(days=1)
            for matchup_number, matchup_date in schedule.items():
                if yesterday >= matchup_date[0] and yesterday == matchup_date[1]:
                    matchup = matchup_number
                    break
        if matchup == -1:
            return

        all_pairs, _, league_name = utils.get_scoreboard_stats(league, sport, matchup, sleep_timeout)
        for matchup_results in all_pairs:
            opp_dict = {}
            for sc in matchup_results:
                opp_dict[sc[0][0]] = sc[1][0]
                opp_dict[sc[1][0]] = sc[0][0]
                overall_scores[sc[0][0]].append(sc[0][1])
                overall_scores[sc[1][0]].append(sc[1][1])

            matchup_scores = _get_sorted_matchup_scores(matchup_results)
            matchup_places = utils.get_places(matchup_scores)
            matchup_luck = _get_luck(matchup_results, matchup_places)
            format_lambda = lambda x: x if x % 1.0 > utils.ZERO else int(x)
            for team in matchup_luck:
                luck[team].append(format_lambda(matchup_luck[team]))
                places[team].append(format_lambda(matchup_places[team]))
                opp_luck[team].append(format_lambda(matchup_luck[opp_dict[team]]))
                opp_places[team].append(format_lambda(matchup_places[opp_dict[team]]))

        for team in luck:
            luck[team].extend([
                np.sum(np.array(luck[team]) < 0),
                np.sum(np.array(luck[team]) > 0),
                np.sum(luck[team])
            ])
            places[team].extend([
                np.sum(np.array(places[team]) / len(luck) > 1.0 - styling.TOP_PERC),
                np.sum(np.array(places[team]) / len(luck) <= styling.TOP_PERC),
                np.sum(places[team])
            ])
            opp_luck[team].extend([
                np.sum(np.array(opp_luck[team]) > 0),
                np.sum(np.array(opp_luck[team]) < 0),
                np.sum(opp_luck[team])
            ])
            opp_places[team].extend([
                np.sum(np.array(opp_places[team]) / len(luck) <= styling.TOP_PERC),
                np.sum(np.array(opp_places[team]) / len(luck) > 1.0 - styling.TOP_PERC),
                np.sum(opp_places[team])
            ])

        matchups = [w for w in range(1, matchup + 1)]
        cols = [*matchups, '&#128532;', '&#128526;', 'SUM']
        df_teams = pd.DataFrame(data=list(map(lambda x: x[0], luck.keys())), index=luck.keys(), columns=['Team'])
        overall_tables = {}

        df_luck = pd.DataFrame(data=list(luck.values()), index=luck.keys(), columns=cols)
        df_luck = df_teams.merge(df_luck, how='outer', left_index=True, right_index=True)
        sort_indexes = np.lexsort((df_luck['&#128526;'], -df_luck['&#128532;'], df_luck['SUM']))
        df_luck = df_luck.iloc[sort_indexes]
        df_luck = utils.add_position_column(df_luck)
        styler = df_luck.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            applymap(styling.color_value, subset=matchups)
        leagues_tables[league_name]['Luck scores'] = styler.render()

        df_places = pd.DataFrame(data=list(places.values()), index=places.keys(), columns=cols)
        df_places = df_teams.merge(df_places, how='outer', left_index=True, right_index=True)
        sort_indexes = np.lexsort((df_places['&#128532;'], -df_places['&#128526;'], df_places['SUM']))
        df_places = df_places.iloc[sort_indexes]
        df_places = utils.add_position_column(df_places)
        styler = df_places.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            apply(styling.color_place_column, subset=matchups)
        leagues_tables[league_name]['Places'] = styler.render()

        df_opp_luck = pd.DataFrame(data=list(opp_luck.values()), index=opp_luck.keys(), columns=cols)
        df_opp_luck = df_teams.merge(df_opp_luck, how='outer', left_index=True, right_index=True)
        sort_indexes = np.lexsort((df_opp_luck['&#128532;'], -df_opp_luck['&#128526;'], df_opp_luck['SUM']))
        df_opp_luck = df_opp_luck.iloc[sort_indexes]
        df_opp_luck = utils.add_position_column(df_opp_luck)
        styler = df_opp_luck.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            applymap(styling.color_opponent_value, subset=matchups)
        leagues_tables[league_name]['Opponent luck scores'] = styler.render()

        df_opp_places = pd.DataFrame(data=list(opp_places.values()), index=opp_places.keys(), columns=cols)
        df_opp_places = df_teams.merge(df_opp_places, how='outer', left_index=True, right_index=True)
        sort_indexes = np.lexsort((df_opp_places['&#128526;'], -df_opp_places['&#128532;'], df_opp_places['SUM']))
        df_opp_places = df_opp_places.iloc[sort_indexes]
        df_opp_places = utils.add_position_column(df_opp_places)
        styler = df_opp_places.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            apply(styling.color_opponent_place_column, subset=matchups)
        leagues_tables[league_name]['Opponent places'] = styler.render()

    n_top = int(len(overall_scores) / len(leagues))
    last_matchup_scores = [(team[0], overall_scores[team][-1], team[2]) for team in overall_scores]
    df_data = sorted(last_matchup_scores, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *data_row] for i, data_row in enumerate(df_data)],
                        index=np.arange(1, 1 + n_top), columns=['Pos', 'Team', 'Score', 'League'])
    styler = data.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index()
    overall_tables['Best scores this matchup'] = styler.render()

    all_scores = []
    for team in overall_scores:
        team_scores = [(team[0], score, team[2], index + 1) for index, score in enumerate(overall_scores[team])]
        all_scores.extend(team_scores)
    df_data = sorted(all_scores, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *data_row] for i, data_row in enumerate(df_data)],
                        index=np.arange(1, 1 + n_top), columns=['Pos', 'Team', 'Score', 'League', 'Matchup'])
    styler = data.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index()
    overall_tables['Best scores this season'] = styler.render()

    total_sums = [(team[0], np.sum(scores), team[2], scores[-1]) for team, scores in overall_scores.items()]
    df_data = sorted(total_sums, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *data_row] for i, data_row in enumerate(df_data)],
                        index=np.arange(1, 1 + n_top), columns=['Pos', 'Team', 'Score', 'League', 'Last matchup'])
    styler = data.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index()
    overall_tables['Best total scores this season'] = styler.render()

    if len(leagues) > 1:
        overall_places = defaultdict(list)
        for i in range(matchup):
            matchup_overall_scores = [(team, overall_scores[team][i]) for team in overall_scores]
            matchup_overall_scores = sorted(matchup_overall_scores, key=_itemgetter(1), reverse=True)
            matchup_overall_places = utils.get_places(matchup_overall_scores)
            for team in matchup_overall_places:
                overall_places[team].append(matchup_overall_places[team])
        for team in overall_places:
            overall_places[team].extend([
                np.sum(np.array(overall_places[team]) / len(overall_places) > 1.0 - styling.TOP_PERC),
                np.sum(np.array(overall_places[team]) / len(overall_places) <= styling.TOP_PERC),
                np.sum(overall_places[team])
            ])
        df_all_teams = pd.DataFrame(data=list(map(lambda x: (x[2], x[0]), overall_places.keys())),
                                    index=overall_places.keys(), columns=['League', 'Team'])
        df_overall_places = pd.DataFrame(data=list(overall_places.values()), index=overall_places.keys(), columns=cols)
        df_overall_places = df_all_teams.merge(df_overall_places, how='outer', left_index=True, right_index=True)
        sort_indexes = np.lexsort((df_overall_places['&#128532;'], -df_overall_places['&#128526;'],
                                   df_overall_places['SUM']))
        df_overall_places = df_overall_places.iloc[sort_indexes]
        df_overall_places = utils.add_position_column(df_overall_places)
        styler = df_overall_places.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            apply(styling.color_place_column, subset=matchups)
        overall_tables['Overall places'] = styler.render()

    season_str = f'{this_season_begin_year}-{str(this_season_begin_year + 1)[-2:]}'
    utils.export_tables_to_html(sport, leagues_tables, overall_tables, leagues[0], season_str, matchup, test_mode_on)
