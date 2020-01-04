from collections import defaultdict
from operator import itemgetter as _itemgetter

import pandas as pd
import numpy as np

from src import styling
from src.utils import export_tables_to_html, get_places, get_scoreboard_stats, make_data_row, ATTRS, STYLES, ZERO


def _get_luck_score(matchups, places):
    luck_score = {}
    for player1, player2 in matchups:
        if player1[1] > player2[1]:
            place1 = places[player1[0]]
            luck_score[player1[0]] = max(0, place1 - len(places) / 2)

            place2 = places[player2[0]]
            luck_score[player2[0]] = min(0, place2 - len(places) / 2 - 1)
        elif player1[1] < player2[1]:
            place1 = places[player1[0]]
            luck_score[player1[0]] = min(0, place1 - len(places) / 2 - 1)

            place2 = places[player2[0]]
            luck_score[player2[0]] = max(0, place2 - len(places) / 2)
        else:
            place = places[player1[0]]
            if place > len(places) / 2:
                luck_score[player1[0]] = (place - len(places) / 2) / 2
                luck_score[player2[0]] = (place - len(places) / 2) / 2
            else:
                luck_score[player1[0]] = (place - len(places) / 2 - 1) / 2
                luck_score[player2[0]] = (place - len(places) / 2 - 1) / 2
    return luck_score


def _get_sorted_week_scores(week_matchups):
    scores = []
    for matchup in week_matchups:
        scores.extend(matchup)
    return sorted(scores, key=_itemgetter(1), reverse=True)


def export_week_stats(leagues, sport, week, sleep_timeout=10):
    all_scores_dict = defaultdict(list)
    leagues_tables = defaultdict(dict)
    for league in leagues:
        luck_score = defaultdict(list)
        opp_luck_score = defaultdict(list)
        places = defaultdict(list)
        opp_places = defaultdict(list)

        all_matchups, _, league_name = get_scoreboard_stats(league, sport, week, sleep_timeout)
        for week_results in all_matchups:
            opp_dict = {}
            for sc in week_results:
                opp_dict[sc[0][0]] = sc[1][0]
                opp_dict[sc[1][0]] = sc[0][0]
                all_scores_dict[(sc[0][0], league_name)].append(sc[0][1])
                all_scores_dict[(sc[1][0], league_name)].append(sc[1][1])

            week_scores = _get_sorted_week_scores(week_results)
            week_places = get_places(week_scores)
            week_luck_score = _get_luck_score(week_results, week_places)
            format_lambda = lambda x: x if x % 1.0 > ZERO else int(x)
            for team in week_luck_score:
                luck_score[team].append(format_lambda(week_luck_score[team]))
                places[team].append(format_lambda(week_places[team]))
                opp_luck_score[team].append(format_lambda(week_luck_score[opp_dict[team]]))
                opp_places[team].append(format_lambda(week_places[opp_dict[team]]))

        for team in luck_score:
            luck_score[team].extend([
                np.sum(np.array(luck_score[team]) < 0),
                np.sum(np.array(luck_score[team]) > 0),
                np.sum(luck_score[team])
            ])
            places[team].extend([
                np.sum(np.array(places[team]) / len(luck_score) >= styling.TOP_PERCENTAGE),
                np.sum(np.array(places[team]) / len(luck_score) <= styling.TOP_PERCENTAGE),
                np.sum(places[team])
            ])
            opp_luck_score[team].extend([
                np.sum(np.array(opp_luck_score[team]) > 0),
                np.sum(np.array(opp_luck_score[team]) < 0),
                np.sum(opp_luck_score[team])
            ])
            opp_places[team].extend([
                np.sum(np.array(opp_places[team]) / len(luck_score) <= styling.TOP_PERCENTAGE),
                np.sum(np.array(opp_places[team]) / len(luck_score) >=styling.TOP_PERCENTAGE),
                np.sum(opp_places[team])
            ])

        weeks = [w for w in range(1, week + 1)]
        cols = ['Team', *weeks, '&#128532;', '&#128526;', 'SUM']
        total_tables = {}

        df_luck_score = pd.DataFrame(data=list(map(make_data_row, luck_score.items())), columns=cols)
        df_luck_score = df_luck_score.iloc[np.lexsort((-df_luck_score['&#128532;'], df_luck_score['SUM']))]
        styler = df_luck_score.style.set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index().\
            applymap(styling.color_value, subset=weeks)
        leagues_tables[league_name]['Luck scores'] = styler.render()

        df_places = pd.DataFrame(data=list(map(make_data_row, places.items())), columns=cols)
        df_places = df_places.iloc[np.lexsort((-df_places['&#128526;'], df_places['SUM']))]
        styler = df_places.style.set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index().\
            apply(styling.color_place_column, subset=weeks)
        leagues_tables[league_name]['Places'] = styler.render()

        df_opp_luck_score = pd.DataFrame(data=list(map(make_data_row, opp_luck_score.items())), columns=cols)
        df_opp_luck_score = df_opp_luck_score.iloc[np.lexsort((-df_opp_luck_score['&#128526;'], df_opp_luck_score['SUM']))]
        styler = df_opp_luck_score.style.set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index().\
            applymap(styling.color_opponent_value, subset=weeks)
        leagues_tables[league_name]['Opponent luck scores'] = styler.render()

        df_opp_places = pd.DataFrame(data=list(map(make_data_row, opp_places.items())), columns=cols)
        df_opp_places = df_opp_places.iloc[np.lexsort((-df_opp_places['&#128532;'], df_opp_places['SUM']))]
        styler = df_opp_places.style.set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index().\
            apply(styling.color_opponent_place_column, subset=weeks)
        leagues_tables[league_name]['Opponent places'] = styler.render()

    n_top = int(len(all_scores_dict) / len(leagues))
    last_week_scores = [(team[0], all_scores_dict[team][-1], team[1]) for team in all_scores_dict]
    df_data = sorted(last_week_scores, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *df_data[i]] for i in range(len(df_data))],
                        index=np.arange(1, 1 + n_top), columns=['Place', 'Team', 'Score', 'League'])
    styler = data.style.set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index()
    total_tables['Best scores this week'] = styler.render()

    all_scores = []
    for team in all_scores_dict:
        team_scores = [(team[0], score, team[1], index + 1) for index, score in enumerate(all_scores_dict[team])]
        all_scores.extend(team_scores)
    df_data = sorted(all_scores, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *df_data[i]] for i in range(len(df_data))],
                        index=np.arange(1, 1 + n_top), columns=['Place', 'Team', 'Score', 'League', 'Week'])
    styler = data.style.set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index()
    total_tables['Best scores this season'] = styler.render()

    total_sums = [(team[0], np.sum(scores), team[1], scores[-1]) for team, scores in all_scores_dict.items()]
    df_data = sorted(total_sums, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *df_data[i]] for i in range(len(df_data))],
                        index=np.arange(1, 1 + n_top), columns=['Place', 'Team', 'Score', 'League', 'Last week'])
    styler = data.style.set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index()
    total_tables['Best total scores this season'] = styler.render()

    export_tables_to_html(sport, leagues_tables, total_tables, f'{leagues[0]}.html')
