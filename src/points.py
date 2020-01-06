import datetime
from collections import defaultdict
from operator import itemgetter as _itemgetter

import pandas as pd
import numpy as np

import styling
import utils

def _get_luck_score(pairs, places):
    luck_score = {}
    for player1, player2 in pairs:
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


def _get_sorted_matchup_scores(matchup_pairs):
    scores = []
    for pair in matchup_pairs:
        scores.extend(pair)
    return sorted(scores, key=_itemgetter(1), reverse=True)


def export_matchup_stats(leagues, sport, test_mode_on=False, sleep_timeout=10):
    all_scores_dict = defaultdict(list)
    leagues_tables = defaultdict(dict)
    for league in leagues:
        luck_score = defaultdict(list)
        opp_luck_score = defaultdict(list)
        places = defaultdict(list)
        opp_places = defaultdict(list)

        today = datetime.datetime.today().date()
        this_season_begin_year = today.year if today.month > 6 else today.year - 1
        matchup = 1
        if not test_mode_on:
            matchup = -1
            schedule, _ = utils.get_league_main_info(league, sport, this_season_begin_year, sleep_timeout)
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
                all_scores_dict[(sc[0][0], league_name)].append(sc[0][1])
                all_scores_dict[(sc[1][0], league_name)].append(sc[1][1])

            matchup_scores = _get_sorted_matchup_scores(matchup_results)
            matchup_places = utils.get_places(matchup_scores)
            matchup_luck_score = _get_luck_score(matchup_results, matchup_places)
            format_lambda = lambda x: x if x % 1.0 > utils.ZERO else int(x)
            for team in matchup_luck_score:
                luck_score[team].append(format_lambda(matchup_luck_score[team]))
                places[team].append(format_lambda(matchup_places[team]))
                opp_luck_score[team].append(format_lambda(matchup_luck_score[opp_dict[team]]))
                opp_places[team].append(format_lambda(matchup_places[opp_dict[team]]))

        for team in luck_score:
            luck_score[team].extend([
                np.sum(np.array(luck_score[team]) < 0),
                np.sum(np.array(luck_score[team]) > 0),
                np.sum(luck_score[team])
            ])
            places[team].extend([
                np.sum(np.array(places[team]) / len(luck_score) > 1.0 - styling.TOP_PERC),
                np.sum(np.array(places[team]) / len(luck_score) <= styling.TOP_PERC),
                np.sum(places[team])
            ])
            opp_luck_score[team].extend([
                np.sum(np.array(opp_luck_score[team]) > 0),
                np.sum(np.array(opp_luck_score[team]) < 0),
                np.sum(opp_luck_score[team])
            ])
            opp_places[team].extend([
                np.sum(np.array(opp_places[team]) / len(luck_score) <= styling.TOP_PERC),
                np.sum(np.array(opp_places[team]) / len(luck_score) > 1.0 - styling.TOP_PERC),
                np.sum(opp_places[team])
            ])

        matchups = [w for w in range(1, matchup + 1)]
        cols = ['Team', *matchups, '&#128532;', '&#128526;', 'SUM']
        total_tables = {}

        df_luck_score = pd.DataFrame(data=list(map(utils.make_data_row, luck_score.items())), columns=cols)
        df_luck_score = df_luck_score.iloc[np.lexsort((-df_luck_score['&#128532;'], df_luck_score['SUM']))]
        styler = df_luck_score.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            applymap(styling.color_value, subset=matchups)
        leagues_tables[league_name]['Luck scores'] = styler.render()

        df_places = pd.DataFrame(data=list(map(utils.make_data_row, places.items())), columns=cols)
        df_places = df_places.iloc[np.lexsort((-df_places['&#128526;'], df_places['SUM']))]
        styler = df_places.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            apply(styling.color_place_column, subset=matchups)
        leagues_tables[league_name]['Places'] = styler.render()

        df_opp_luck_score = pd.DataFrame(data=list(map(utils.make_data_row, opp_luck_score.items())), columns=cols)
        df_opp_luck_score = df_opp_luck_score.iloc[np.lexsort((-df_opp_luck_score['&#128526;'],
                                                               df_opp_luck_score['SUM']))]
        styler = df_opp_luck_score.style.set_table_styles(utils.STYLES).\
            set_table_attributes(utils.ATTRS).hide_index().\
            applymap(styling.color_opponent_value, subset=matchups)
        leagues_tables[league_name]['Opponent luck scores'] = styler.render()

        df_opp_places = pd.DataFrame(data=list(map(utils.make_data_row, opp_places.items())), columns=cols)
        df_opp_places = df_opp_places.iloc[np.lexsort((-df_opp_places['&#128532;'], df_opp_places['SUM']))]
        styler = df_opp_places.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
            apply(styling.color_opponent_place_column, subset=matchups)
        leagues_tables[league_name]['Opponent places'] = styler.render()

    n_top = int(len(all_scores_dict) / len(leagues))
    last_matchup_scores = [(team[0], all_scores_dict[team][-1], team[1]) for team in all_scores_dict]
    df_data = sorted(last_matchup_scores, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *df_data[i]] for i in range(len(df_data))],
                        index=np.arange(1, 1 + n_top), columns=['Place', 'Team', 'Score', 'League'])
    styler = data.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index()
    total_tables['Best scores this matchup'] = styler.render()

    all_scores = []
    for team in all_scores_dict:
        team_scores = [(team[0], score, team[1], index + 1) for index, score in enumerate(all_scores_dict[team])]
        all_scores.extend(team_scores)
    df_data = sorted(all_scores, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *df_data[i]] for i in range(len(df_data))],
                        index=np.arange(1, 1 + n_top), columns=['Place', 'Team', 'Score', 'League', 'Matchup'])
    styler = data.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index()
    total_tables['Best scores this season'] = styler.render()

    total_sums = [(team[0], np.sum(scores), team[1], scores[-1]) for team, scores in all_scores_dict.items()]
    df_data = sorted(total_sums, key=_itemgetter(1), reverse=True)[:n_top]
    data = pd.DataFrame(data=[[i+1, *df_data[i]] for i in range(len(df_data))],
                        index=np.arange(1, 1 + n_top), columns=['Place', 'Team', 'Score', 'League', 'Last matchup'])
    styler = data.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index()
    total_tables['Best total scores this season'] = styler.render()

    season_str = f'{this_season_begin_year}-{str(this_season_begin_year + 1)[-2:]}'
    utils.export_tables_to_html(sport, leagues_tables, total_tables, leagues[0], season_str, matchup, test_mode_on)
