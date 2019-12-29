from collections import defaultdict
from operator import itemgetter as _itemgetter

import pandas as pd
import numpy as np

from src import styling, utils


def get_luck_score(matchups, places):
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


def get_sorted_week_scores(week_matchups):
    scores = []
    for matchup in week_matchups:
        scores.extend(matchup)
    return sorted(scores, key=_itemgetter(1), reverse=True)


def display_week_stats(leagues, sport, n_weeks, sleep_timeout=10):
    all_scores_dict = defaultdict(list)
    for league in leagues:
        luck_score = defaultdict(list)
        opp_luck_score = defaultdict(list)
        places = defaultdict(list)
        opp_places = defaultdict(list)

        all_matchups, _, league_name = utils.get_scoreboard_stats(league, sport, n_weeks, sleep_timeout)
        for week_results in all_matchups:
            opp_dict = {}
            for sc in week_results:
                opp_dict[sc[0][0]] = sc[1][0]
                opp_dict[sc[1][0]] = sc[0][0]
                all_scores_dict[(sc[0][0], league_name)].append(sc[0][1])
                all_scores_dict[(sc[1][0], league_name)].append(sc[1][1])

            week_scores = get_sorted_week_scores(week_results)
            week_places = utils.get_places(week_scores)
            week_luck_score = get_luck_score(week_results, week_places)
            format_lambda = lambda x: x if x % 1.0 > utils.ZERO else int(x)
            for team in week_luck_score:
                luck_score[team].append(format_lambda(week_luck_score[team]))
                places[team].append(format_lambda(week_places[team]))
                opp_luck_score[team].append(format_lambda(week_luck_score[opp_dict[team]]))
                opp_places[team].append(format_lambda(week_places[opp_dict[team]]))

        for team in luck_score:
            luck_score[team].append(np.sum(luck_score[team]))
            places[team].append(np.sum(places[team]))
            opp_luck_score[team].append(np.sum(opp_luck_score[team]))
            opp_places[team].append(np.sum(opp_places[team]))

        weeks = [f'Week {i+1}' for i in range(n_weeks)]
        cols = weeks + ['SUM']
        styles = [dict(selector='caption', props=[('text-align', 'center')])]

        df_luck_score = pd.DataFrame(data=list(luck_score.values()), index=luck_score.keys(), columns=cols)
        df_luck_score = df_luck_score.sort_values(['SUM'])
        display(df_luck_score.style.set_table_styles(styles).\
                applymap(styling.color_value, subset=weeks).\
                set_caption(f'{league_name}: luck scores'))

        df_places = pd.DataFrame(data=list(places.values()), index=places.keys(), columns=cols)
        df_places = df_places.sort_values(['SUM'])
        display(df_places.style.set_table_styles(styles).\
                apply(styling.color_place_column, subset=weeks).\
                set_caption(f'{league_name}: places'))

        df_opp_luck_score = pd.DataFrame(data=list(opp_luck_score.values()), index=opp_luck_score.keys(), columns=cols)
        df_opp_luck_score = df_opp_luck_score.sort_values(['SUM'])
        display(df_opp_luck_score.style.set_table_styles(styles).\
                applymap(styling.color_opponent_value, subset=weeks).\
                set_caption(f'{league_name}: opponent luck scores'))

        df_opp_places = pd.DataFrame(data=list(opp_places.values()), index=opp_places.keys(), columns=cols)
        df_opp_places = df_opp_places.sort_values(['SUM'])
        display(df_opp_places.style.set_table_styles(styles).\
                apply(styling.color_opponent_place_column, subset=weeks).\
                set_caption(f'{league_name}: opponent places'))

    n_top = int(len(all_scores_dict) / len(leagues))
    last_week_scores = [(team[0], all_scores_dict[team][-1], team[1]) for team in all_scores_dict]
    data = pd.DataFrame(data=sorted(last_week_scores, key=_itemgetter(1), reverse=True)[:n_top],
                        index=np.arange(1, 1 + n_top), columns=['Team', 'Score', 'League'])
    display(data.style.set_table_styles(styles).set_caption('Best scores this week'))

    all_scores = []
    for team in all_scores_dict:
        team_scores = [(team[0], score, team[1], index + 1) for index, score in enumerate(all_scores_dict[team])]
        all_scores.extend(team_scores)
    data = pd.DataFrame(data=sorted(all_scores, key=_itemgetter(1), reverse=True)[:n_top],
                        index=np.arange(1, 1 + n_top), columns=['Team', 'Score', 'League', 'Week'])
    display(data.style.set_table_styles(styles).set_caption('Best scores this season'))

    total_sums = [(team[0], np.sum(all_scores_dict[team]), team[1]) for team in all_scores_dict]
    data = pd.DataFrame(data=sorted(total_sums, key=_itemgetter(1), reverse=True)[:n_top],
                        index=np.arange(1, 1 + n_top), columns=['Team', 'Score', 'League'])
    display(data.style.set_table_styles(styles).set_caption('Best total scores this season'))
