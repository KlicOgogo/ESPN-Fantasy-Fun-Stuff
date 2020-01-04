from collections import defaultdict
from operator import itemgetter as _itemgetter

from jinja2 import Template
import pandas as pd
import numpy as np

from src import styling
from src.utils import get_places, get_scoreboard_stats, ZERO


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


def display_week_stats(leagues, sport, week, sleep_timeout=10):
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
            luck_score[team].append(np.sum(luck_score[team]))
            places[team].append(np.sum(places[team]))
            opp_luck_score[team].append(np.sum(opp_luck_score[team]))
            opp_places[team].append(np.sum(opp_places[team]))

        weeks = [f'week {w}' for w in range(1, week + 1)]
        cols = weeks + ['SUM']
        styles = [
            dict(selector='td', props=[('border-left', '1px solid black'), ('border-right', '1px solid black'),
                                       ('text-align', 'right'), ('padding-left', '8px')]),
            dict(selector='td:first-child', props=[('border-left', 'none')]),
            dict(selector='td:last-child', props=[('border-right', 'none')]),
            dict(selector='th', props=[('border-left', '1px solid black'), ('border-right', '1px solid black'),
                                       ('border-bottom', '1px solid black')]),
        ]
        attrs = 'style="border-collapse: collapse; border: 1px solid black;" align= "center"'
        total_tables = {}

        df_luck_score = pd.DataFrame(data=list(luck_score.values()), index=luck_score.keys(), columns=cols)
        df_luck_score = df_luck_score.sort_values(['SUM'])
        styler = df_luck_score.style.set_table_styles(styles).set_table_attributes(attrs).\
            applymap(styling.color_value, subset=weeks)
        leagues_tables[league_name]['Luck scores'] = styler.render()

        df_places = pd.DataFrame(data=list(places.values()), index=places.keys(), columns=cols)
        df_places = df_places.sort_values(['SUM'])
        styler = df_places.style.set_table_styles(styles).set_table_attributes(attrs).\
            apply(styling.color_place_column, subset=weeks)
        leagues_tables[league_name]['Places'] = styler.render()

        df_opp_luck_score = pd.DataFrame(data=list(opp_luck_score.values()), index=opp_luck_score.keys(), columns=cols)
        df_opp_luck_score = df_opp_luck_score.sort_values(['SUM'])
        styler = df_opp_luck_score.style.set_table_styles(styles).set_table_attributes(attrs).\
            applymap(styling.color_opponent_value, subset=weeks)
        leagues_tables[league_name]['Opponent luck scores'] = styler.render()

        df_opp_places = pd.DataFrame(data=list(opp_places.values()), index=opp_places.keys(), columns=cols)
        df_opp_places = df_opp_places.sort_values(['SUM'])
        styler = df_opp_places.style.set_table_styles(styles).set_table_attributes(attrs).\
            apply(styling.color_opponent_place_column, subset=weeks)
        leagues_tables[league_name]['Opponent places'] = styler.render()

    n_top = int(len(all_scores_dict) / len(leagues))
    last_week_scores = [(team[0], all_scores_dict[team][-1], team[1]) for team in all_scores_dict]
    data = pd.DataFrame(data=sorted(last_week_scores, key=_itemgetter(1), reverse=True)[:n_top],
                        index=np.arange(1, 1 + n_top), columns=['Team', 'Score', 'League'])
    styler = data.style.set_table_styles(styles).set_table_attributes(attrs)
    total_tables['Best scores this week'] = styler.render()

    all_scores = []
    for team in all_scores_dict:
        team_scores = [(team[0], score, team[1], index + 1) for index, score in enumerate(all_scores_dict[team])]
        all_scores.extend(team_scores)
    data = pd.DataFrame(data=sorted(all_scores, key=_itemgetter(1), reverse=True)[:n_top],
                        index=np.arange(1, 1 + n_top), columns=['Team', 'Score', 'League', 'Week'])
    styler = data.style.set_table_styles(styles).set_table_attributes(attrs)
    total_tables['Best scores this season'] = styler.render()

    total_sums = [(team[0], np.sum(all_scores_dict[team]), team[1]) for team in all_scores_dict]
    data = pd.DataFrame(data=sorted(total_sums, key=_itemgetter(1), reverse=True)[:n_top],
                        index=np.arange(1, 1 + n_top), columns=['Team', 'Score', 'League'])
    styler = data.style.set_table_styles(styles).set_table_attributes(attrs)
    total_tables['Best total scores this season'] = styler.render()

    with open('template.html', 'r') as template_fp:
        template = Template(template_fp.read())
        html_str = template.render({
            'leagues': leagues_tables,
            'total_tables': total_tables
        })
        with open(f'{leagues[0]}.html', 'w') as html_fp:
            html_fp.write(html_str)
