from collections import defaultdict, Counter
from operator import itemgetter
import time

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from selenium import webdriver


from src.styling import color_extremums, color_matchup_result, color_place_column, color_value
from src.utils import get_places, get_scoreboard_stats, ZERO


NUMBERED_VALUE_COLS = {'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS', 'TP'}


def format_value(x):
    return str(x) if x % 1.0 > ZERO else str(int(x))


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


def get_category_stats(results):
    category_stats = {}
    for matchup in results:
        for team, total_score in matchup:
            category_stats[team] = [score if score % 1.0 > ZERO else int(score) for _, score in total_score]
    return category_stats


def get_diff(expected, real):
    return list(map(lambda x: float(x[0]) - float(x[1]), zip(expected.split('-'), real.split('-'))))


def _get_expected_category_probs(score_pairs, category):
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
    category_stats = get_category_stats(results)

    for i, cat in enumerate(categories):
        pairs = [(team, category_stats[team][i]) for team in category_stats]
        expected_stats = _get_expected_category_probs(pairs, cat)
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
        pairs = [(team, table[col].loc[team]) for team in table[col].index]
        pairs_sorted = sorted(pairs, key=itemgetter(1), reverse=False if col == 'TO' else True)
        places = get_places(pairs_sorted)
        for team in places:
            places_data[team].append(places[team])
    for team in places_data:
        places_data[team].append(np.sum(places_data[team]))
    return places_data


def get_team_power_dict(category_stats, categories):
    team_power = {}
    for team in category_stats:
        week_wins = Counter()
        for opp in category_stats:
            if opp == team:
                continue
            fake_matchup_res = get_matchup_result(category_stats[team], category_stats[opp], categories)
            week_wins[fake_matchup_res] += 1
        tr = np.round((week_wins['W'] + week_wins['D'] * 0.5) / (len(category_stats) - 1), 2)
        team_power[team] = tr if tr % 1.0 > ZERO else int(tr)
    return team_power


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


def display_last_week_stats(is_each_category_type, league_name, scoreboard_html_source, week_scores, week):
    week_matchups, categories = get_week_matchups(scoreboard_html_source)
    exp_score, exp_result = get_expected_score_and_result(week_matchups, categories)
    category_stats = get_category_stats(week_matchups)

    week_scores_dict = {}
    for s in week_scores:
        week_scores_dict.update(s)

    full_df = pd.DataFrame(data=list(category_stats.values()), index=category_stats.keys(), columns=categories)
    score_df = pd.DataFrame(data=list(week_scores_dict.values()), index=week_scores_dict.keys(), columns=['Score'])
    places_data = get_places_data(full_df)
    places_df = pd.DataFrame(data=list(places_data.values()), index=places_data.keys(),
                             columns=[f'{col} ' for col in categories] + ['SUM'])

    full_df = full_df.merge(score_df, how='outer', left_index=True, right_index=True)
    if is_each_category_type:
        exp_score_df = pd.DataFrame(data=list(exp_score.values()), index=exp_score.keys(), columns=['ExpScore'])
        full_df = full_df.merge(exp_score_df, how='outer', left_index=True, right_index=True)
    else:
        team_power = get_team_power_dict(category_stats, categories)
        team_power_df = pd.DataFrame(data=list(team_power.values()), index=team_power.keys(), columns=['TP'])
        full_df = full_df.merge(team_power_df, how='outer', left_index=True, right_index=True)
        exp_res_df = pd.DataFrame(data=list(exp_result.values()), index=exp_result.keys(), columns=['ER'])
        full_df = full_df.merge(exp_res_df, how='outer', left_index=True, right_index=True)
    full_df = full_df.merge(places_df, how='outer', left_index=True, right_index=True)
    full_df = full_df.sort_values(['SUM', 'PTS'])
    best_and_worst_df = pd.DataFrame(data=list(get_best_and_worst_rows(full_df)), index=['Best', 'Worst'])
    final_df = full_df.append(best_and_worst_df, sort=False)

    best_and_worst_cols = categories + ['Score', 'TP']
    if is_each_category_type:
        best_and_worst_cols = categories + ['Score', 'ExpScore']

    styles = [dict(selector='caption', props=[('text-align', 'center')])]
    final_df_styler = final_df.style.set_table_styles(styles).set_caption(f'{league_name}, week {week}').\
        apply(color_extremums, subset=pd.IndexSlice[final_df.index, best_and_worst_cols]).\
        apply(color_place_column, subset=pd.IndexSlice[full_df.index, [f'{c} ' for c in categories]])
    if not is_each_category_type:
        final_df_styler = final_df_styler.applymap(color_matchup_result, subset=pd.IndexSlice[full_df.index, ['ER']])
    display(final_df_styler)


def display_week_stats(leagues, cat_subset, sport, week, sleep_timeout=10):
    for league in leagues:
        all_scores = defaultdict(list)
        all_matchup_results = defaultdict(list)
        all_exp_scores = defaultdict(list)
        all_matchup_exp_results = defaultdict(list)
        all_matchups_won = defaultdict(list)
        
        all_matchups, soups, league_name = get_scoreboard_stats(league, sport, week, sleep_timeout, scoring='categories')
        display_last_week_stats(league in cat_subset, league_name, soups[-1], all_matchups[-1], week)

        for scores, scoreboard_html_source in zip(all_matchups, soups):
            res, cat = get_week_matchups(scoreboard_html_source)

            week_scores = get_category_stats(res)
            for team in week_scores:
                week_wins = Counter()
                for opp in week_scores:
                    if opp == team:
                        continue
                    fake_matchup_res = get_matchup_result(week_scores[team], week_scores[opp], cat)
                    week_wins[fake_matchup_res] += 1
                all_matchups_won[team].append('-'.join(map(str, [week_wins['W'], week_wins['L'], week_wins['D']])))
                
            scores_dict = {}
            opp_dict = {}
            for sc in scores:
                scores_dict.update(sc)
                opp_dict[sc[0][0]] = sc[1][0]
                opp_dict[sc[1][0]] = sc[0][0]
                
            for team in opp_dict:
                team_sc = np.array(scores_dict[team].split('-'))
                opp_sc = np.array(scores_dict[opp_dict[team]].split('-'))
                if list(team_sc[[0, 2, 1]]) > list(opp_sc[[0, 2, 1]]):
                    all_matchup_results[team].append('W')
                elif list(team_sc[[0, 2, 1]]) < list(opp_sc[[0, 2, 1]]):
                    all_matchup_results[team].append('L')
                else:
                    all_matchup_results[team].append('D')
            
            for team in scores_dict:
                all_scores[team].append(scores_dict[team])

            exp_score, exp_result = get_expected_score_and_result(res, cat)
            for team in exp_score:
                all_exp_scores[team].append(exp_score[team])
                all_matchup_exp_results[team].append(exp_result[team])

        for team in all_scores:
            stats = [np.array(list(map(float, score.split('-')))) for score in all_scores[team]]
            stats_array = np.vstack(stats)
            res = stats_array.sum(axis=0)
            all_scores[team].append('-'.join(list(map(format_value, res))))
            
        for team in all_exp_scores:
            stats = [np.array(list(map(float, score.split('-')))) for score in all_exp_scores[team]]
            stats_array = np.vstack(stats)
            res = list(map(lambda x: np.round(x, 1), stats_array.sum(axis=0)))
            all_exp_scores[team].append('-'.join(map(format_value, res)))
            
        for team in all_matchup_results:
            all_matchup_results[team].append(get_team_win_stat(all_matchup_results[team]))
            all_matchup_exp_results[team].append(get_team_win_stat(all_matchup_exp_results[team]))
        
        for team in all_matchups_won:
            stats = [np.array(list(map(float, score.split('-')))) for score in all_matchups_won[team]]
            stats_array = np.vstack(stats)
            res = list(map(lambda x: np.round(x, 1), stats_array.sum(axis=0)))
            all_matchups_won[team].append('-'.join(map(format_value, res)))
        
        table_data_dict = {team: all_exp_scores[team][len(all_exp_scores[team])-8:] for team in all_exp_scores}
        for team in table_data_dict:
            table_data_dict[team].append(all_scores[team][-1])
            table_data_dict[team].extend(get_diff(table_data_dict[team][-1], table_data_dict[team][-2]))
            
        table_win_data_dict = {team: all_matchup_exp_results[team] for team in all_matchup_exp_results}
        for team in table_win_data_dict:
            table_win_data_dict[team].append(all_matchup_results[team][-1])
            table_win_data_dict[team].extend(get_diff(table_win_data_dict[team][-1], table_win_data_dict[team][-2]))
        
        matchups_data_dict = {team: all_matchups_won[team] for team in all_matchups_won}
        for team in matchups_data_dict:
            matchups_data_dict[team].extend(map(int, matchups_data_dict[team].pop().split('-')))

        df_matchups = pd.DataFrame(data=list(matchups_data_dict.values()), index=matchups_data_dict.keys(),
                                   columns=[f'week {w}' for w in range(1, week + 1)] + ['W', 'L', 'D'])
        df_matchups = df_matchups.sort_values(['W', 'D'], ascending=False)
        to_add = pd.DataFrame(data=list(get_best_and_worst_rows(df_matchups)), index=['Best', 'Worst'])
        df_matchups = df_matchups.append(to_add, sort=False)
        styles = [dict(selector='caption', props=[('text-align', 'center')])]
        df_matchups_styler = df_matchups.style.set_table_styles(styles).set_caption(f'{league_name}').\
            apply(color_extremums, subset=[f'week {w}' for w in range(1, week + 1)])
        display(df_matchups_styler)

        if league not in cat_subset:
            df_win = pd.DataFrame(data=list(table_win_data_dict.values()), index=table_win_data_dict.keys(),
                                  columns=[f'week {i+1}' for i in range(week)] + ['Total', 'ESPN', 'WD', 'LD', 'DD'])
            df_win = df_win.sort_values(['WD', 'DD'], ascending=False)
            styles = [dict(selector='caption', props=[('text-align', 'center')])]
            df_win_styler = df_win.style.set_table_styles(styles).set_caption(f'{league_name}').\
                applymap(color_matchup_result, subset=[f'week {w}' for w in range(1, week + 1)]).\
                applymap(color_value, subset=pd.IndexSlice[table_data_dict.keys(), ['WD']])
            display(df_win_styler)

        if league in cat_subset:
            df = pd.DataFrame(data=list(table_data_dict.values()), index=table_data_dict.keys(),
                              columns=[f'week {i+1}' for i in range(week - 7, week)] + ['Total', 'ESPN', 'WD', 'LD', 'DD'])
            df = df.sort_values(['WD', 'DD'], ascending=False)
            to_add = pd.DataFrame(data=list(get_best_and_worst_rows(df)), index=['Best', 'Worst'])
            df = df.append(to_add, sort=False)
            styles = [dict(selector='caption', props=[('text-align', 'center')])]
            df_styler = df.style.set_table_styles(styles).set_caption(f'{league_name}').\
                apply(color_extremums, subset=pd.IndexSlice[df.index, [f'week {i+1}' for i in range(week-7, week)] + ['Total', 'ESPN']]).\
                applymap(color_value, subset=pd.IndexSlice[table_data_dict.keys(), ['WD']])
            display(df_styler)
