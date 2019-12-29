from collections import defaultdict, Counter
from operator import itemgetter as _itemgetter

import numpy as np
import pandas as pd

from src.styling import color_extremums, color_matchup_result, color_place_column, color_value
from src.utils import get_places, get_scoreboard_stats, ZERO


NUMBERED_VALUE_COLS = {'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS', 'TP'}


def _add_stats_sum(stats_dict):
    for team in stats_dict:
        team_stats = [np.array(list(map(float, score.split('-')))) for score in stats_dict[team]]
        team_stats_array = np.vstack(team_stats)
        stats_sum = map(lambda x: np.round(x, 1), team_stats_array.sum(axis=0))
        stats_dict[team].append('-'.join(map(_format_value, stats_sum)))


def _display_last_week_stats(is_each_category_type, scoreboard_html_source, week_scores, caption):
    week_matchups, categories = _get_week_matchups(scoreboard_html_source)
    exp_score, exp_result = _get_expected_score_and_result(week_matchups, categories)
    category_stats = _get_category_stats(week_matchups)
    places_data = _get_places_data(category_stats, categories)
    week_scores_dict = {}
    for s in week_scores:
        week_scores_dict.update(s)

    full_df = pd.DataFrame(data=list(category_stats.values()), index=category_stats.keys(), columns=categories)
    score_df = pd.DataFrame(data=list(week_scores_dict.values()), index=week_scores_dict.keys(), columns=['Score'])
    places_df = pd.DataFrame(data=list(places_data.values()), index=places_data.keys(),
                             columns=[f'{col} ' for col in categories] + ['SUM'])
    full_df = full_df.merge(score_df, how='outer', left_index=True, right_index=True)
    if is_each_category_type:
        exp_score_df = pd.DataFrame(data=list(exp_score.values()), index=exp_score.keys(), columns=['ExpScore'])
        full_df = full_df.merge(exp_score_df, how='outer', left_index=True, right_index=True)
    else:
        comparison_stats = _get_comparison_stats(category_stats, categories)
        calc_power_lambda = lambda x, y: np.round((x[0] + x[2] * 0.5) / y, 2)
        n_opponents = len(comparison_stats) - 1
        team_power = {team: calc_power_lambda(comparison_stats[team], n_opponents) for team in comparison_stats}
        team_power_df = pd.DataFrame(data=list(team_power.values()), index=team_power.keys(), columns=['TP'])
        full_df = full_df.merge(team_power_df, how='outer', left_index=True, right_index=True)
        exp_res_df = pd.DataFrame(data=list(exp_result.values()), index=exp_result.keys(), columns=['ER'])
        full_df = full_df.merge(exp_res_df, how='outer', left_index=True, right_index=True)
    full_df = full_df.merge(places_df, how='outer', left_index=True, right_index=True)
    full_df = full_df.sort_values(['SUM', 'PTS'])
    best_and_worst_df = pd.DataFrame(data=list(_get_best_and_worst_rows(full_df)), index=['Best', 'Worst'])
    final_df = full_df.append(best_and_worst_df, sort=False)

    best_and_worst_cols = categories + ['Score', 'TP']
    if is_each_category_type:
        best_and_worst_cols = categories + ['Score', 'ExpScore']

    styles = [dict(selector='caption', props=[('text-align', 'center')])]
    final_df_styler = final_df.style.set_table_styles(styles).set_caption(caption).\
        apply(color_extremums, subset=pd.IndexSlice[final_df.index, best_and_worst_cols]).\
        apply(color_place_column, subset=pd.IndexSlice[full_df.index, [f'{c} ' for c in categories]])
    if not is_each_category_type:
        final_df_styler = final_df_styler.applymap(color_matchup_result, subset=pd.IndexSlice[full_df.index, ['ER']])
    display(final_df_styler)


def _format_value(x):
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
        format_score_lambda = lambda x: '-'.join(map(_format_value, [x[i] for i in [0, 2, 1]]))
        return format_score_lambda(max_val), format_score_lambda(min_val)


def _get_best_and_worst_rows(table):
    empty_value_cols = {'ER', 'SUM', 'W', 'L', 'D', 'WD', 'LD', 'DD'} | {f'{col} ' for col in NUMBERED_VALUE_COLS}
    best = {}
    worst = {}
    for col in table.columns:
        if col in empty_value_cols:
            best[col], worst[col] = ('', '')
        else:
            best[col], worst[col] = _get_best_and_worst_values(table, col)
    return best, worst


def _get_category_stats(results):
    category_stats = {}
    for matchup in results:
        for team, total_score in matchup:
            category_stats[team] = [score if score % 1.0 > ZERO else int(score) for _, score in total_score]
    return category_stats


def _get_comparison_stats(category_stats, categories):
    comparison_stats = {}
    for team in category_stats:
        week_wins = Counter()
        for opp in category_stats:
            if opp == team:
                continue
            fake_matchup_res = _get_matchup_result(category_stats[team], category_stats[opp], categories)
            week_wins[fake_matchup_res] += 1
        comparison_stats[team] = [week_wins['W'], week_wins['L'], week_wins['D']]
    return comparison_stats


def _get_diff(expected, real):
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


def _get_expected_score_and_result(results, categories):
    res = {}
    opponents_dict = {}
    for matchup in results:
        opponents_dict[matchup[0][0]] = matchup[1][0]
        opponents_dict[matchup[1][0]] = matchup[0][0]
        res[matchup[0][0]] = np.array([0.0, 0.0, 0.0])
        res[matchup[1][0]] = np.array([0.0, 0.0, 0.0])
    category_stats = _get_category_stats(results)

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
    exp_scores = {team: '-'.join(map(lambda x: _format_value(np.round(x, 1)), res[team])) for team in res}
    return exp_scores, matchup_results


def _get_matchup_result(team_stat, opp_stat, categories):
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


def _get_places_data(category_stats, categories):
    places_data = defaultdict(list)
    for index, cat in enumerate(categories):
        pairs = [(team, category_stats[team][index]) for team in category_stats]
        pairs_sorted = sorted(pairs, key=_itemgetter(1), reverse=False if cat == 'TO' else True)
        places = get_places(pairs_sorted)
        for team in places:
            places_data[team].append(places[team])
    for team in places_data:
        places_data[team].append(np.sum(places_data[team]))
    return places_data


def _get_team_win_stat(team_stat):
    return '-'.join(map(_format_value, [team_stat.count('W'), team_stat.count('L'), team_stat.count('D')]))


def _get_week_matchups(scoreboard_html_source):
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
             (team_names[1], [(cat, float(stat)) for cat, stat in zip(categories, second_team_stats)]))
        )
    return matchups, categories


def display_week_stats(leagues, cat_subset, sport, week, sleep_timeout=10):
    for league in leagues:
        all_scores = defaultdict(list)
        all_exp_scores = defaultdict(list)
        all_matchup_results = defaultdict(list)
        all_matchup_exp_results = defaultdict(list)
        matchups_data_dict = defaultdict(list)

        all_matchups, soups, league_name = get_scoreboard_stats(league, sport, week, sleep_timeout, 'categories')
        _display_last_week_stats(league in cat_subset, soups[-1], all_matchups[-1], f'{league_name}, week {week}')

        for scores, scoreboard_html_source in zip(all_matchups, soups):
            week_matchups, categories = _get_week_matchups(scoreboard_html_source)
            exp_score, exp_result = _get_expected_score_and_result(week_matchups, categories)
            category_stats = _get_category_stats(week_matchups)
            comparison_stat = _get_comparison_stats(category_stats, categories)
            for team in comparison_stat:
                matchups_data_dict[team].append('-'.join(map(str, comparison_stat[team])))

            opp_dict = {}
            for sc in scores:
                opp_dict[sc[0][0]] = sc[1][0]
                opp_dict[sc[1][0]] = sc[0][0]
                all_scores[sc[0][0]].append(sc[0][1])
                all_scores[sc[1][0]].append(sc[1][1])
            for team in opp_dict:
                matchup_result = _get_matchup_result(category_stats[team], category_stats[opp_dict[team]], categories)
                all_matchup_results[team].append(matchup_result)
            for team in exp_score:
                all_exp_scores[team].append(exp_score[team])
                all_matchup_exp_results[team].append(exp_result[team])

        _add_stats_sum(all_scores)
        _add_stats_sum(all_exp_scores)
        _add_stats_sum(matchups_data_dict)

        for team in all_matchup_results:
            all_matchup_results[team].append(_get_team_win_stat(all_matchup_results[team]))
            all_matchup_exp_results[team].append(_get_team_win_stat(all_matchup_exp_results[team]))
        for team in matchups_data_dict:
            matchups_data_dict[team].extend(map(int, matchups_data_dict[team].pop().split('-')))

        weeks = [f'week {w}' for w in range(1, week + 1)]
        df_matchups = pd.DataFrame(data=list(matchups_data_dict.values()), index=matchups_data_dict.keys(),
                                   columns=weeks + ['W', 'L', 'D'])
        df_matchups = df_matchups.sort_values(['W', 'D'], ascending=False)
        best_and_worst_df = pd.DataFrame(data=list(_get_best_and_worst_rows(df_matchups)), index=['Best', 'Worst'])
        df_matchups = df_matchups.append(best_and_worst_df, sort=False)
        styles = [dict(selector='caption', props=[('text-align', 'center')])]
        df_matchups_styler = df_matchups.style.set_table_styles(styles).set_caption(f'{league_name}').\
            apply(color_extremums, subset=weeks)
        display(df_matchups_styler)

        if league not in cat_subset:
            table_win_data_dict = all_matchup_exp_results.copy()
            for team in table_win_data_dict:
                table_win_data_dict[team].append(all_matchup_results[team][-1])
                win_diff = _get_diff(table_win_data_dict[team][-1], table_win_data_dict[team][-2])
                table_win_data_dict[team].extend(win_diff)

            df_win = pd.DataFrame(data=list(table_win_data_dict.values()), index=table_win_data_dict.keys(),
                                  columns=weeks + ['Total', 'ESPN', 'WD', 'LD', 'DD'])
            df_win = df_win.sort_values(['WD', 'DD'], ascending=False)
            df_win_styler = df_win.style.set_table_styles(styles).set_caption(league_name).\
                applymap(color_matchup_result, subset=weeks).\
                applymap(color_value, subset=pd.IndexSlice[table_win_data_dict.keys(), ['WD']])
            display(df_win_styler)

        if league in cat_subset:
            table_data_dict = {team: all_exp_scores[team][len(all_exp_scores[team])-8:] for team in all_exp_scores}
            for team in table_data_dict:
                table_data_dict[team].append(all_scores[team][-1])
                table_data_dict[team].extend(_get_diff(table_data_dict[team][-1], table_data_dict[team][-2]))

            df_exp = pd.DataFrame(data=list(table_data_dict.values()), index=table_data_dict.keys(),
                                  columns=weeks[week - 7:] + ['Total', 'ESPN', 'WD', 'LD', 'DD'])
            df_exp = df_exp.sort_values(['WD', 'DD'], ascending=False)
            best_and_worst_df = pd.DataFrame(data=list(_get_best_and_worst_rows(df_exp)), index=['Best', 'Worst'])
            df_exp = df_exp.append(best_and_worst_df, sort=False)
            df_styler = df_exp.style.set_table_styles(styles).set_caption(league_name).\
                apply(color_extremums, subset=pd.IndexSlice[df_exp.index, weeks[week-7:] + ['Total', 'ESPN']]).\
                applymap(color_value, subset=pd.IndexSlice[table_data_dict.keys(), ['WD']])
            display(df_styler)
