import datetime
from collections import defaultdict, Counter
from operator import itemgetter
import re

import numpy as np
import pandas as pd

import styling
import utils


CATEGORY_COLS = {'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS'}


def _add_stats_sum(stats_dict, split=True):
    for team in stats_dict:
        team_stats = [np.array(list(map(float, score.split('-')))) for score in stats_dict[team]]
        team_stats_array = np.vstack(team_stats)
        stats_sum = map(lambda x: np.round(x, 1), team_stats_array.sum(axis=0))
        stats_dict[team].append('-'.join(map(_format_value, stats_sum)))


def _export_last_matchup_stats(is_each_category_type, matchup_pairs, matchup_scores, minutes, categories, is_overall, display_draw):
    category_stats = _get_category_stats(matchup_pairs)
    opp_dict = utils.get_opponent_dict(matchup_pairs)
    exp_score, exp_result = _get_expected_score_and_result(category_stats, opp_dict, categories)
    places_data = _get_places_data(category_stats, categories)
    matchup_scores_dict = {}
    for s in matchup_scores:
        matchup_scores_dict.update(s)

    teams_df = pd.DataFrame(list(map(itemgetter(2, 0) if is_overall else itemgetter(0), category_stats.keys())),
                            index=category_stats.keys(), columns=['League', 'Team'] if is_overall else ['Team'])
    stats_df = pd.DataFrame(list(category_stats.values()), index=category_stats.keys(), columns=categories)
    if minutes is None:
        full_df = teams_df.merge(stats_df, how='outer', left_index=True, right_index=True)
    else:
        minutes_df = pd.DataFrame(list(minutes.values()), index=minutes.keys(), columns=['MIN'])
        full_df = teams_df.merge(minutes_df, how='outer', left_index=True, right_index=True)
        full_df = full_df.merge(stats_df, how='outer', left_index=True, right_index=True)
    score_df = pd.DataFrame(list(matchup_scores_dict.values()), index=matchup_scores_dict.keys(), columns=['Score'])
    places_df = pd.DataFrame(list(places_data.values()), index=places_data.keys(),
                             columns=[f'{col} ' for col in categories] + ['SUM'])
    full_df = full_df.merge(score_df, how='outer', left_index=True, right_index=True)
    if is_each_category_type:
        slice_end = 3 if display_draw else 2
        exp_score_str = {team: '-'.join(map(lambda x: _format_value(np.round(x, 1)), exp_score[team][:slice_end])) for team in exp_score}
        exp_score_df = pd.DataFrame(list(exp_score_str.values()), index=exp_score_str.keys(), columns=['ExpScore'])
        full_df = full_df.merge(exp_score_df, how='outer', left_index=True, right_index=True)
    else:
        comparison_stats = _get_comparison_stats(category_stats, categories)
        calc_power_lambda = lambda x, y: np.round((x[0] + x[2] * 0.5) / y, 2)
        n_opponents = len(comparison_stats) - 1
        team_power = {team: calc_power_lambda(comparison_stats[team], n_opponents) for team in comparison_stats}
        team_power_df = pd.DataFrame(list(team_power.values()), index=team_power.keys(), columns=['TP'])
        full_df = full_df.merge(team_power_df, how='outer', left_index=True, right_index=True)
        exp_res_df = pd.DataFrame(list(exp_result.values()), index=exp_result.keys(), columns=['ER'])
        full_df = full_df.merge(exp_res_df, how='outer', left_index=True, right_index=True)
    full_df = full_df.merge(places_df, how='outer', left_index=True, right_index=True)
    full_df = full_df.iloc[np.lexsort((-full_df['PTS'], full_df['SUM']))]
    full_df = utils.add_position_column(full_df)
    best_and_worst_df = pd.DataFrame(list(_get_best_and_worst_rows(full_df)), index=['Best', 'Worst'])
    final_df = full_df.append(best_and_worst_df, sort=False)

    best_and_worst_cols = categories + ['Score', 'MIN']
    if is_each_category_type:
        best_and_worst_cols = categories + ['Score', 'ExpScore', 'MIN']

    styler = final_df.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
        apply(styling.color_extremums, subset=pd.IndexSlice[final_df.index, best_and_worst_cols]).\
        apply(styling.color_place_column, subset=pd.IndexSlice[full_df.index, [f'{c} ' for c in categories]])
    if not is_each_category_type:
        styler = styler.applymap(styling.color_pair_result, subset=pd.IndexSlice[full_df.index, ['ER']]).\
            applymap(styling.color_percentage, subset=pd.IndexSlice[full_df.index, ['TP']])
    return styler.render()


def _format_value(x):
    return str(x) if x % 1.0 > utils.ZERO else str(int(x))


def _get_best_and_worst_values(table, col):
    if col in CATEGORY_COLS | {'MIN'}:
        extremums = (table[col].max(), table[col].min())
        return extremums[::-1] if col == 'TO' else extremums
    else:
        scores_for_sort = []
        for sc in table[col]:
            sc_values = list(map(float, sc.split('-')))
            if len(sc_values) == 3:
                scores_for_sort.append([sc_values[i] for i in [0, 2, 1]])
            elif len(sc_values) == 2:
                scores_for_sort.append([sc_values[0], -sc_values[1]])
            else:
                raise Exception('Unexpected value format.')
        max_val = max(scores_for_sort)
        min_val = min(scores_for_sort)
        format_score_lambda = lambda x: '-'.join(map(_format_value, [x[i] for i in [0, 2, 1]] if len(x) == 3 else [x[0], -x[1]]))
        return format_score_lambda(max_val), format_score_lambda(min_val)


def _get_best_and_worst_rows(table):
    no_value_cols = {'Pos', '%', 'League', 'TP', 'ER', 'SUM', 'W', 'L', 'D', 'WD', 'LD', 'DD'}
    best = {}
    worst = {}
    for col in table.columns:
        if col in no_value_cols | {f'{col} ' for col in CATEGORY_COLS}:
            best[col], worst[col] = ('', '')
        elif col == 'Team':
            best[col], worst[col] = ('Best', 'Worst')
        else:
            best[col], worst[col] = _get_best_and_worst_values(table, col)
    return best, worst


def _get_category_stats(results):
    category_stats = {}
    for pair in results:
        for team, total_score in pair:
            category_stats[team] = [score if score % 1.0 > utils.ZERO else int(score) for _, score in total_score]
    return category_stats


def _get_comparison_stats(category_stats, categories):
    comparison_stats = {}
    for team in category_stats:
        matchup_wins = Counter()
        for opp in category_stats:
            if opp == team:
                continue
            fake_pair_res = _get_pair_result(category_stats[team], category_stats[opp], categories)
            matchup_wins[fake_pair_res] += 1
        comparison_stats[team] = [matchup_wins['W'], matchup_wins['L'], matchup_wins['D']]
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


def _get_expected_score_and_result(category_stats, opponents_dict, categories):
    res = {team: np.array([0.0, 0.0, 0.0]) for team in category_stats}
    for i, cat in enumerate(categories):
        pairs = [(team, category_stats[team][i]) for team in category_stats]
        expected_stats = _get_expected_category_probs(pairs, cat)
        for team in expected_stats:
            concatted = np.vstack((expected_stats[team], res[team]))
            res[team] = concatted.sum(axis=0)
    pair_results = {}
    for team in opponents_dict:
        if list(res[team][[0, 2, 1]]) > list(res[opponents_dict[team]][[0, 2, 1]]):
            pair_results[team] = 'W'
        elif list(res[team][[0, 2, 1]]) < list(res[opponents_dict[team]][[0, 2, 1]]):
            pair_results[team] = 'L'
        else:
            pair_results[team] = 'D'
    return res, pair_results


def _get_matchup_pairs(scoreboard_html, league_name, league_id):
    pairs_html = scoreboard_html.findAll('div', {'Scoreboard__Row'})
    pairs = []
    for m in pairs_html:
        opponents = m.findAll('li', 'ScoreboardScoreCell__Item')
        team_names = [o.findAll('div', {'class': 'ScoreCell__TeamName'})[0].text for o in opponents]
        team_ids = [re.findall(r'teamId=(\d+)', o.findAll('a', {'class': 'truncate'})[0]['href'])[0] for o in opponents]
        teams = [(team_name, team_id, league_name, league_id) for team_name, team_id in zip(team_names, team_ids)]

        rows = m.findAll('tr', {'Table2__tr'})
        categories = [header.text for header in rows[0].findAll('th', {'Table2__th'})[1:]]
        first_team_stats = [data.text for data in rows[1].findAll('td', {'Table2__td'})[1:]]
        second_team_stats = [data.text for data in rows[2].findAll('td', {'Table2__td'})[1:]]

        pairs.append(
            ((teams[0], [(cat, float(stat)) for cat, stat in zip(categories, first_team_stats)]),
             (teams[1], [(cat, float(stat)) for cat, stat in zip(categories, second_team_stats)]))
        )
    return pairs, categories


def _get_pair_result(team_stat, opp_stat, categories):
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
        pairs_sorted = sorted(pairs, key=itemgetter(1), reverse=False if cat == 'TO' else True)
        places = utils.get_places(pairs_sorted)
        for team in places:
            places_data[team].append(places[team])
    for team in places_data:
        places_data[team].append(np.sum(places_data[team]))
    return places_data


def _get_team_win_stat(team_stat):
    return '-'.join(map(_format_value, [team_stat.count('W'), team_stat.count('L'), team_stat.count('D')]))


def _is_each_category_type(scoreboard_html, matchup):
    records = []
    matchups_html = scoreboard_html.findAll('div', {'Scoreboard__Row'})
    for m in matchups_html:
        opponents = m.findAll('li', 'ScoreboardScoreCell__Item')
        for o in opponents:
            record, place = o.findAll('span', 'ScoreboardScoreCell__Record')[0].text.split(',')
            records.append(record.strip())
    record_sums = np.array(list(map(lambda x: np.sum(list(map(int, x.split('-')))), records)))
    is_most_categories_count = np.sum(record_sums == matchup)
    # return False
    return is_most_categories_count == 0


def export_matchup_stats(leagues_tuple, sport, github_login, test_mode_on=False, sleep_timeout=10):
    if len(leagues_tuple) == 1:
        leagues, tiebreaker = leagues_tuple[0], 'NO'
    elif len(leagues_tuple) == 2:
        leagues, tiebreaker = leagues_tuple
    else:
        raise Exception('Wrong config: leagues tuple must contain 1 or 2 elements.')
    leagues_tables = defaultdict(dict)
    overall_minutes_last_matchup = None if test_mode_on else {}
    overall_pairs_last_matchup = []
    overall_scores_last_matchup = []
    for league in leagues:
        all_scores = defaultdict(list)
        all_exp_scores = defaultdict(list)
        all_pair_results = defaultdict(list)
        all_pair_exp_results = defaultdict(list)
        comparisons_data_dict = defaultdict(list)
        h2h_comparisons = defaultdict(lambda: defaultdict(Counter))

        today = datetime.datetime.today().date()
        season_start_year = today.year if today.month > 6 else today.year - 1
        schedule = utils.get_league_schedule(league, sport, season_start_year, sleep_timeout)
        real_matchup = utils.find_proper_matchup(schedule)
        if real_matchup == -1 and not test_mode_on:
            return
        matchup = 1 if test_mode_on else real_matchup

        all_pairs, soups, league_name = utils.get_scoreboard_stats(league, sport, matchup, sleep_timeout, 'categories')
        is_each_category_type = _is_each_category_type(soups[-1], real_matchup)
        minutes = None
        if not test_mode_on:
            teams = []
            for pair in all_pairs[-1]:
                teams.append((pair[0][0], pair[1][0]))
            scoring_period_id = (schedule[real_matchup][0] - schedule[1][0]).days + 1
            minutes = utils.get_minutes(league, matchup, teams,
                                        scoring_period_id, season_start_year + 1, sleep_timeout)
            overall_minutes_last_matchup.update(minutes)

        tables_dict = leagues_tables[league_name]
        display_draw = len(all_pairs[-1][0][0][1].split('-')) == 3
        matchup_pairs, categories = _get_matchup_pairs(soups[-1], league_name, league)
        tables_dict['Past matchup stats'] = _export_last_matchup_stats(is_each_category_type, matchup_pairs,
                                                                       all_pairs[-1], minutes, categories, False, display_draw)
        overall_pairs_last_matchup.extend(matchup_pairs)
        overall_scores_last_matchup.extend(all_pairs[-1])

        for scores, scoreboard_html in zip(all_pairs, soups):
            matchup_pairs, categories = _get_matchup_pairs(scoreboard_html, league_name, league)
            category_stats = _get_category_stats(matchup_pairs)
            opp_dict = utils.get_opponent_dict(scores)
            exp_score, exp_result = _get_expected_score_and_result(category_stats, opp_dict, categories)
            comparison_stat = _get_comparison_stats(category_stats, categories)
            for team in comparison_stat:
                comparisons_data_dict[team].append('-'.join(map(str, comparison_stat[team])))

            for sc in scores:
                all_scores[sc[0][0]].append(sc[0][1])
                all_scores[sc[1][0]].append(sc[1][1])
            for team in opp_dict:
                pair_result = _get_pair_result(category_stats[team], category_stats[opp_dict[team]], categories)
                all_pair_results[team].append(pair_result)
            for team in exp_score:
                all_exp_scores[team].append(exp_score[team])
                all_pair_exp_results[team].append(exp_result[team])

            for team in category_stats:
                for opp in category_stats:
                    if team == opp:
                        continue
                    h2h_res = _get_pair_result(category_stats[team], category_stats[opp], categories)
                    h2h_comparisons[team][opp][h2h_res] += 1

        _add_stats_sum(all_scores)
        _add_stats_sum(comparisons_data_dict)
        for team in all_exp_scores:
            team_stats_array = np.vstack(all_exp_scores[team])
            all_exp_scores[team].append(team_stats_array.sum(axis=0))

        for team in all_pair_results:
            all_pair_results[team].append(_get_team_win_stat(all_pair_results[team]))
            all_pair_exp_results[team].append(_get_team_win_stat(all_pair_exp_results[team]))
        for team in comparisons_data_dict:
            total_comparison_stat = list(map(int, comparisons_data_dict[team].pop().split('-')))
            comparisons_data_dict[team].extend(total_comparison_stat)
            team_power = np.sum(np.array(total_comparison_stat) * np.array([1.0, 0.0, 0.5]))
            team_power_normalized = team_power / np.sum(total_comparison_stat)
            comparisons_data_dict[team].append(np.round(team_power_normalized, 2))

        matchups = [m for m in range(1, matchup + 1)]
        teams_df = pd.DataFrame(data=list(map(lambda x: x[0], all_scores.keys())),
                                index=all_scores.keys(), columns=['Team'])

        df_pairs = pd.DataFrame(data=list(comparisons_data_dict.values()), index=comparisons_data_dict.keys(),
                                columns=[*matchups, 'W', 'L', 'D', '%'])
        df_pairs = teams_df.merge(df_pairs, how='outer', left_index=True, right_index=True)
        df_pairs = df_pairs.iloc[np.lexsort((-df_pairs['W'], -df_pairs['W'] - df_pairs['D'] * 0.5))]
        df_pairs = utils.add_position_column(df_pairs)
        best_and_worst_df = pd.DataFrame(data=list(_get_best_and_worst_rows(df_pairs)), index=['Best', 'Worst'])
        df_pairs_final = df_pairs.append(best_and_worst_df, sort=False)
        df_pairs_styler = df_pairs_final.style.set_table_styles(utils.STYLES).\
            set_table_attributes(utils.ATTRS).hide_index().\
            apply(styling.color_extremums, subset=matchups).\
            applymap(styling.color_percentage, subset=pd.IndexSlice[df_pairs.index, ['%']])
        tables_dict['Pairwise comparisons h2h'] = utils.render_h2h_table(h2h_comparisons)
        tables_dict['Pairwise comparisons by matchup'] = df_pairs_styler.render()

        if not is_each_category_type:
            table_win_data_dict = all_pair_exp_results.copy()
            for team in table_win_data_dict:
                table_win_data_dict[team].append(all_pair_results[team][-1])
                win_diff = _get_diff(table_win_data_dict[team][-1], table_win_data_dict[team][-2])
                table_win_data_dict[team].extend(win_diff)

            df_win = pd.DataFrame(data=list(table_win_data_dict.values()), index=table_win_data_dict.keys(),
                                  columns=[*matchups, 'Total', 'ESPN', 'WD', 'LD', 'DD'])
            df_win = teams_df.merge(df_win, how='outer', left_index=True, right_index=True)
            df_win = df_win.iloc[np.lexsort((-df_win['WD'], -df_win['WD'] - df_win['DD'] * 0.5))]
            df_win = utils.add_position_column(df_win)
            df_win_styler = df_win.style.set_table_styles(utils.STYLES).\
                set_table_attributes(utils.ATTRS).hide_index().\
                applymap(styling.color_pair_result, subset=matchups).\
                applymap(styling.color_value, subset=pd.IndexSlice[list(table_win_data_dict.keys()), ['WD']])
            tables_dict['Expected matchup win stats'] = df_win_styler.render()

        if is_each_category_type:
            for team in all_exp_scores:
                slice_end = 3 if display_draw else 2
                real_scores = np.array(list(map(float, all_scores[team][-1].split('-'))))
                n_draw = matchup * len(categories) - np.sum(real_scores[:2])
                real_scores = real_scores if display_draw else np.array([*real_scores, n_draw])
                all_exp_scores[team].append(real_scores)
                all_exp_scores[team].extend(all_exp_scores[team][-1] - all_exp_scores[team][-2])
                for i in range(len(all_exp_scores[team]) - 3):
                    all_exp_scores[team][i] = '-'.join(map(lambda x: _format_value(np.round(x, 1)), all_exp_scores[team][i][:slice_end]))
                for i in range(len(all_exp_scores[team]) - 3, len(all_exp_scores[team])):
                    all_exp_scores[team][i] = np.round(all_exp_scores[team][i], 1)

            df_exp = pd.DataFrame(data=list(all_exp_scores.values()), index=all_exp_scores.keys(),
                                  columns=[*matchups, 'Total', 'ESPN', 'WD', 'LD', 'DD'])
            df_exp = teams_df.merge(df_exp, how='outer', left_index=True, right_index=True)
            df_exp = df_exp.iloc[np.lexsort((-df_exp['WD'], -df_exp['WD'] - df_exp['DD'] * 0.5))]
            df_exp = utils.add_position_column(df_exp)
            best_and_worst_df = pd.DataFrame(data=list(_get_best_and_worst_rows(df_exp)), index=['Best', 'Worst'])
            df_exp = df_exp.append(best_and_worst_df, sort=False)
            df_styler = df_exp.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
                applymap(styling.color_value, subset=pd.IndexSlice[list(all_exp_scores.keys()), ['WD']]).\
                apply(styling.color_extremums, subset=pd.IndexSlice[df_exp.index, matchups + ['Total', 'ESPN']])
            tables_dict['Expected category win stats'] = df_styler.render()

    overall_tables = {}
    if len(leagues) > 1:
        overall_tables['Past matchup overall stats'] = _export_last_matchup_stats(is_each_category_type,
            overall_pairs_last_matchup, overall_scores_last_matchup, overall_minutes_last_matchup, categories, True, display_draw)

    season_str = f'{season_start_year}-{str(season_start_year + 1)[-2:]}'
    utils.export_tables_to_html(sport, leagues_tables, overall_tables,
                                leagues[0], season_str, matchup, github_login, schedule, test_mode_on)
