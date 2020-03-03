import datetime
from collections import defaultdict, Counter
import copy
from operator import itemgetter
import re

import numpy as np
import pandas as pd

import html_utils
import styling
import utils


CATEGORY_COLS = {'FG%', 'FT%', '3PM', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PTS'}


def _calc_power_coeff(x):
    return x[0] + x[2] * 0.5


def _export_past_matchup_stats(each_category_type_flag, categories, matchup_pairs, matchup_scores,
        minutes, tiebreaker, is_overall, display_draw_flag):
    category_stats = _get_category_stats(matchup_pairs)
    opp_dict = utils.get_opponent_dict(matchup_pairs)
    exp_score, exp_result = _get_expected_score_and_result(category_stats, opp_dict, categories, tiebreaker)

    df_teams = pd.DataFrame(list(map(itemgetter(2, 0) if is_overall else itemgetter(0), category_stats.keys())),
                            index=category_stats.keys(), columns=['League', 'Team'] if is_overall else ['Team'])
    df_stats = pd.DataFrame(list(category_stats.values()), index=category_stats.keys(), columns=categories)
    if minutes is None:
        df = df_teams.merge(df_stats, how='outer', left_index=True, right_index=True)
    else:
        df_minutes = pd.DataFrame(list(minutes.values()), index=minutes.keys(), columns=['MIN'])
        df = df_teams.merge(df_minutes, how='outer', left_index=True, right_index=True)
        df = df.merge(df_stats, how='outer', left_index=True, right_index=True)
    matchup_scores_dict = {}
    for s in matchup_scores:
        matchup_scores_dict.update(s)
    df_score = pd.DataFrame(list(matchup_scores_dict.values()), index=matchup_scores_dict.keys(), columns=['Score'])
    df = df.merge(df_score, how='outer', left_index=True, right_index=True)
    if each_category_type_flag:
        slice_end = 3 if display_draw_flag else 2
        exp_score_str = {}
        for team in exp_score:
            exp_score_str[team] = '-'.join(map(lambda x: _format_value(np.round(x, 1)), exp_score[team][:slice_end]))
        exp_score_df = pd.DataFrame(list(exp_score_str.values()), index=exp_score_str.keys(), columns=['ExpScore'])
        df = df.merge(exp_score_df, how='outer', left_index=True, right_index=True)
    else:
        comparisons = _get_comparison_stats(category_stats, categories, tiebreaker)
        n_opponents = len(comparisons) - 1
        team_power = {team: np.round(_calc_power_coeff(comparisons[team]) / n_opponents, 2) for team in comparisons}
        df_tp = pd.DataFrame(list(team_power.values()), index=team_power.keys(), columns=['TP'])
        df = df.merge(df_tp, how='outer', left_index=True, right_index=True)
        df_exp = pd.DataFrame(list(exp_result.values()), index=exp_result.keys(), columns=['ER'])
        df = df.merge(df_exp, how='outer', left_index=True, right_index=True)
    places_data = _get_places_data(category_stats, categories)
    places_cols = [f'{col} ' for col in categories]
    df_places = pd.DataFrame(list(places_data.values()), index=places_data.keys(), columns=places_cols + ['SUM'])
    df = df.merge(df_places, how='outer', left_index=True, right_index=True)
    df = df.iloc[np.lexsort((-df['PTS'], df['SUM']))]
    df = utils.add_position_column(df)
    df_extremums = pd.DataFrame(list(_get_best_and_worst_rows(df)), index=['Best', 'Worst'])
    df = df.append(df_extremums, sort=False)

    extremum_cols = categories + ['Score']
    if each_category_type_flag:
        extremum_cols.append('ExpScore')
    if minutes is not None:
        extremum_cols.append('MIN')

    num_cols = set(df.columns) - {'Team', 'League', 'Score', 'ER', 'ExpScore'}
    styler = df.style.format('{:g}', subset=pd.IndexSlice[df_teams.index, num_cols - {*categories, 'MIN'}]).\
        format('{:g}', subset=pd.IndexSlice[df.index, categories]).\
        set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
        apply(styling.color_extremums, subset=pd.IndexSlice[df.index, extremum_cols]).\
        apply(styling.color_place_column, subset=pd.IndexSlice[df_teams.index, places_cols])
    if not each_category_type_flag:
        styler = styler.applymap(styling.color_pair_result, subset=pd.IndexSlice[df_teams.index, ['ER']]).\
            applymap(styling.color_percentage, subset=pd.IndexSlice[df_teams.index, ['TP']])
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
                scores_for_sort.append([_calc_power_coeff(sc_values), *[sc_values[i] for i in [0, 2, 1]]])
            elif len(sc_values) == 2:
                scores_for_sort.append([_calc_power_coeff([*sc_values, -np.sum(sc_values)]), *sc_values])
            else:
                raise Exception('Unexpected value format.')
        max_val = max(scores_for_sort)
        min_val = min(scores_for_sort)
        score_normalizer = lambda x: [x[i] for i in [1, 3, 2]] if len(x) == 4 else x[1:]
        score_formatter = lambda x: '-'.join(map(_format_value, score_normalizer(x)))
        return score_formatter(max_val), score_formatter(min_val)


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
            category_stats[team] = [score for _, score in total_score]
    return category_stats


def _get_comparison_stats(category_stats, categories, tiebreaker):
    comparison_stats = {}
    for team in category_stats:
        matchup_wins = Counter()
        for opp in category_stats:
            if opp == team:
                continue
            fake_pair_res = _get_pair_result(category_stats[team], category_stats[opp], categories, tiebreaker)
            matchup_wins[fake_pair_res] += 1
        comparison_stats[team] = [matchup_wins['W'], matchup_wins['L'], matchup_wins['D']]
    return comparison_stats


def _get_expected_category_probs(score_pairs, category):
    scores = np.array([score for _, score in score_pairs])
    result = {}
    for team, sc in score_pairs:
        counts = np.array([np.sum(scores < sc), np.sum(scores > sc), np.sum(scores == sc) - 1])
        result[team] = counts[[1, 0, 2]] / (len(scores) - 1) if category == 'TO' else counts / (len(scores) - 1)
    return result


def _get_expected_score_and_result(category_stats, opponents_dict, categories, tiebreaker):
    res = {team: np.array([0.0, 0.0, 0.0]) for team in category_stats}
    tiebreaker_stats = copy.deepcopy(res)
    for i, cat in enumerate(categories):
        pairs = [(team, category_stats[team][i]) for team in category_stats]
        expected_stats = _get_expected_category_probs(pairs, cat)
        if cat == tiebreaker:
            tiebreaker_stats = copy.deepcopy(expected_stats)
        for team in expected_stats:
            res[team] += expected_stats[team]
    pair_results = {}
    for team in opponents_dict:
        if list(res[team][[0, 2, 1]]) > list(res[opponents_dict[team]][[0, 2, 1]]):
            pair_results[team] = 'W'
        elif list(res[team][[0, 2, 1]]) < list(res[opponents_dict[team]][[0, 2, 1]]):
            pair_results[team] = 'L'
        else:
            if list(tiebreaker_stats[team][[0, 2, 1]]) > list(tiebreaker_stats[opponents_dict[team]][[0, 2, 1]]):
                pair_results[team] = 'W'
            elif list(tiebreaker_stats[team][[0, 2, 1]]) < list(tiebreaker_stats[opponents_dict[team]][[0, 2, 1]]):
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


def _get_pair_result(team_stat, opp_stat, categories, tiebreaker):
    win_count = 0
    lose_count = 0
    for index, cat in enumerate(categories):
        if team_stat[index] > opp_stat[index]:
            lose_count += (cat == 'TO') * ((tiebreaker == cat) + 2) / 2
            win_count += (cat != 'TO') * ((tiebreaker == cat) + 2) / 2
        elif team_stat[index] < opp_stat[index]:
            lose_count += (cat != 'TO') * ((tiebreaker == cat) + 2) / 2
            win_count += (cat == 'TO')  * ((tiebreaker == cat) + 2) / 2
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


def _get_league_tables_data(league, league_name, all_pairs, soups, tiebreaker):
    category_record = {}
    expected_category_stats = defaultdict(list)
    win_record = defaultdict(Counter)
    expected_win_stats = defaultdict(list)
    comparisons_matchup = defaultdict(list)
    comparisons_h2h = defaultdict(lambda: defaultdict(Counter))
    for scores, scoreboard_html in zip(all_pairs, soups):
        for sc in scores:
            for i in range(len(sc)):
                if sc[i][0] not in category_record:
                    category_record[sc[i][0]] = np.array(list(map(float, sc[i][1].split('-'))))
                else:
                    category_record[sc[i][0]] += np.array(list(map(float, sc[i][1].split('-'))))

        matchup_pairs, categories = _get_matchup_pairs(scoreboard_html, league_name, league)
        category_stats = _get_category_stats(matchup_pairs)
        for team in category_stats:
            for opp in category_stats:
                if team == opp:
                    continue
                h2h_res = _get_pair_result(category_stats[team], category_stats[opp], categories, tiebreaker)
                comparisons_h2h[team][opp][h2h_res] += 1

        opp_dict = utils.get_opponent_dict(scores)
        for team in opp_dict:
            result = _get_pair_result(category_stats[team], category_stats[opp_dict[team]], categories, tiebreaker)
            win_record[team][result] += 1

        exp_score, exp_result = _get_expected_score_and_result(category_stats, opp_dict, categories, tiebreaker)
        for team in exp_score:
            expected_category_stats[team].append(exp_score[team])
            expected_win_stats[team].append(exp_result[team])

        comparison_stat = _get_comparison_stats(category_stats, categories, tiebreaker)
        for team in comparison_stat:
            comparisons_matchup[team].append('-'.join(map(str, comparison_stat[team])))

    category_stats = (category_record, expected_category_stats)
    win_stats = (win_record, expected_win_stats)
    comparisons_stats = (comparisons_matchup, comparisons_h2h)
    return category_stats, win_stats, comparisons_stats


def _is_each_category_type(scoreboard_html, matchup):
    records = []
    matchups_html = scoreboard_html.findAll('div', {'Scoreboard__Row'})
    for m in matchups_html:
        opponents = m.findAll('li', 'ScoreboardScoreCell__Item')
        for o in opponents:
            record, _ = o.findAll('span', 'ScoreboardScoreCell__Record')[0].text.split(',')
            records.append(record.strip())
    record_sums = np.array(list(map(lambda x: np.sum(list(map(int, x.split('-')))), records)))
    return np.sum(record_sums == matchup) == 0


def _render_category_win_stats_table(category_stats, matchup, n_categories, display_draw_flag):
    category_record, expected_category_stats = category_stats
    df_data = copy.deepcopy(expected_category_stats)
    for team in df_data:
        team_stats_array = np.vstack(df_data[team])
        df_data[team].append(team_stats_array.sum(axis=0))
        n_draw = matchup * n_categories - np.sum(category_record[team][:2])
        record_with_draw = category_record[team] if display_draw_flag else np.array([*category_record[team], n_draw])
        df_data[team].append(record_with_draw)
        df_data[team].extend(map(lambda x: np.round(x, 1), df_data[team][-1] - df_data[team][-2]))
        slice_end = 3 if display_draw_flag else 2
        for i in range(len(df_data[team]) - 3):
            df_data[team][i] = '-'.join(map(lambda x: _format_value(np.round(x, 1)), df_data[team][i][:slice_end]))

    df_teams = pd.DataFrame(list(map(itemgetter(0), df_data.keys())), index=df_data.keys(), columns=['Team'])
    matchups = np.arange(1, matchup + 1)
    df = pd.DataFrame(list(df_data.values()), index=df_data.keys(),
                      columns=[*matchups, 'Total', 'ESPN', 'WD', 'LD', 'DD'])
    df = df_teams.merge(df, how='outer', left_index=True, right_index=True)
    df = df.iloc[np.lexsort((-df['WD'], -df['WD'] - df['DD'] * 0.5))]
    df = utils.add_position_column(df)
    df_extremums = pd.DataFrame(list(_get_best_and_worst_rows(df)), index=['Best', 'Worst'])
    df = df.append(df_extremums, sort=False)
    styler = df.style.format('{:g}', subset=pd.IndexSlice[list(expected_category_stats.keys()), ['DD', 'WD', 'LD']]).\
        set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
        applymap(styling.color_value, subset=pd.IndexSlice[list(expected_category_stats.keys()), ['WD']]).\
        apply(styling.color_extremums, subset=pd.IndexSlice[df.index, [*matchups, 'Total', 'ESPN']])
    return styler.render()


def _render_matchup_comparisons_table(comparisons, matchups):
    df_data = copy.deepcopy(comparisons)
    for team in df_data:
        team_stats = [np.array(list(map(int, score.split('-')))) for score in df_data[team]]
        comparisons_sum = np.vstack(team_stats).sum(axis=0)
        df_data[team].extend(comparisons_sum)
        team_power = np.sum(comparisons_sum * np.array([1.0, 0.0, 0.5]))
        team_power_norm = team_power / np.sum(comparisons_sum)
        df_data[team].append(np.round(team_power_norm, 2))
    df_teams = pd.DataFrame(list(map(itemgetter(0), df_data.keys())), index=df_data.keys(), columns=['Team'])
    df = pd.DataFrame(list(df_data.values()), index=df_data.keys(), columns=[*matchups, 'W', 'L', 'D', '%'])
    df = df_teams.merge(df, how='outer', left_index=True, right_index=True)
    df = df.iloc[np.lexsort((-df['W'], -df['W'] - df['D'] * 0.5))]
    df = utils.add_position_column(df)
    df_extremums = pd.DataFrame(list(_get_best_and_worst_rows(df)), index=['Best', 'Worst'])
    df = df.append(df_extremums, sort=False)
    styler = df.style.format('{:g}', subset=pd.IndexSlice[list(df_data.keys()), ['%']]).\
        set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
        apply(styling.color_extremums, subset=matchups).\
        applymap(styling.color_percentage, subset=pd.IndexSlice[list(df_data.keys()), ['%']])
    return styler.render()


def _render_matchup_win_stats_table(win_stats, matchups):
    win_record, expected_win_stats = win_stats
    df_data = copy.deepcopy(expected_win_stats)
    for team in df_data:
        res_order = ['W', 'L', 'D']
        expected_record = Counter(df_data[team])
        expected_record_str = '-'.join(map(_format_value, [expected_record[res] for res in res_order]))
        df_data[team].append(expected_record_str)
        win_record_str = '-'.join(map(_format_value, [win_record[team][res] for res in res_order]))
        df_data[team].append(win_record_str)
        df_data[team].extend([win_record[team][res] - expected_record[res] for res in res_order])
    df_teams = pd.DataFrame(list(map(itemgetter(0), df_data.keys())), index=df_data.keys(), columns=['Team'])
    df = pd.DataFrame(list(df_data.values()), index=df_data.keys(),
                      columns=[*matchups, 'Total', 'ESPN', 'WD', 'LD', 'DD'])
    df = df_teams.merge(df, how='outer', left_index=True, right_index=True)
    df = df.iloc[np.lexsort((-df['WD'], -df['WD'] - df['DD'] * 0.5))]
    df = utils.add_position_column(df)
    styler = df.style.set_table_styles(utils.STYLES).set_table_attributes(utils.ATTRS).hide_index().\
        applymap(styling.color_pair_result, subset=matchups).\
        applymap(styling.color_value, subset=pd.IndexSlice[list(expected_win_stats.keys()), ['WD']])
    return styler.render()


def export_matchup_stats(leagues, sport, tiebreaker, github_login, test_mode_on=False, sleep_timeout=10):
    leagues_tables = defaultdict(dict)
    past_matchup_minutes = None if test_mode_on or sport.lower() != 'basketball' else {}
    past_matchup_pairs = []
    past_matchup_scores = []
    for league in leagues:
        today = datetime.datetime.today().date()
        season_start_year = today.year if today.month > 6 else today.year - 1
        schedule = html_utils.get_league_schedule(league, sport, season_start_year, sleep_timeout)
        real_matchup = utils.find_proper_matchup(schedule)
        if real_matchup == -1 and not test_mode_on:
            return
        matchup = 1 if test_mode_on else real_matchup

        all_pairs, soups, league_name = html_utils.get_scoreboard_stats(league, sport, matchup, sleep_timeout)
        past_matchup_scores.extend(all_pairs[-1])
        each_category_type_flag = _is_each_category_type(soups[-1], real_matchup)
        minutes = None
        if not test_mode_on and sport.lower() == 'basketball':
            teams = []
            for pair in all_pairs[-1]:
                teams.append((pair[0][0], pair[1][0]))
            scoring_period_id = (schedule[real_matchup][0] - schedule[1][0]).days + 1
            minutes = html_utils.get_minutes(league, matchup, teams,
                                             scoring_period_id, season_start_year + 1, sleep_timeout)
            past_matchup_minutes.update(minutes)

        display_draw_flag = len(all_pairs[-1][0][0][1].split('-')) == 3
        matchup_pairs, categories = _get_matchup_pairs(soups[-1], league_name, league)
        past_matchup_pairs.extend(matchup_pairs)
        tables = leagues_tables[league_name]
        tables['Past matchup stats'] = _export_past_matchup_stats(each_category_type_flag, categories, matchup_pairs,
            all_pairs[-1], minutes, tiebreaker, False, display_draw_flag)

        category_stats, win_stats, comparisons_stats = _get_league_tables_data(league, league_name,
                                                                               all_pairs, soups, tiebreaker)
        comparisons_matchup, comparisons_h2h = comparisons_stats
        matchups = np.arange(1, matchup + 1)
        tables['Pairwise comparisons h2h'] = utils.render_h2h_table(comparisons_h2h)
        tables['Pairwise comparisons by matchup'] = _render_matchup_comparisons_table(comparisons_matchup, matchups)
        if each_category_type_flag:
            tables['Expected category win stats'] = _render_category_win_stats_table(category_stats, matchup,
                                                                                     len(categories), display_draw_flag)
        else:
            tables['Expected matchup win stats'] = _render_matchup_win_stats_table(win_stats, matchups)

    overall_tables = {}
    if len(leagues) > 1:
        overall_tables['Past matchup overall stats'] = _export_past_matchup_stats(each_category_type_flag, categories,
            past_matchup_pairs, past_matchup_scores, past_matchup_minutes, tiebreaker, True, display_draw_flag)

    season_str = f'{season_start_year}-{str(season_start_year + 1)[-2:]}'
    utils.export_tables_to_html(sport, leagues_tables, overall_tables,
                                leagues[0], season_str, matchup, github_login, schedule, test_mode_on)
