from collections import Counter, defaultdict
import datetime
from operator import itemgetter
import os
from pathlib import Path
import re

from jinja2 import Template
import numpy as np
import pandas as pd

from styling import color_percentage


ATTRS = 'style="border-collapse: collapse; border: 1px solid black;" align= "center"'
REPO_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
STYLES = [
    dict(selector='td', props=[('border-left', '1px solid black'), ('border-right', '1px solid black'),
                               ('text-align', 'right'), ('padding-left', '4px'), ('padding-right', '4px')]),
    dict(selector='td:first-child', props=[('border-left', 'none')]),
    dict(selector='td:last-child', props=[('border-right', 'none')]),
    dict(selector='th', props=[('border-left', '1px solid black'), ('border-right', '1px solid black'),
                               ('border-bottom', '1px solid black'), ('background', '#FFFFFF'),
                               ('padding-left', '6px')]),
    dict(selector='tr:nth-child(odd)', props=[('background', '#F0F0F0')]),
]
ZERO = 1e-7


def _get_previous_reports_data(report_dir, matchup, github_login, schedule):
    contents = [os.path.join(report_dir, path) for path in os.listdir(report_dir)]
    html_list = [path for path in contents if os.path.splitext(path)[1] == '.html']
    reports = [html for html in html_list if re.match(r'matchup_\d+\.html', os.path.basename(html))]
    matchup_list = [int(re.findall(r'matchup_(\d+)\.html', os.path.basename(r))[0]) for r in reports]
    prev_reports = []
    for r, m in zip(reports, matchup_list):
        if m < matchup:
            prev_reports.append((r.replace(REPO_ROOT_DIR.rstrip('/') + '/', ''), m))

    previous_reports_data = {}
    for report_link_end, number in prev_reports:
        report_link = f'https://{github_login.lower()}.github.io/ESPN-Fantasy-Fun-Stuff/{report_link_end}'
        this_matchup_begin, this_matchup_end = map(lambda x: x.strftime("%d/%m/%Y"), schedule[number])
        report_text = f'Matchup {number} ({this_matchup_begin} - {this_matchup_end}).'
        previous_reports_data[number] = (report_text, report_link)
    return previous_reports_data


def add_position_column(df):
    position = {index: i + 1 for i, index in enumerate(df.index)}
    df_position = pd.DataFrame(list(position.values()), index=position.keys(), columns=['Pos'])
    result_df = df_position.merge(df, how='outer', left_index=True, right_index=True)
    return result_df


def export_tables_to_html(sport, leagues_tables, total_tables, league_id, season, matchup, 
                          github_login, schedule, test_mode_on):
    report_dir = os.path.join(REPO_ROOT_DIR, 'reports', sport, str(league_id), season)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    previous_reports_data = _get_previous_reports_data(report_dir, matchup, github_login, schedule)

    with open(os.path.join(REPO_ROOT_DIR, 'templates/matchup_report.html'), 'r') as template_fp:
        template = Template(template_fp.read())
    html_str = template.render({
        'matchup': matchup,
        'sport': sport,
        'leagues': leagues_tables,
        'total_tables': total_tables,
        'previous_reports': previous_reports_data
    })
    index_path = os.path.join(report_dir, 'index.html')
    with open(index_path, 'w') as html_fp:
        html_fp.write(html_str)
    if not test_mode_on:
        matchup_path = os.path.join(report_dir, f'matchup_{matchup}.html')
        with open(matchup_path, 'w') as html_fp:
            html_fp.write(html_str)


def find_proper_matchup(schedule):
    four_days_ago = datetime.datetime.today().date() - datetime.timedelta(days=4)
    for matchup_number, matchup_date in schedule.items():
        if matchup_date[0] <= four_days_ago <= matchup_date[1]:
            return matchup_number
    return -1


def get_opponent_dict(scores):
    opp_dict = {}
    for s in scores:
        opp_dict[s[0][0]] = s[1][0]
        opp_dict[s[1][0]] = s[0][0]
    return opp_dict


def get_places(sorted_scores):
    places = {}
    i = 1
    while i <= len(sorted_scores):
        j = i + 1
        while j <= len(sorted_scores) and sorted_scores[j-1][1] == sorted_scores[i-1][1]:
            j += 1
        place = (i + j - 1) / 2
        for k in range(i, j):
            places[sorted_scores[k - 1][0]] = place
        i = j
    return places


def render_h2h_table(h2h_comparisons):
    h2h_sums = {}
    h2h_powers = {}
    for team in h2h_comparisons:
        team_h2h_sum = sum(h2h_comparisons[team].values(), Counter())
        h2h_sums[team] = [team_h2h_sum[result] for result in ['W', 'L', 'D']]
        h2h_powers[team] = (np.sum(np.array(h2h_sums[team]) * np.array([1.0, 0.0, 0.5])), team_h2h_sum['W'])
    h2h_sums_sorted = sorted(h2h_powers.items(), key=itemgetter(1), reverse=True)
    h2h_order = [team for team, _ in h2h_sums_sorted]

    h2h_data = defaultdict(list)
    for team in h2h_order:
        for opp in h2h_order:
            if team == opp:
                h2h_data[team].append('')
            else:
                comp = h2h_comparisons[team][opp]
                h2h_data[team].append('-'.join(map(str, [comp['W'], comp['L'], comp['D']])))

    df_data = []
    for team in h2h_order:
        team_data = []
        team_data.append(team[0])
        team_data.extend(h2h_data[team])
        team_data.extend(h2h_sums[team])
        team_data.append(np.round(h2h_powers[team][0] / np.sum(h2h_sums[team]), 2))
        df_data.append(team_data)

    df = pd.DataFrame(df_data, columns=['Team', *np.arange(1, len(df_data)+1), 'W', 'L', 'D', '%'])
    df = add_position_column(df)
    styler = df.style.format({'%': '{:g}'}).set_table_styles(STYLES).set_table_attributes(ATTRS).hide_index().\
        applymap(color_percentage, subset=['%'])
    return styler.render()
