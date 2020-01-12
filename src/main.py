#!/usr/bin/env python3
import json
import os
import sys

import categories
import points
from utils import REPO_ROOT_DIR


def _export_stats(sport, config, github_login, timeout, test_mode_on=False):
    for cat_leagues in config['categories']:
        if len(cat_leagues) == 1:
            leagues, tiebreaker = cat_leagues[0], 'NO'
        elif len(cat_leagues) == 2:
            leagues, tiebreaker = cat_leagues
        else:
            raise Exception('Wrong config: leagues tuple must contain 1 or 2 elements.')
        categories.export_matchup_stats(leagues, sport, tiebreaker, github_login, test_mode_on, timeout)
    for points_leagues in config['points']:
        points.export_matchup_stats(points_leagues, sport, github_login, test_mode_on, timeout)


def main():
    with open(os.path.join(REPO_ROOT_DIR, '.config'), 'r') as config_fp:
        config = json.load(config_fp)
    basketball_key = 'basketball'
    hockey_key = 'hockey'
    github_key = 'github'
    timeout_key = 'timeout'
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        _export_stats(basketball_key, config[basketball_key], config[github_key], config[timeout_key], True)
        _export_stats(hockey_key, config[hockey_key], config[github_key], config[timeout_key], True)
    elif len(sys.argv) == 2 and sys.argv[1] == 'testhockey':
        _export_stats(hockey_key, config[hockey_key], config[github_key], config[timeout_key], True)
    elif len(sys.argv) == 2 and sys.argv[1] == 'testbasketball':
        _export_stats(basketball_key, config[basketball_key], config[github_key], config[timeout_key], True)
    elif len(sys.argv) == 2 and sys.argv[1] == 'hockey':
        _export_stats(hockey_key, config[hockey_key], config[github_key], config[timeout_key])
    elif len(sys.argv) == 2 and sys.argv[1] == 'basketball':
        _export_stats(basketball_key, config[basketball_key], config[github_key], config[timeout_key])
    elif len(sys.argv) == 1:
        _export_stats(basketball_key, config[basketball_key], config[github_key], config[timeout_key])
        _export_stats(hockey_key, config[hockey_key], config[github_key], config[timeout_key])


if __name__ == '__main__':
    main()
