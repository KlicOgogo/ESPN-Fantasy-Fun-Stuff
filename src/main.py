#!/usr/bin/env python3
import json
import os
import sys

import categories
import points
from utils import REPO_ROOT_DIR


def _export_stats(sport, config, timeout, test_mode_on=False):
    for cat_leagues in config['categories']:
        categories.export_matchup_stats(cat_leagues, sport, test_mode_on, timeout)
    for points_leagues in config['points']:
        points.export_matchup_stats(points_leagues, sport, test_mode_on, timeout)


def main():
    with open(os.path.join(REPO_ROOT_DIR, '.config'), 'r') as config_fp:
        config = json.load(config_fp)
    basketball = 'basketball'
    hockey = 'hockey'
    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        _export_stats(basketball, config[basketball], config['timeout'], True)
        _export_stats(hockey, config[hockey], config['timeout'], True)
    elif len(sys.argv) == 2 and sys.argv[1] == 'testhockey':
        _export_stats(hockey, config[hockey], config['timeout'], True)
    elif len(sys.argv) == 2 and sys.argv[1] == 'testbasketball':
        _export_stats(basketball, config[basketball], config['timeout'], True)
    elif len(sys.argv) == 2 and sys.argv[1] == 'hockey':
        _export_stats(hockey, config[hockey], config['timeout'])
    elif len(sys.argv) == 2 and sys.argv[1] == 'basketball':
        _export_stats(basketball, config[basketball], config['timeout'])
    else:
        _export_stats(basketball, config[basketball], config['timeout'])
        _export_stats(hockey, config[hockey], config['timeout'])


if __name__ == '__main__':
    main()
