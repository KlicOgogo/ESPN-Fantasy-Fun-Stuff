#!/usr/bin/env python3
import categories
import points


SLEEP_TIMEOUT_IN_SECONDS = 7


def _export_basketball_stats():
    matchup = 10
    sport = 'basketball'
    categories.export_matchup_stats([43767928], True, sport, matchup, SLEEP_TIMEOUT_IN_SECONDS)
    categories.export_matchup_stats([134112], True, sport, matchup, SLEEP_TIMEOUT_IN_SECONDS)
    rwh_leagues = [174837, 142634, 199973, 282844, 99121987]
    categories.export_matchup_stats(rwh_leagues, False, sport, matchup, SLEEP_TIMEOUT_IN_SECONDS)


def _export_hockey_stats():
    matchup = 13
    sport = 'hockey'
    points.export_matchup_stats([27465869], sport, matchup, SLEEP_TIMEOUT_IN_SECONDS)
    rwh_leagues = [8290, 31769, 33730, 52338, 57256, 73362809]
    points.export_matchup_stats(rwh_leagues, sport, matchup, SLEEP_TIMEOUT_IN_SECONDS)


def main():
    _export_basketball_stats()
    _export_hockey_stats()


if __name__ == '__main__':
    main()
