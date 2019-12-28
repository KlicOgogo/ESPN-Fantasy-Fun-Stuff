TOP_PERCENTAGE = 0.4


def color_extremums(s):
    attr = 'background-color'
    return [f'{attr}: lightgreen' if v == s['Best'] else f'{attr}: orange' if v == s['Worst'] else '' for v in s]


def color_matchup_result(v):
    color = 'darkred' if v == 'L' else 'black' if v == 'D' else 'darkgreen'
    return f'color: {color}'


def color_opponent_place_column(s):
    return ['color: red' if val / s.max() <= TOP_PERCENTAGE else 'color: blue' for val in s]


def color_opponent_value(v):
    return color_value(-v)


def color_place_column(s):
    return ['color: blue' if val / s.max() <= TOP_PERCENTAGE else 'color: red' for val in s]


def color_value(v):
    color = 'red' if v < 0 else 'black' if v == 0 else 'green'
    return f'color: {color}'
