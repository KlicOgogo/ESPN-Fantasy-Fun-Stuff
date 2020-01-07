TOP_PERC = 0.4


def color_extremums(s):
    attr = 'background-color'
    return [f'{attr}: lightgreen' if v == s['Best'] else f'{attr}: orange' if v == s['Worst'] else '' for v in s]


def color_opponent_place_column(s):
    return ['color: red' if v / len(s) <= TOP_PERC else 'color: blue' if v / len(s) > 1 - TOP_PERC else '' for v in s]


def color_opponent_value(v):
    return color_value(-v)


def color_pair_result(v):
    color = 'darkred' if v == 'L' else 'black' if v == 'D' else 'darkgreen'
    return f'color: {color}'


def color_percentage(v):
    color = 'red' if v <= 0.3 else 'green' if v >= 0.7 else 'black'
    return f'color: {color}'


def color_place_column(s):
    return ['color: blue' if v / len(s) <= TOP_PERC else 'color: red' if v / len(s) > 1 - TOP_PERC else '' for v in s]


def color_value(v):
    color = 'red' if v < 0 else 'black' if v == 0 else 'green'
    return f'color: {color}'
