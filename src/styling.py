TOP_PERC = 0.3333333333


def _norm(place, place_list):
    return (place - 1) / (len(place_list) - 1)


def color_extremums(s):
    attr = 'background-color'
    return [f'{attr}: lightgreen' if v == s['Best'] else f'{attr}: orange' if v == s['Worst'] else '' for v in s]


def color_opponent_place_column(s):
    return ['color: red' if is_top(v, s) else 'color: blue' if is_bottom(v, s) else '' for v in s]


def color_opponent_value(v):
    return color_value(-v)


def color_pair_result(v):
    color = 'darkred' if v == 'L' else 'black' if v == 'D' else 'darkgreen'
    return f'color: {color}'


def color_percentage(v):
    color = 'red' if v < TOP_PERC else 'green' if v >= 1 - TOP_PERC else 'black'
    return f'color: {color}'


def color_place_column(s):
    return ['color: blue' if is_top(v, s) else 'color: red' if is_bottom(v, s) else '' for v in s]


def color_value(v):
    color = 'red' if v < 0 else 'black' if v == 0 else 'green'
    return f'color: {color}'


def is_top(place, place_list):
    return _norm(place, place_list) <= TOP_PERC


def is_bottom(place, place_list):
    return _norm(place, place_list) > 1.0 - TOP_PERC
