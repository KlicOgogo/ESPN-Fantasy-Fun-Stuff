import operator


def get_luck_score(matchups, places):
    lk = {}
    for player1, player2 in matchups:
        if player1[1] > player2[1]:
            place1 = places[player1[0]]
            lk[player1[0]] = max(0, place1 - len(places) / 2)

            place2 = places[player2[0]]
            lk[player2[0]] = min(0, place2 - len(places) / 2 - 1)
        elif player1[1] < player2[1]:
            place1 = places[player1[0]]
            lk[player1[0]] = min(0, place1 - len(places) / 2 - 1)

            place2 = places[player2[0]]
            lk[player2[0]] = max(0, place2 - len(places) / 2)
        else:
            place = places[player1[0]]
            if place > len(places) / 2:
                lk[player1[0]] = (place - len(places) / 2) / 2
                lk[player2[0]] = (place - len(places) / 2) / 2
            else:
                lk[player1[0]] = (place - len(places) / 2 - 1) / 2
                lk[player2[0]] = (place - len(places) / 2 - 1) / 2
    return lk


def get_sorted_week_scores(week_matchups):
    scores = []
    for matchup in week_matchups:
        scores.extend(matchup)
    return sorted(scores, key=operator.itemgetter(1), reverse=True)
