import numpy as np


class StatsTracker:
    def __init__(self, n_games_window=100, initial_num_games=0):
        self.n_games_window = int(n_games_window)
        self.a_outcomes = []  # +1 for Attacker win, 0 = draw, -1 for Attacker loss
        self.num_moves = []  # Number of total moves in a game
        self.game_durations = []  # in seconds
        self.initial_num_games = int(initial_num_games)

    def __str__(self):
        if len(self.a_outcomes) == 0:
            return "Insufficient games tracked for StatsTracker."
        else:
            return """Attacker win rate is {:0.4f} over the last {} games, {:0.4f} over total of {} games.
        Draw rate is {:0.4f} over the last {} games, {:0.4f} over total of {} games.
    Avg num moves is {:6.1f} over the last {} games, {:6.1f} over total of {} games.
Avg game duration is {:0.4f} over the last {} games, {:0.4f} over total of {} games.""".format(
                self.a_win_rate_window(), len(self.a_outcomes[-1 * self.n_games_window:]),
                self.a_win_rate_tot(), len(self.a_outcomes) + self.initial_num_games,
                self.draw_rate_window(), len(self.a_outcomes[-1 * self.n_games_window:]),
                self.draw_rate_tot(), len(self.a_outcomes) + self.initial_num_games,
                np.mean(self.num_moves[-1 * self.n_games_window:]), len(self.num_moves[-1 * self.n_games_window:]),
                np.mean(self.num_moves), len(self.num_moves) + self.initial_num_games,
                np.mean(self.game_durations[-1 * self.n_games_window:]),
                len(self.game_durations[-1 * self.n_games_window:]),
                np.mean(self.game_durations), len(self.game_durations) + self.initial_num_games)

    def a_win_rate_tot(self):
        if len(self.a_outcomes) == 0:
            return 0.5
        return sum([1 if outcome > 0. else 0 for outcome in self.a_outcomes]) / float(len(self.a_outcomes))

    def a_win_rate_window(self):
        if len(self.a_outcomes) == 0:
            return 0.5
        return sum([1 if outcome > 0. else 0 for outcome in self.a_outcomes[-1 * self.n_games_window:]]) / float(
            len(self.a_outcomes[-1 * self.n_games_window:]))

    def draw_rate_tot(self):
        if len(self.a_outcomes) == 0:
            return 0.0
        return sum([1 if outcome == 0. else 0 for outcome in self.a_outcomes]) / max(float(len(self.a_outcomes)), 1)

    def draw_rate_window(self):
        if len(self.a_outcomes) == 0:
            return 0.0
        return sum([1 if outcome == 0. else 0 for outcome in self.a_outcomes[-1 * self.n_games_window:]]) / max(float(
            len(self.a_outcomes[-1 * self.n_games_window:])), 1)

    def add_game_results(self, a_score, num_move,
                         game_duration):  # ascore: +1 = Attacker wins, 0 = draw, or -1 = Defender wins
        self.a_outcomes.append(a_score)
        self.num_moves.append(num_move)
        self.game_durations.append(game_duration)

    def num_games_total(self):
        return len(self.a_outcomes)

