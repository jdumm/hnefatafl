"""Plot training statistics from the latest StatsTracker checkpoint."""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np

# Ensure StatsTracker class is importable for unpickling.
import stats_tracker  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot win rates, draw rate, avg moves, and avg duration from latest StatsTracker checkpoint."
    )
    parser.add_argument(
        "model_dir",
        help="Model/version directory containing StatsTracker_*_games.pkl files (e.g. models_brandubh_v45)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Rolling window size override. Defaults to StatsTracker.n_games_window.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <model_dir>/training_stats_latest.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving PNG.",
    )
    return parser.parse_args()


def find_latest_stats_file(model_dir: Path) -> Path:
    pattern = re.compile(r"StatsTracker_(\d+)_games\.pkl$")
    candidates = []
    for path in model_dir.glob("StatsTracker_*_games.pkl"):
        match = pattern.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No StatsTracker_*_games.pkl files found in {model_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be > 0")
    n = len(series)
    out = np.zeros(n, dtype=np.float64)
    csum = np.cumsum(np.insert(series.astype(np.float64), 0, 0.0))
    for i in range(n):
        start = max(0, i - window + 1)
        total = csum[i + 1] - csum[start]
        count = i - start + 1
        out[i] = total / count
    return out


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists() or not model_dir.is_dir():
        raise NotADirectoryError(f"Invalid model directory: {model_dir}")

    latest_stats_path = find_latest_stats_file(model_dir)
    with open(latest_stats_path, "rb") as f:
        stats = pickle.load(f)

    outcomes = np.array(stats.a_outcomes, dtype=np.float64)
    moves = np.array(stats.num_moves, dtype=np.float64)
    durations = np.array(stats.game_durations, dtype=np.float64)
    if len(outcomes) == 0:
        raise ValueError(f"StatsTracker file is empty: {latest_stats_path}")

    window = args.window if args.window is not None else int(stats.n_games_window)
    if window < 1:
        raise ValueError("Window must be >= 1")

    attacker_wins = (outcomes > 0).astype(np.float64)
    defender_wins = (outcomes < 0).astype(np.float64)
    draws = (outcomes == 0).astype(np.float64)

    attacker_rate = rolling_mean(attacker_wins, window)
    defender_rate = rolling_mean(defender_wins, window)
    draw_rate = rolling_mean(draws, window)
    avg_moves = rolling_mean(moves, window)
    avg_duration = rolling_mean(durations, window)

    initial_games = int(getattr(stats, "initial_num_games", 0))
    x = np.arange(1, len(outcomes) + 1) + initial_games

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Training Stats: {model_dir.name}\nSource: {latest_stats_path.name} | window={window}")

    ax1.plot(x, attacker_rate, label="Attacker Win Rate", linewidth=1.8)
    ax1.plot(x, defender_rate, label="Defender Win Rate", linewidth=1.8)
    ax1.plot(x, draw_rate, label="Draw Rate", linewidth=1.8)
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    line_moves = ax2.plot(x, avg_moves, label="Avg Moves", color="tab:green", linewidth=1.8)[0]
    ax2.set_ylabel("Avg Moves", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Games")

    ax2b = ax2.twinx()
    line_dur = ax2b.plot(x, avg_duration, label="Avg Duration (s)", color="tab:red", linewidth=1.8)[0]
    ax2b.set_ylabel("Avg Duration (s)", color="tab:red")
    ax2b.tick_params(axis="y", labelcolor="tab:red")

    ax2.legend([line_moves, line_dur], ["Avg Moves", "Avg Duration (s)"], loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(args.output) if args.output else model_dir / "training_stats_latest.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
