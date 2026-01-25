# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hnefatafl (Viking board game) implementation with reinforcement learning AIs. The project includes:
- Player vs Player gameplay via pygame
- AI training through self-play using Keras/TensorFlow neural networks
- Support for multiple Tafl game variants (Hnefatafl 11x11, Brandubh 7x7, and simplified versions)
- Trained models stored as .h5 files with game count tracking

## Commands

### Running the Game
```bash
# Player vs Player
python hnefatafl.py

# AI training (batch mode, no display)
python hnefatafl_train.py --batch --train-attacker --train-defender

# Player vs AI Defender
python hnefatafl_train.py --interactive --human-attacker --ai-defender

# Player vs AI Attacker
python hnefatafl_train.py --interactive --human-defender --ai-attacker

# Watch AI vs AI
python hnefatafl_train.py --interactive --ai-attacker --ai-defender
```

### Training Options
```bash
# Load latest models and continue training
python hnefatafl_train.py --batch --train-attacker --train-defender --load-latest

# Train specific variant (e.g., Brandubh 7x7)
python hnefatafl_train.py --game-name Brandubh --train-attacker --train-defender

# Set learning rate and epsilon (exploration rate)
python hnefatafl_train.py --initial-lr 0.001 --epsilon 0.1 --lr-decay

# Control model saving frequency
python hnefatafl_train.py --cache-model-every 50
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Core Game Logic (hnefatafl.py)
- **Board class**: Manages board state and piece initialization for different game variants
  - Parses grid strings where 'a'=attacker, 'd'=defender, 'k'=king, 'x'=corners, 'c'=center
  - `tafl.DIM` is set based on game variant (11 for Hnefatafl, 7 for Brandubh, etc.)
- **Move class**: Handles turn logic, move validation, piece selection, and capture rules
  - King captured when surrounded on all 4 sides
  - Regular pieces captured by sandwiching (corners and center count as hostile)
  - King escapes by reaching any corner
- **Piece classes**: Attacker, Defender sprites with pygame rendering
  - Global sprite groups: `tafl.Attackers`, `tafl.Defenders`, `tafl.Kings`

### AI Training System (hnefatafl_train.py)
- **Board State Encoding**: `game_state_to_3d_array()` creates (DIM, DIM, 3) numpy arrays
  - 3 channels: attackers (A_DIM=0), king (K_DIM=1), defenders (D_DIM=2)
- **Move Selection**: `do_best_move()` evaluates all legal moves using batch prediction
  - Explores state space by trying each piece's valid moves
  - Scores positions with the neural network
  - Applies temperature for exploration vs exploitation
  - Uses epsilon-greedy strategy for random moves during training
- **Training Loop**: Self-play stores game states and outcomes
  - After each game, states are scored based on win/loss
  - Models updated via gradient descent on scored positions
  - Uses intermediate reward shaping (not just terminal states)
- **Symmetry Augmentation**: Optional flag to train on rotated/flipped board states

### Neural Network Models (models.py)
Three architectures available:
- **simple_model**: Lightweight CNN for basic variants (7x7 boards)
  - Single conv layer → 1x1 conv → dense layers
- **sonnet_model**: Medium-complexity with skip connections
  - Residual blocks, multi-scale convolutions, batch normalization
- **claude_model**: Deep network with 4 residual blocks
  - Multiple conv kernel sizes (5x5, 3x3, 1x1) for pattern recognition

All models:
- Output single value in [-1, +1] via tanh (board evaluation score)
- Take 3-channel board state as input
- Use TruncatedNormal initialization to prevent score explosion

### Statistics Tracking (stats_tracker.py)
- **StatsTracker class**: Monitors win rates, game length, duration
  - Tracks both windowed (recent N games) and total statistics
  - Saved as pickle files alongside trained models
  - Used for dynamic training adjustments

## Key Implementation Details

### Model Persistence
- Models saved as `{attacker/defender}_model_{num_games}_games.h5`
- StatsTracker saved as `StatsTracker_{num_games}_games.pkl`
- Two model directories: `models/` (Hnefatafl 11x11) and `models_brandubh_v12/` (Brandubh 7x7)
- Use `--load-latest` to automatically find and load the most recent models

### Game State Management
- Global state stored in tafl module sprite groups (Attackers, Defenders, Kings)
- Board state must be captured before evaluating moves, then restored after
- `game_state_cache` dict used to avoid re-encoding identical positions during move search

### Training Strategies
- **Dynamic Training**: `--dynamic-train` pauses defender training when attacker win rate is lopsided
- **Shaping Decay**: Intermediate reward shaping decays exponentially with `exp(-games/k)`
- **Learning Rate Decay**: `--lr-decay` reduces learning rate as training progresses
- **Symmetry**: `--use-symmetry` trains on board rotations/reflections for data augmentation

### Game Variants
The `--game-name` parameter supports:
- `Hnefatafl`: Classic 11x11 board
- `Brandubh`: 7x7 Irish variant
- `simple`: Minimal 5x5 test variant
- `brandubh_simple`: Simplified 7x7 for testing

Each variant has different initial piece layouts and board dimensions configured in Board.__init__().

## Known Issues / Notes

### Nested Undo in Move Simulations
The `move.undo()` function properly restores captured pieces using `RemovedPieces`, `RemovedAttackers`, etc. sprite groups. However, `remove_pieces()` clears these groups at the START of each call (lines 183-186 in hnefatafl.py). This means nested move simulations (like the `defender_move_gives_attacker_win` safety check in do_best_move) can lose track of outer captures when inner calls clear the Removed groups.

**Impact**: The `defender_move_gives_attacker_win` function does nested lookahead which can cause "Best move logic failed" errors at log-level > 0. These are non-fatal (fallback to score 0.0) and don't affect training at log-level 0.

**Potential fix**: Use a stack-based approach for Removed groups, or save/restore them before nested calls.

## Text-Based Board Display

### Board Display Functions
- `get_board_text(include_labels=True)` - Returns string representation of board
  - Shows pieces: 'a' (attackers), 'd' (defenders), 'K' (king)
  - Shows empty special squares: 'x' (corners), 'c' (center throne)
- `print_board_state(move, turn_num, extra_info)` - Prints formatted board with metadata

### Interactive Mode Features
- Board state automatically printed at game start, after moves, and at game end
- Press 'P' key during gameplay to print current board state on demand
- Only prints in interactive mode (not batch training)
