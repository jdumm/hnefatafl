# Context
The Player vs Player version of this game and some description below was forked from: [slowen/hnefatafl](https://github.com/slowen/hnefatafl).  The original source file has only minor modifications.  The purpose of this repo is to build Attacker and Defender AIs through Reinforcement Learning.  These AIs can continue to learn by playing each other or versus a Player.

# Hnefatafl

Hnefatafl is an ancient Viking strategy board game. There are two teams- an attacking team and a defending team. The attacking team tries to capture the other team's king, whereas the defending team tries to get their king to one of the corners of the board.

![startgame](https://cloud.githubusercontent.com/assets/5671974/8666780/faf2ba88-29c3-11e5-8d53-a7349d4e76b4.png)

# Rules of the Game
The game is played on an 11x11 board, and the initial layout is shown above. The attacking team (red) starts first.

Every piece can move in the same way- just like a rook in chess. They can move horizontally or vertically as far as they want, but they cannot jump over any other pieces. The king (green) is the only piece that can move *onto* the center and corner tiles, but all of the other pieces can move *through* the center tile.

Players can capture their opponent's pieces by sandwiching an opponent's piece. The center and corners are considered hostile territory, so they can be used to sandwich/capture an opponent. For instance, if the defending player moves their piece such that they sandwich an attacking piece between their piece and the corner, then the attacking piece is removed from the board.

The king is captured when it is enclosed on all four sides.

# Running the Game
Dependencies can be installed by running:
```
pip install -r requirements.txt 
```

To run the PvP game, simply run:
```
python hnefatafl.py
```

While in the game, pressing ```r``` will ask the user if they want to restart the game, which they can confirm with ```y``` or ```n```. Also, when one player has won the game, they can start a new game by pressing ```y``` or exit the game by pressing ```n```.

The text on the bottom will tell you whose turn it is (red is the attacker, blue is the defender, and the king is green).

# AI training Using Reinforcement Learning
Keras Deep Neural Nets are used to predict the score (-1 = very bad to +1 = very good) of any given game board state (current arrangement of pieces) for possible future moves.  These DNNs are initialized with random values and learn through adversarial self play.  When a game is finished, the states leading up to the final outcome are scored higher/lower for the team who won/lost.  These DNNs are updated after each game, learning from their mistakes.  These trained AIs can be saved and reloaded.  

I followed the basic approach outlined in a [Kaggle notebook on self-learning Tic-Tac-Toe](https://www.kaggle.com/dhanushkishore/a-self-learning-tic-tac-toe-program/notebook), though I alter more just than the last state leading up to a win since hnefatafl takes many more turns to complete.

Additional dependencies for the AI models are:
* keras with tensorflow backend
* numpy

# Train module
The training module can be executed using:
```
python hnefatafl_train.py
```

Options can be configured from the command line for different modes, including human-vs-AI and AI-vs-AI. Explore all current options with:

```
python hnefatafl_train.py --help
```

## Training architecture (current)
- A single shared value model is used for both sides.
- Value is attacker-perspective:
  - `+1` attacker win
  - `-1` defender win
  - `-0.5` draw
- On attacker turns the move picker maximizes value; on defender turns it minimizes value.
- Fixed board encoding is always 8 channels:
  - attackers, king, defenders
  - corners, center
  - turn channel
  - attacker count, defender count

## Model files and stats files
- Checkpoints are saved as:
  - `shared_model_<N>_games.keras`
- Rolling game stats are saved as:
  - `StatsTracker_<N>_games.pkl`
- Use `--load-latest` to continue from the latest saved checkpoint/stats in `models_<game>_v<version>`.

## Core training controls
- `--train-model`: enable model updates after each game.
- `--cache-model-every <N>`: save model and stats every N games.
- `--exit-after-cache`: useful for scheduled restarts.
- `--model-preset [auto|simple|brandubh|hnefatafl]`: select architecture preset.
- `--use-td/--no-td`, `--gamma`: target generation mode.
- `--probabilistic-symmetry` or `--use-symmetry`: data augmentation.
- `--sample-frac <0..1>`: fraction of candidate pieces/moves evaluated per turn (`1.0` = strongest, slower).

## Exploration controls
- `--initial-temp`, `--final-temp`, `--temp-decay`
- `--initial-epsilon`, `--final-epsilon`, `--epsilon-decay`
- Current defaults:
  - temperature: `0.5 -> 0.02`
  - epsilon: `0.0 -> 0.0`
- Previous epsilon defaults (older behavior):
  - epsilon: `0.3 -> 0.01`
- Previous temperature defaults (older behavior):
  - temperature: `0.5 -> 0.02` (unchanged)

## Scripted policy controls
The trainer supports scripted one-ply heuristics that can be used as full policies or mixed with model play.

- Side policy:
  - `--attacker-policy [model|scripted|random]`
  - `--defender-policy [model|scripted|random]`
- Hybrid mix while keeping `model` policy:
  - `--attacker-scripted-frac <0..1>`
  - `--defender-scripted-frac <0..1>`
  - Example: `--defender-policy model --defender-scripted-frac 0.25`
    means 25% of defender AI turns use scripted logic, 75% model.
- Scripted behavior knobs:
  - `--defender-escape-turn <int>`: defender shifts from peel/attrition to king-escape focus after this move number.
  - `--scripted-noise <float>`: adds randomness to scripted scoring to avoid deterministic play.

## Logging controls
- `--log-level 0`: minimal output.
- `--log-level 1`: periodic summaries.
- `--log-level 2`: detailed per-game debugging.
- `--log-every <N>`: summary interval for log level 1.

At each summary point the trainer logs policy telemetry:
- Policy mix (window): human/model/scripted/random usage by side since last summary.
- Policy mix (total): cumulative usage since run start.

## Recommended command examples
Continue Brandubh self-play with moderate logging:
```
python hnefatafl_train.py --game-name Brandubh --batch --train-model --probabilistic-symmetry --model-preset brandubh -v 46 --load-latest --cache-model-every 1000 --log-level 1 --log-every 200 --sample-frac 1.0 --initial-temp 0.5 --final-temp 0.02 --initial-epsilon 0.0 --final-epsilon 0.0 *> brandubh_v46.log
```

Model attacker vs scripted defender curriculum:
```
python hnefatafl_train.py --game-name Brandubh --batch --train-model --attacker-policy model --defender-policy scripted --defender-escape-turn 16 --scripted-noise 0.05 --probabilistic-symmetry --model-preset brandubh -v 47 --cache-model-every 1000 --log-level 1 --log-every 200 --sample-frac 1.0 --initial-temp 0.5 --final-temp 0.02 --initial-epsilon 0.0 --final-epsilon 0.0 *> brandubh_v47.log
```

Hybrid defender curriculum (mostly model with scripted assists):
```
python hnefatafl_train.py --game-name Brandubh --batch --train-model --attacker-policy model --defender-policy model --defender-scripted-frac 0.25 --defender-escape-turn 16 --model-preset brandubh -v 47 --cache-model-every 1000 --log-level 1 --log-every 200 --sample-frac 1.0 --initial-temp 0.5 --final-temp 0.02 --initial-epsilon 0.0 --final-epsilon 0.0 *> brandubh_v47_hybrid.log
```

## Plotting training curves
Use the included plotting script to graph latest tracker data:
```
python plot_training_stats.py models_brandubh_v46
```
Optional:
```
python plot_training_stats.py models_brandubh_v46 --window 500 --output brandubh_v46_stats.png --show
```

# Resources
If you want to learn more about Hnefatafl, check out http://tafl.cyningstan.com/. It explains the game really well and has some great advice on strategy for both sides.  [Wikipedia](https://en.wikipedia.org/wiki/Tafl_games) describes the interesting history and many variations.  
