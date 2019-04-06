#Context
The Player vs Player version of this game and some description below was forked from: [slowen/hnefatafl](https://github.com/slowen/hnefatafl).  The original source file has only minor modifcations.  The purpose of this repo is to build Attacker and Defender AIs through Reinforcement Learning.  These AIs can continue to learn by playing each other or versus a Player.

#Hnefatafl
Hnefatafl is an ancient Viking strategy board game. There are two teams- an attacking team and a defending team. The attacking team tries to capture the other team's king, whereas the defending team tries to get their king to one of the corners of the board.

![startgame](https://cloud.githubusercontent.com/assets/5671974/8666780/faf2ba88-29c3-11e5-8d53-a7349d4e76b4.png)

#Rules of the Game
The game is played on an 11x11 board, and the initial layout is shown above. The attacking team (red) starts first.

Every piece can move in the same way- just like a rook in chess. They can move horizontally or vertically as far as they want, but they cannot jump over any other pieces. The king (green) is the only piece that can move *onto* the center and corner tiles, but all of the other pieces can move *through* the center tile.

Players can capture their opponents pieces by sandwiching an opponents piece. The center and corners are considered hostile territory, so they can be used to sandwich/capture an opponent. For instance, if the defending player moves their piece such that they sandwich an attacking piece between their piece and the corner, then the attacking piece is removed from the board.

The king is captured when it is enclosed on all four sides.

#Running the Game
The only imported packages are sys and pygame for the PvP version. To install pygame on Unix: 
```
sudo apt-get install python-pygame
```
On Mac:
```
python3 -m pip install -U pygame --user
```

To run the PvP game, simply run:
```
python hnefatafl.py
```

While in the game, pressing ```r``` will ask the user if they want to restart the game, which they can confirm with ```y``` or ```n```. Also, when one player has won the game, they can start a new game by pressing ```y``` or exit the game by pressing ```n```.

The text on the bottom will tell you whose turn it is (red is the attacker, blue is the defender, and the king is green).

#Limitations
The game is not quite finished yet. It cannot detect if there is a draw game. Other than that, it is fully functional.

#AI training using reinforcement learning
Keras Deep Neural Nets are used to predict the score (-1 = very bad to +1 = very good) of any given game board state (current arrangment of pieces) for possible future moves and performing the best move.  These DNNs are initialized with random values and learn through adversarial self play.  When a game is finished, the states leading up to the final outcome are scored more highly/lowly for the team who won/lost.  These DNNs are updated after each game, learning from their mistakes.  These trained AIs can be saved and reloaded.  

I followed the basic approach outlined in a [Kaggle notebook on self-learning Tic-Tac-Toe](https://www.kaggle.com/dhanushkishore/a-self-learning-tic-tac-toe-program/notebook), though I alter more than the last state leading up to a win since hnefatafl takes many more turns to complete.

Additional dependencies for the AI models are:
* keras with tensorflow backend
* numpy

#Resources
If you want to learn more about Hnefatafl, check out http://tafl.cyningstan.com/. It explains the game really well and has some great advice on strategy for both sides.  [Wikipedia](https://en.wikipedia.org/wiki/Tafl_games) describes the interesting history and many variations.  
