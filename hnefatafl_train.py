"""
An extension of Hnefatafl to include AI training, AI vs AI, and Player vs AI (attacker or defender) modes.

A full description of the game can be found here: https://en.wikipedia.org/wiki/Tafl_games

Author: Jon Dumm
Date: 4/4/2019

"""

import sys
import pygame
from pygame.locals import *
import time
import random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import hnefatafl as tafl


def run_game_random(screen=None):

    """Start a new game with random (legal) moves.
       TODO: Remove?  I think it's no longer needed.
    """
    board = tafl.Board()
    move = tafl.Move()
    tafl.initialize_pieces(board)
    num_moves = 0
    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
               pass
        
        do_random_move(move)
        num_moves += 1
        if(num_moves >= 1000):
            print("Draw game after {} moves".format(num_moves))
            return False

        """Text to display on bottom of game."""
        text2 = None
        if move.escaped:
            text = "King escaped! Defenders win!"
            print(text)
            text2 = "Play again? y/n"
            return False
        if move.king_killed:
            text = "King killed! Attackers win!"
            print(text)
            text2 = "Play again? y/n"
            return False
        if move.restart:
            text = "Restart game? y/n"
            print(text)
            return False
        if screen is not None:
            tafl.update_image(screen, board, move, text)
            pygame.display.flip()
        #time.sleep(1)

def do_random_move(move):
    """ Purely random but legal moves
    """ 
    if move.a_turn:
        pieces = tafl.Attackers
    else:
        pieces = tafl.Defenders
    while 1:
        piece = random.choice(pieces.sprites())
        move.select(piece)
        tafl.Current.add(piece)
        if len(move.vm)==0:
            move.select(piece)
            tafl.Current.empty()
            continue
        else:
            pos = random.choice(tuple(move.vm))
            if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                if tafl.Current.sprites()[0] in tafl.Kings:
                    move.king_escaped(tafl.Kings)
                if move.a_turn:
                    move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
                else:
                    move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)
                move.end_turn(tafl.Current.sprites()[0])
                tafl.Current.empty()
            break

def do_mostly_random_but_strike_to_kill_move(move): 
    """ Very basic rules for defender logic.  King moves to escape or next-to-escape tiles if an option.
        And in 10% of moves, the king tries to move away from the center.  
    """ 

    if move.a_turn:
        pieces = tafl.Attackers
    else:
        # If King can win, do it.
        for king in tafl.Kings:
            move.select(king)
            tafl.Current.add(king)
            for pos in [(0,0),(0,10),(10,0),(10,10), (0,1),(1,0),(0,9),(1,10),(10,1),(9,0),(10,9),(9,10)]:
                if pos in move.vm:
                    if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                        move.king_escaped(tafl.Kings)
                    if move.a_turn:
                        move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
                    else:
                        move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)
                    move.end_turn(tafl.Current.sprites()[0])
                    tafl.Current.empty()
                    return 
            if random.random()<0.10: # Push King out from center if possible sometimes
                if len(move.vm)==0: break 
                else: 
                    for m in move.vm:
                        if abs(5-m[0])>3 or abs(5-m[1])>3:
                            if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                                move.king_escaped(tafl.Kings)
                            if move.a_turn:
                                move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
                            else:
                                move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)
                        move.end_turn(tafl.Current.sprites()[0])
                        tafl.Current.empty()
                        return 

            move.select(king)
            tafl.Current.empty()
        pieces = tafl.Defenders

    while 1:
        piece = random.choice(pieces.sprites())
        move.select(piece)
        tafl.Current.add(piece)

        if len(move.vm)==0:
            move.select(piece)
            tafl.Current.empty()
            continue
        else:
            pos = random.choice(tuple(move.vm))
            if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                if tafl.Current.sprites()[0] in tafl.Kings:
                    move.king_escaped(tafl.Kings)
                if move.a_turn:
                    move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
                else:
                    move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)
                move.end_turn(tafl.Current.sprites()[0])
                tafl.Current.empty()
            break


def run_game(attacker_model=None,defender_model=None,human_attacker=False,human_defender=False,screen=None):
    """Start and run one game of computer attacker vs computer defender hnefatafl.
 
       Args:
           attacker_model: Keras model that can '.predict' based on the game state.  Used
                           to determine the best of available moves. Random moves by default.
           defender_model: Same as attacker_model but for defender.  
           screen: Optional, used to monitor matches in pygame.

    """
    board = tafl.Board()
    move = tafl.Move()
    tafl.initialize_pieces(board)
    a_game_states = []
    a_predicted_scores = []
    d_game_states = []
    d_predicted_scores = []
    play = True
    num_moves = 0
    while 1:
        num_moves += 1
        """Text to display on bottom of game."""
        if screen is not None and not human_attacker and not human_defender:
            tafl.update_image(screen, board, move, "Red: {}".format(len(tafl.Attackers.sprites())), "Blue: {}".format(len(tafl.Defenders.sprites())))
        if screen is not None:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pass
        if(num_moves >= 350):
            print("Draw game after {} moves".format(num_moves))
            a_predicted_scores.append(0.0)
            d_predicted_scores.append(0.0)
            return play,a_game_states,a_predicted_scores[1:], d_game_states,d_predicted_scores[1:]

        if move.a_turn:
            #print("Attacker's Turn: Move {}".format(num_moves))
            if human_attacker:
                play = do_human_turn(screen, board, move)
            elif attacker_model is None:
                game_state = do_random_move(move)
                predicted_score = (random.random()-0.5) * 2
                a_game_states.append(game_state)
                a_predicted_scores.append(predicted_score)
            else:
                if human_defender: time.sleep(0.5)
                game_state,predicted_score = do_best_move(move,attacker_model,sample_frac=0.50)
                a_game_states.append(game_state)
                a_predicted_scores.append(predicted_score)
        else:
            #print("Defender's Turn: Move {}".format(num_moves))
            if human_defender:
                play = do_human_turn(screen, board, move)
            elif defender_model is None: 
                game_state = do_mostly_random_but_strike_to_kill_move(move)
                predicted_score = (random.random()-0.5) * 2
                d_game_states.append(game_state)
                d_predicted_scores.append(predicted_score)
            else:
                if human_attacker: time.sleep(0.5)
                game_state,predicted_score = do_best_move(move,defender_model,sample_frac=0.50)
                d_game_states.append(game_state)
                d_predicted_scores.append(predicted_score)

        if move.escaped:
            text = "King escaped! Defenders win!"
            print(text)
            text2 = "Play again? y/n"
            a_predicted_scores.append(-1.0)
            d_predicted_scores.append(+1.0)
            tafl.update_image(screen, board, move, text, text2)
            pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play,a_game_states,a_predicted_scores[1:], d_game_states,d_predicted_scores[1:] # i.e. the corrected scores from RL
        if move.king_killed:
            text = "King killed! Attackers win!"
            print(text)
            text2 = "Play again? y/n"
            a_predicted_scores.append(+1.0)
            d_predicted_scores.append(-1.0)
            tafl.update_image(screen, board, move, text, text2)
            pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play,a_game_states,a_predicted_scores[1:], d_game_states,d_predicted_scores[1:] # i.e. the corrected scores from RL
        if move.restart:
            return play,a_game_states,a_predicted_scores[1:], d_game_states,d_predicted_scores[1:] # i.e. the corrected scores from RL

def end_game_loop(move):
    while 1:  # Wait for human input
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if move.game_over and event.key == pygame.K_n:
                    return False
                if move.game_over and event.key == pygame.K_y:
                    return True


def do_human_turn(screen,board,move):
    print("Starting human turn")
    current_turn = move.a_turn

    while 1:  # Wait for human input
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if move.game_over and event.key == pygame.K_n:
                    return False
                if move.game_over and event.key == pygame.K_y:
                    return True
                if move.restart and event.key == pygame.K_n:
                    move.restart = False
                if move.restart and event.key == pygame.K_y:
                    return True
                if event.key == pygame.K_r:
                    move.restart = True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                if move.game_over:
                    pass
                elif move.restart:
                    pass
                elif not move.selected:
                    if move.a_turn:
                        for piece in tafl.Attackers:
                            if piece.rect.collidepoint(pos):
                                move.select(piece)
                                tafl.Current.add(piece)
                    else:
                        for piece in tafl.Defenders:
                            if piece.rect.collidepoint(pos):
                                move.select(piece)
                                tafl.Current.add(piece)

                else:
                    if tafl.Current.sprites()[0].rect.collidepoint(pos):
                        move.select(tafl.Current.sprites()[0])
                        tafl.Current.empty()
                        #pygame.display.flip()
                    elif move.is_valid_move(pos, tafl.Current.sprites()[0]):
                        if tafl.Current.sprites()[0] in tafl.Kings:
                            move.king_escaped(tafl.Kings)
                        if move.a_turn:
                            move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
                        else:
                            move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)
                        move.end_turn(tafl.Current.sprites()[0])
                        tafl.Current.empty()

        """Text to display on bottom of game."""
        text2 = None
        if move.a_turn:
            text = "Attacker's Turn"
        if not move.a_turn:
            text = "Defender's Turn"
        #if move.escaped:
        #    text = "King escaped! Defenders win!"
        #    text2 = "Play again? y/n"
        #if move.king_killed:
        #    text = "King killed! Attackers win!"
        #    text2 = "Play again? y/n"
        if move.restart:
            text = "Restart game? y/n"
        tafl.update_image(screen, board, move, text, text2)
        pygame.display.flip()

        if current_turn != move.a_turn: # turn ended
            return True



def do_best_move(move,model,sample_frac=1.0):
    """ Function to try all possible moves and select the best according to the model provided.

        Args: 
              move: talf game state and valid moves.
              model: Keras model used for predicting all possible moves.
              sample_frac: Fraction of pieces AND fraction of their moves to consider, for speed.
                           Default 1.0 considers all possible pieces and moves.
    """

    game_state = game_state_to_array() # Preserves the current game state

    if move.a_turn:
        pieces = tafl.Attackers
        king = (tafl.Kings.sprites()[0].x_tile, tafl.Kings.sprites()[0].y_tile)
    else:
        pieces = tafl.Defenders
        # If King can win, do it.
        for king in tafl.Kings:
            move.select(king)
            tafl.Current.add(king)
            for pos in [(0,0),(0,10),(10,0),(10,10), (0,1),(1,0),(0,9),(1,10),(10,1),(9,0),(10,9),(9,10)]:
                if pos in move.vm:
                    if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                        move.king_escaped(tafl.Kings)
                    if move.a_turn:
                        move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
                    else:
                        move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)
                    move.end_turn(tafl.Current.sprites()[0])
                    tafl.Current.empty()
                    return game_state,1.0
            move.select(king)
            tafl.Current.empty()

    if(len(pieces)==0): return game_state,0.0

    best_score = -1.0
    best_piece = pieces.sprites()[0]
    best_move = None
    best_game_state = None
    for piece in pieces:
        if random.random() > sample_frac: continue
        move.select(piece) # Move class defines all possible valid moves
        tafl.Current.add(piece)
        if len(move.vm)==0: # No valid moves for this piece, move on
            move.select(piece)
            tafl.Current.empty()
            continue
        else:
            for m in move.vm:
                if move.a_turn and move.kill_king(king[0],king[1],pieces,m[0],m[1]): # Move would kill king, do it.
                    best_score = 1.0
                    best_piece = piece
                    best_move = m
                    break
                if random.random() > sample_frac: continue
                # Swap game state for candidate move
                temp = game_state[piece.x_tile][piece.y_tile]
                game_state[piece.x_tile][piece.y_tile] = 0
                game_state[m[0]][m[1]] = temp

                try: # model.predict crashed once... 
                    score = model.predict(game_state.reshape(1,11*11))[0][0]
                except:
                    score = -1.0
                    #score = random.random()

                if score >= best_score:
                    best_score = score
                    best_piece = piece
                    best_move  = m

                # Reverse swap to restore game state
                temp = game_state[m[0]][m[1]]
                game_state[piece.x_tile][piece.y_tile] = temp
                game_state[m[0]][m[1]] = 0
        move.select(piece) # Deselect
        tafl.Current.empty()

    if best_piece is None or best_move is None: return game_state,0.0

    move.select(best_piece)
    tafl.Current.add(best_piece)

    if move.is_valid_move(best_move,tafl.Current.sprites()[0], True):
        if tafl.Current.sprites()[0] in tafl.Kings:
            move.king_escaped(tafl.Kings)
        if move.a_turn:
            move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
        else:
            move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)
        move.end_turn(tafl.Current.sprites()[0])
        tafl.Current.empty()

        # Just update the game state here for efficiency
        temp = game_state[best_piece.x_tile][best_piece.y_tile]
        game_state[best_piece.x_tile][best_piece.y_tile] = 0
        game_state[best_move[0]][best_move[1]] = temp

        #print(game_state,best_score)
        return game_state,best_score
    else:
        print("ERROR: Efficient move logic failed... Fix!")
        sys.exit(1)


def initialize_random_nn_model():
    """ Initialize Keras Deep Neural Networks models and print summary.
    """
    print("Initializing randomized NN model")
    model = Sequential()
    model.add(Dense(2*11*11, input_dim=11*11,kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(11*11, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.1))
    # Adding more test layers

    model.add(Dense(9, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(5, kernel_initializer='normal',activation='relu'))

    model.add(Dense(1,kernel_initializer='normal'))

    learning_rate = 0.001
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()
    return model

def game_state_to_array():
    """ 2D Numpy array representation of game state for ML model.
    """
    if tafl.Attackers is None or tafl.Defenders is None or tafl.Kings is None:
        print("Game not properly initialized.  Exiting.")
        sys.exit(1)
    arr = np.zeros((11,11),dtype=int)

    for p in tafl.Attackers:
        arr[p.x_tile][p.y_tile] = '1'
    for p in tafl.Defenders:
        arr[p.x_tile][p.y_tile] = '2'
    for p in tafl.Kings:
        arr[p.x_tile][p.y_tile] = '3'

    return arr

def unison_shuffled_copies(a, b):
    """ Used to shuffle the game states and corrected scores before retraining.
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def smooth_corrected_scores(corrected_scores,num_to_smooth=50):
    """ Smooth out the lead up to the final state for faster learning.
        I tried a number of strategies for speeding up training...
    """
    num_to_smooth = min(num_to_smooth, len(corrected_scores))
    for i in range(num_to_smooth-1):
        corrected_scores[-1*(i+2)] = (corrected_scores[-1*(i+2)] + 2*corrected_scores[-1*(i+1)]) / 3. # weighted average

def main():
    """Main training loop."""

    # TODO: Add command line option parsing.
    # True to let human players play
    human_attacker = False
    human_defender = False
    # True to display the pygame screen to watch the game
    interactive = human_attacker or human_defender or True
    # True to Update the attacker/defender models as you go
    train_attacker = True
    train_defender = True

    if human_attacker and train_attacker:
        print("Conflicting options human_attacker={} and train_attacker={}. Exiting.".format(human_attacker,train_attacker))
        sys.exit(1)
    if human_defender and train_defender:
        print("Conflicting options human_defender={} and train_defender={}. Exiting.".format(human_defender,train_defender))
        sys.exit(1)

    if interactive:
        pygame.init()
        screen = pygame.display.set_mode(tafl.WINDOW_SIZE)
    else:
        screen = None
    tafl.initialize_groups()

    num_train_games = 0
    version         = 4  # Used to track major changes/restarts

    attacker_model = None
    if not human_attacker:
        # Set to 0 to initialize random DNNs or used to load saved models:
        attacker_load   = 6420
        if attacker_load==0: attacker_model = initialize_random_nn_model()
        else:                attacker_model = load_model('models/attacker_model_after_{}_games_pass{}.h5'.format(attacker_load,version))

    defender_model = None
    if not human_defender:
        # Set to 0 to initialize random DNNs (-1 for defender has some basic rules) or used to load saved models:
        defender_load   = 6420
        #if defender_load == -1:  defender_model = None  # Defaults to mostly random + some extra King movements
        if defender_load == 0: defender_model = initialize_random_nn_model()
        else:                  defender_model = load_model('models/defender_model_after_{}_games_pass{}.h5'.format(defender_load,version))

    play = True
    while play:
        num_train_games += 1

        play, a_game_states,a_corrected_scores, d_game_states,d_corrected_scores = \
          run_game(attacker_model,defender_model,human_attacker,human_defender,screen)

        # Just some basic debugging to monitor how the training is progressing:
        print("{}, {}, {}, {}".format(num_train_games,len(a_corrected_scores)+len(d_corrected_scores),a_corrected_scores[-5:],d_corrected_scores[-5:]))

        if train_attacker and attacker_model is not None and len(a_corrected_scores)>0:
            smooth_corrected_scores(a_corrected_scores,num_to_smooth=max(20,int(len(a_corrected_scores)/1.)))
            a_game_states,a_corrected_scores = unison_shuffled_copies(np.array(a_game_states),np.array(a_corrected_scores))
            attacker_model.fit(np.array(a_game_states).reshape(-1,11*11),np.array(a_corrected_scores),epochs=1,batch_size=1,verbose=0)
        if train_defender and defender_model is not None and len(d_corrected_scores)>0:
            smooth_corrected_scores(d_corrected_scores,num_to_smooth=max(20,int(len(d_corrected_scores)/1.)))
            d_game_states,d_corrected_scores = unison_shuffled_copies(np.array(d_game_states),np.array(d_corrected_scores))
            defender_model.fit(np.array(d_game_states).reshape(-1,11*11),np.array(d_corrected_scores),epochs=1,batch_size=1,verbose=0)
        if(num_train_games%20==0):
            print('--- num games played: {}'.format(num_train_games))
            if train_attacker: attacker_model.save('models/attacker_model_after_{}_games_pass{}.h5'.format(num_train_games,version))
            if train_defender: defender_model.save('models/defender_model_after_{}_games_pass{}.h5'.format(num_train_games,version))

        if interactive:
            time.sleep(2)
        if num_train_games >= 100000: play = False

        tafl.cleanup()


if __name__ == '__main__':
    main()
