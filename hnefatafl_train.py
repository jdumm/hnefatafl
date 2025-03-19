"""
An extension of Hnefatafl to include AI training, AI vs AI, and Player vs AI (attacker or defender) modes.

A full description of the game can be found here: https://en.wikipedia.org/wiki/Tafl_games

Author: Jon Dumm
Date: 4/4/2019

"""

import os
import sys
from glob import glob
from timeit import default_timer as timer
# import pyximport; pyximport.install()
import pygame
from pygame.locals import *
import click
import time
import random
import numpy as np
import math
import itertools
import pickle
from collections import deque
from copy import deepcopy
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import TruncatedNormal

import hnefatafl as tafl
from stats_tracker import StatsTracker


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
        if len(move.vm) == 0:
            move.select(piece)
            tafl.Current.empty()
            continue
        else:
            pos = random.choice(tuple(move.vm))
            if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                if tafl.Current.sprites()[0] in tafl.Kings:
                    move.king_escaped(tafl.Kings)
                if move.a_turn:
                    move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                else:
                    move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)
                move.end_turn(tafl.Current.sprites()[0])
                tafl.Current.empty()
            break


def do_dummy_1_defender_move(move):
    """ Very basic rules for defender logic.  King moves to escape or next-to-escape tiles if an option.
        Basically rules to clear room and have King escape.
    """

    if move.a_turn:
        pieces = tafl.Attackers
    else:
        # If King can win, do it.
        for king in tafl.Kings:
            move.select(king)
            tafl.Current.add(king)
            kx, ky = king.x_tile, king.y_tile
            for pos in [(0, 0), (0, 10), (10, 0), (10, 10), (0, 1), (1, 0), (0, 9), (1, 10), (10, 1), (9, 0), (10, 9),
                        (9, 10)]:
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
            # If King can move to a higher position, do it some of the time.
            if len(move.vm) > 0 and random.random() < 0.02:
                pos = max(move.vm, key=lambda pos: max(pos[0], pos[1]))  # Max positional move of King
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

        # pieces = tafl.Defenders
        # If a defender is to the right or above the king, move down some of the time
        if random.random() < 0.6:
            for p in tafl.Defenders:
                if (p.x_tile == kx and p.y_tile > ky) or (p.y_tile == ky and p.x_tile > kx):
                    move.select(p)
                    tafl.Current.add(p)
                    if len(move.vm) == 0:
                        move.select(p)
                        tafl.Current.empty()
                        continue
                    else:
                        pos = min(move.vm,
                                  key=lambda pos: min(pos[0], pos[1]))  # Lowest possible move of King-obstructing piece
                        if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                            if tafl.Current.sprites()[0] in tafl.Kings:
                                move.king_escaped(tafl.Kings)
                            if move.a_turn:
                                move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings)
                            else:
                                move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings)

                            move.end_turn(tafl.Current.sprites()[0])
                            # move.select(p)
                            tafl.Current.empty()
                            return
                        move.select(p)
                        tafl.Current.empty()
        else:  # Otherwise do a completely random move
            do_random_move(move)
            return
        # return


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
            for pos in [(0, 0), (0, 10), (10, 0), (10, 10), (0, 1), (1, 0), (0, 9), (1, 10), (10, 1), (9, 0), (10, 9),
                        (9, 10)]:
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
            if random.random() < 0.10:  # Push King out from center if possible sometimes
                if len(move.vm) == 0:
                    break
                else:
                    for m in move.vm:
                        if abs(5 - m[0]) > 3 or abs(5 - m[1]) > 3:
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

        if len(move.vm) == 0:
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


def run_game(attacker_model=None, defender_model=None, human_attacker=False, human_defender=False, screen=None,
             game_name='Hnefatafl', sample_frac=1.0, frac_attackers_to_remove=0, frac_defenders_to_remove=0):
    """Start and run one game of computer attacker vs computer defender hnefatafl.
 
       Args:
           attacker_model: Keras model that can '.predict' based on the game state.  Used
                           to determine the best of available moves. Random moves by default.
           defender_model: Same as attacker_model but for defender.  
           screen: Optional, used to monitor matches in pygame.
           game_name: Variant of Tafl to play.
           sample_frac: Fraction of pieces AND fraction of their moves to consider, for speed.
                        Default 1.0 considers all possible pieces and moves.
           frac_attackers_to_remove: Fraction of Attacker's pieces to remove at random, for autobalancing.
           frac_defenders_to_remove: Fraction of Defender's pieces to remove at random, for autobalancing.
    """
    board = tafl.Board(game_name)
    move = tafl.Move()
    tafl.initialize_pieces(board)
    move.remove_random_pieces(tafl.Attackers, frac_attackers_to_remove)
    move.remove_random_pieces(tafl.Defenders, frac_defenders_to_remove)
    if game_name == "simple":
        if random.random() < 0.5:
            tafl.Attackers.sprites()[0].kill()
        else:
            tafl.Attackers.sprites()[1].kill()
    a_game_states = []
    a_predicted_scores = []
    d_game_states = []
    d_predicted_scores = []
    game_state_cache = deque(maxlen=20)
    play = True
    num_moves = 0
    while 1:
        num_moves += 1
        """Text to display on bottom of game."""
        if screen is not None and not human_attacker and not human_defender:
            tafl.update_image(screen, board, move, "Red: {}".format(len(tafl.Attackers.sprites())),
                              "Blue: {}".format(len(tafl.Defenders.sprites())))
        if screen is not None:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pass
        if (num_moves >= 100):
            print("--- Draw game after {} moves".format(num_moves))
            a_predicted_scores.append(0.0)
            d_predicted_scores.append(0.0)
            return play, a_game_states, a_predicted_scores[1:], d_game_states, d_predicted_scores[1:]

        if move.a_turn:
            if game_name.lower() == "simple":
                move.a_turn = False
                continue
                # print("Attacker's Turn: Move {}".format(num_moves))
            if human_attacker:
                play = do_human_turn(screen, board, move)
            elif attacker_model is None:
                game_state = do_random_move(move)
                predicted_score = (random.random() - 0.5) * 2
                a_game_states.append(game_state)
                a_predicted_scores.append(predicted_score)
            else:
                if human_defender: time.sleep(0.5)
                game_state, predicted_score = do_best_move(move,
                                                           attacker_model,
                                                           game_state_cache,
                                                           sample_frac=sample_frac,
                                                           enable_remove=True if game_name.lower() != 'simple' else False,
                                                           )
                # game_state,predicted_score = do_best_move(move,attacker_model,game_state_cache,sample_frac=1.00,screen=screen,board=board)
                a_game_states.append(game_state)
                a_predicted_scores.append(predicted_score)
        else:
            # print("Defender's Turn: Move {}".format(num_moves))
            if human_defender:
                play = do_human_turn(screen, board, move)
            elif defender_model is None:
                # game_state = do_mostly_random_but_strike_to_kill_move(move)
                game_state = do_dummy_1_defender_move(move)
                predicted_score = (random.random() - 0.5) * 2
                d_game_states.append(game_state)
                d_predicted_scores.append(predicted_score)
            else:
                if human_attacker: time.sleep(0.5)
                game_state, predicted_score = do_best_move(move,
                                                           defender_model,
                                                           game_state_cache,
                                                           sample_frac=sample_frac,
                                                           enable_remove=True if game_name.lower() != 'simple' else False,
                                                           )
                # game_state,predicted_score = do_best_move(move,defender_model,game_state_cache,sample_frac=1.00,screen=screen,board=board)
                d_game_states.append(game_state)
                d_predicted_scores.append(predicted_score)

        if move.escaped:
            text = "--- King escaped! Defenders win!"
            print(text)
            text2 = "Play again? y/n"
            a_predicted_scores.append(-1.0)
            d_predicted_scores.append(+1.0)
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, a_game_states, a_predicted_scores[1:], d_game_states, d_predicted_scores[
                                                                               1:]  # i.e. the corrected scores from RL
        if move.king_killed:
            text = "--- King killed! Attackers win!"
            print(text)
            text2 = "Play again? y/n"
            a_predicted_scores.append(+1.0)
            d_predicted_scores.append(-1.0)
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, a_game_states, a_predicted_scores[1:], d_game_states, d_predicted_scores[
                                                                               1:]  # i.e. the corrected scores from RL
        if move.restart:
            return play, a_game_states, a_predicted_scores[1:], d_game_states, d_predicted_scores[
                                                                               1:]  # i.e. the corrected scores from RL


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


def do_human_turn(screen, board, move):
    # print("Starting human turn")
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
                    move.game_over = True
                    if move.a_turn:
                        move.escaped = True
                    else:
                        move.king_killed = True
                    move.restart = True
                    move.a_turn = not move.a_turn
                    tafl.Current.empty()
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
                    elif move.is_valid_move(pos, tafl.Current.sprites()[0]):
                        if tafl.Current.sprites()[0] in tafl.Kings:
                            move.king_escaped(tafl.Kings)
                        if move.a_turn:
                            move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                        else:
                            move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)
                        move.end_turn(tafl.Current.sprites()[0])
                        tafl.Current.empty()

        """Text to display on bottom of game."""
        if move.a_turn:
            text = "Attacker's Turn"
        if not move.a_turn:
            text = "Defender's Turn"
        text2 = "Resign? (r)"
        # if move.escaped:
        #    text = "King escaped! Defenders win!"
        #    text2 = "Play again? y/n"
        # if move.king_killed:
        #    text = "King killed! Attackers win!"
        #    text2 = "Play again? y/n"
        if move.restart and screen:
            text = "Restart game? y/n"
        tafl.update_image(screen, board, move, text, text2)
        pygame.display.flip()

        if current_turn != move.a_turn:  # turn ended
            return True


def do_best_move(move, model, game_state_cache, sample_frac=1.0, screen=None, board=None, enable_remove=True):
    """ Function to try all possible moves and select the best according to the model provided.

        Args: 
              move: tafl game state and valid moves.
              model: Keras model used for predicting all possible moves.
              sample_frac: Fraction of pieces AND fraction of their moves to consider, for speed.
                           Default 1.0 considers all possible pieces and moves.
    """

    game_state = game_state_to_3d_array()
    game_state_cache.append(deepcopy(game_state))

    if move.a_turn:
        pieces = tafl.Attackers
        # TODO: Add logic to see if we can kill the king before skipping pieces!
    else:
        pieces = tafl.Defenders
        # If King can win, do it.
        for king in tafl.Kings:
            move.select(king)
            tafl.Current.add(king)
            # Can we win now?
            for pos in tafl.SPECIALSQS.difference([((tafl.DIM - 1) // 2, (tafl.DIM - 1) // 2)]):
                if pos in move.vm:
                    if move.is_valid_move(pos, tafl.Current.sprites()[0], True):
                        move.king_escaped(tafl.Kings)
                    if enable_remove:
                        move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)
                    move.end_turn(tafl.Current.sprites()[0])
                    tafl.Current.empty()
                    return game_state, 1.0
            move.select(king)
            tafl.Current.empty()

    if len(pieces) == 0:
        return game_state, 0.0

    best_score = -1000000.0
    best_piece = pieces.sprites()[0]
    best_move = None
    best_game_state = None
    best_vm = None
    for piece in pieces:
        if random.random() > sample_frac:
            continue
        if screen and board: time.sleep(1.0)
        move.select(piece)  # Move class defines all possible valid moves
        tafl.Current.add(piece)
        if len(move.vm) == 0:  # No valid moves for this piece, move on
            move.select(piece)
            tafl.Current.empty()
            continue
        else:
            num_best_so_far = 0
            for m in move.vm:
                if random.random() > sample_frac:
                    continue

                # Try candidate move
                if screen and board:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            sys.exit()
                    tafl.update_image(screen, board, move, "DEBUG MODE", "Score: {:0.4f}".format(0))
                    pygame.display.flip()
                    time.sleep(1.0)
                move.is_valid_move(m, tafl.Current.sprites()[0], True)
                if enable_remove:
                    if move.a_turn:
                        move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                    else:
                        move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)
                game_state = game_state_to_3d_array()
                if move.a_turn and move.king_killed:  # Move would kill king, do it.
                    best_score = 1.0
                    best_piece = piece
                    best_move = m
                    best_vm = move.vm
                # score = model.predict(game_state.reshape(1, tafl.DIM * tafl.DIM * 3))[0][0]
                # TODO, can I call predict once on the full set of moves?  At least per piece.
                score = model.predict(game_state.reshape(1, tafl.DIM, tafl.DIM, 3))[0][0]
                if screen and board:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            sys.exit()
                    tafl.update_image(screen, board, move, "DEBUG MODE", "Score: {:0.4f}".format(score))
                    pygame.display.flip()
                    time.sleep(0.5)

                # Reverse candidate move to log the piece location correctly
                # print("len pieces1", len(tafl.Pieces))
                move.undo(tafl.Current.sprites()[0])
                # print("len pieces2", len(tafl.Pieces))

                if score == best_score: score = score + random.uniform(-0.01,
                                                                       0.01)  # Add a little noise if they are exactly equal
                # Find best score but don't let it keep repeating the same states
                if score > best_score and not next(
                        (True for elem in itertools.islice(game_state_cache, 4, game_state_cache.maxlen) if
                         np.array_equal(elem, game_state)), False):
                    best_score = score
                    best_piece = piece
                    best_move = m
                    best_vm = move.vm
                    # print("best stats: ",best_score,best_piece,(best_piece.x_tile,best_piece.y_tile),best_move)
                    # print("Valid moves: ",move.vm)

        move.select(piece)  # Deselect
        tafl.Current.empty()

    if best_piece is None or best_move is None:
        print("NO BEST MOVE! No moves at all or a Draw?")
        return game_state, 0.0

    # print(" ",best_score)
    move.select(best_piece)
    tafl.Current.add(best_piece)

    if best_vm != move.vm:
        print('best vm: ', best_vm)
        print('move vm: ', move.vm)

    if move.is_valid_move(best_move, tafl.Current.sprites()[0], True):
        if tafl.Current.sprites()[0] in tafl.Kings:
            move.king_escaped(tafl.Kings)
        if enable_remove:
            if move.a_turn:
                move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
            else:
                move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)

        game_state = game_state_to_3d_array()

        move.end_turn(tafl.Current.sprites()[0])
        tafl.Current.empty()

        # print(game_state,best_score)
        return game_state, best_score
    else:
        print("ERROR: Best move logic failed... Fix! Debugging info follows:")
        print("BEST MOVE", best_move)
        print("Current", tafl.Current.sprites()[0], (best_piece.x_tile, best_piece.y_tile), move.row, move.col)
        print("Valid moves", move.vm)
        print("reValid moves", move.valid_moves(best_piece.special_sqs, debug=True))
        # do_human_turn(screen, board, move)
        time.sleep(30)
        sys.exit(1)


def initialize_random_nn_model():
    """ Initialize Keras Deep Neural Networks models and print summary.
    """
    print("Initializing randomized NN model")
    model = Sequential()
    model.add(
        Dense(2 * tafl.DIM * tafl.DIM, input_dim=tafl.DIM * tafl.DIM, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(tafl.DIM * tafl.DIM, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    # Adding more test layers

    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    learning_rate = 0.05
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()
    return model


def initialize_random_nn_model_v2():
    """ Initialize Keras Deep Neural Networks models and print summary.
    """
    print("Initializing randomized CNN model")
    std = 0.05
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(11, 11, 1), padding='same', activation='relu', strides=(1, 1),
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))
    # model.add(Conv2D( 64, (3,3), padding='same',activation='relu', strides=(1,1), kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))
    # model.add(Conv2D(128, (3,3), padding='same',activation='relu', strides=(2,2), kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))
    # model.add(Dropout(0.1))
    # model.add(Conv2D(64, (3,3), activation='relu',kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))
    model.add(Flatten())
    # model.add(Dropout(0.1))
    # Adding more test layers

    model.add(Dense(64, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))
    model.add(Dropout(0.01))
    model.add(Dense(32, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))
    model.add(Dense(16, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))

    model.add(Dense(1, kernel_initializer=TruncatedNormal(mean=0.0, stddev=std, seed=None)))

    learning_rate = 0.001
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()
    return model


def initialize_random_nn_model_3d():
    """ Initialize Keras Deep Neural Networks models and print summary.
        Architecture for the 3d game states.
    """
    print("Initializing randomized NN model")
    model = Sequential()
    model.add(Dense(2 * tafl.DIM * tafl.DIM * 3, input_dim=tafl.DIM * tafl.DIM * 3, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(tafl.DIM * tafl.DIM * 3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.01))
    # Adding more test layers

    model.add(Dense(tafl.DIM * 3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(tafl.DIM // 2, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    learning_rate = 0.005
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()
    return model


def initialize_random_nn_model_3d_dense_v2():
    """ Initialize Keras Deep Neural Networks models and print summary.
        Architecture for the 3d game states.
    """
    print("Initializing randomized NN model")
    model = Sequential()
    model.add(Dense(tafl.DIM * tafl.DIM * 3, input_dim=tafl.DIM * tafl.DIM * 3, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(tafl.DIM * 3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal'))

    learning_rate = 0.001
    momentum = 0.9

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()
    return model


def initialize_random_nn_model_3d_dense_v3():
    """ Initialize Keras Deep Neural Networks models and print summary.
        Architecture for the 3d game states, boxy layout.
    """
    print("Initializing randomized NN model")
    model = Sequential()
    model.add(Dense(tafl.DIM * tafl.DIM * 3, input_dim=tafl.DIM * tafl.DIM * 3, kernel_initializer='normal',
                    activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(tafl.DIM * tafl.DIM * 3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(tafl.DIM * tafl.DIM * 3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(tafl.DIM * tafl.DIM * 3, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    learning_rate = 0.010
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()
    return model


def initialize_random_cnn_model_3d_v1():
    """ Initialize Keras CNN with fully connected layers and print summary.
        Architecture for the 3d games states.
    """
    print("Initializing randomized CNN model 3d game states v1")
    print((3, tafl.DIM, tafl.DIM))
    model = Sequential()
    model.add(
        Conv2D(32,
               (3, 3),
               input_shape=(tafl.DIM, tafl.DIM, 3),
               padding='same',
               activation='relu',
               strides=(1, 1),
               )
    )
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(2 * tafl.DIM * tafl.DIM, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    #learning_rate = 0.001
    learning_rate = 0.1
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()
    return model


# def game_state_to_array():
#     """ 2D Numpy array representation of game state for ML model.
#
#         DEPRECATED after moving to 3d game state
#     """
#     if tafl.Attackers is None or tafl.Defenders is None or tafl.Kings is None:
#         print("Game not properly initialized.  Exiting.")
#         sys.exit(1)
#     arr = np.zeros((tafl.DIM, tafl.DIM), dtype=int)
#
#     for p in tafl.Attackers:
#         arr[p.x_tile][p.y_tile] = '1'
#     for p in tafl.Defenders:
#         arr[p.x_tile][p.y_tile] = '2'
#     for p in tafl.Kings:
#         arr[p.x_tile][p.y_tile] = '3'
#
#     return arr


A_DIM = 0
D_DIM = 2
K_DIM = 1


def game_state_to_3d_array():
    """ 3D Numpy array representation of game state for ML model.
        2 spatial dimensions + 1 for piece type (Attacker, Defender, King).
        We'll tuck the King Dimension in between the others.
    """
    if tafl.Attackers is None or tafl.Defenders is None or tafl.Kings is None:
        print("Game not properly initialized.  Exiting.")
        sys.exit(1)
    arr = np.zeros((tafl.DIM, tafl.DIM, 3), dtype=int)

    for p in tafl.Attackers:
        arr[p.x_tile][p.y_tile][A_DIM] = '1'
    for p in tafl.Kings:
        arr[p.x_tile][p.y_tile][K_DIM] = '1'
    for p in tafl.Defenders:
        arr[p.x_tile][p.y_tile][D_DIM] = '1'

    return arr


def game_state_3d_to_string():
    """ 2D string representation of game state for us humans.
    """
    if tafl.Attackers is None or tafl.Defenders is None or tafl.Kings is None:
        print("Game not properly initialized.  Exiting.")
        sys.exit(1)
    #grid = ['.'*tafl.DIM+'\n']*tafl.DIM
    s = ['.' * tafl.DIM] * tafl.DIM
    grid = []
    for l in s:
        grid.append(list(l))
    for p in tafl.Attackers:
        grid[p.x_tile][p.y_tile] = 'a'
    for p in tafl.Kings:
        grid[p.x_tile][p.y_tile] = 'k'
    for p in tafl.Defenders:
        grid[p.x_tile][p.y_tile] = 'd'
    return grid


def expand_game_states_symmetries(game_states):
    """ Equivalent game states include all 4 rotations by 90 deg,
        as well as the 4 rotations of the mirror symmetry.
        This returns an expanded list of game states that is
        8x larger than the original!  Yeah, faster learning!
    """
    # Append all possible 90deg rotations:
    game_states_temp = [np.rot90(gs) for gs in game_states]
    game_states = np.concatenate((game_states, game_states_temp))
    game_states_temp = [np.rot90(gs) for gs in game_states_temp]
    game_states = np.concatenate((game_states, game_states_temp))
    game_states_temp = [np.rot90(gs) for gs in game_states_temp]
    game_states = np.concatenate((game_states, game_states_temp))

    # Now mirror all possible rotations
    game_states_temp = [np.flip(gs, axis=0) for gs in game_states]
    game_states = np.concatenate((game_states, game_states_temp))
    return game_states


def unison_shuffled_copies(a, b):
    """ Used to shuffle the game states and corrected scores before retraining.
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def smooth_corrected_scores(corrected_scores, num_to_smooth=50):
    """ Smooth out the lead up to the final state for faster learning.
        I tried a number of strategies for speeding up training...
    """
    num_to_smooth = min(num_to_smooth, len(corrected_scores))
    for i in range(num_to_smooth - 1):
        corrected_scores[-1 * (i + 2)] = (corrected_scores[-1 * (i + 2)] + 2 * corrected_scores[
            -1 * (i + 1)]) / 3.  # weighted average


def smooth_corrected_scores_exp(corrected_scores, dynamic=True, decay_constant=5.):
    """ Smooth out the lead up to the final state for faster learning.
        Exponential strategy.  dynamic mode = True means exp weight scales with game length.
        False means the decay_constant in terms of num game states is used.  
    """
    if dynamic: decay_constant = float(len(corrected_scores)) / 2.  # 1/2 game length
    for i in range(len(corrected_scores) // 2 - 1):
        corrected_scores[-1 * (i + 2)] = (corrected_scores[-1 * (i + 2)] + math.exp(-i / decay_constant) *
                                          corrected_scores[-1 * (i + 1)]) / (1. + math.exp(-i / decay_constant))


@click.command()
@click.option('-g', '--game-name', default='Hnefatafl', help='Name of Tafl variant to play')
@click.option('-ha/-aa', '--human-attacker/--ai-attacker', default=False, help='Set to play attacker manually')
@click.option('-hd/-ad', '--human-defender/--ai-defender', default=False, help='Set to play defender manually')
@click.option('-i/-b', '--interactive/--batch', default=False, help='Set true in order to watch AI vs AI matches')
@click.option('-ta/-na', '--train-attacker/--no-train-attacker', default=False,
              help='Set to update attacker AI after each game')
@click.option('-td/-nd', '--train-defender/--no-train-defender', default=False,
              help='Set to update defender AI after each game')
@click.option('-dt/-st', '--dynamic-train/--static-train', default=False, help='Set to pause defender AI when lopsided')
@click.option('-c', '--cache-model-every', default=100, help='Cache the Keras DNN model every so many games')
@click.option('-e/-ne', '--exit-after-cache/--no-exit-after-cache', default=False, help='Exit after model cache step to allow restart')
@click.option('-s/-ns', '--use-symmetry/--no-symmetry', default=False,
              help='Set to train using symmetrical board states')
@click.option('-al', '--attacker-load', default=0, help='Attacker model file num to load')
@click.option('-dl', '--defender-load', default=0, help='Defender model file num to load')
@click.option('-sl', '--stats-load', default=0, help='Stats model file num to load')
@click.option('-ll/-nl', '--load-latest/--not-latest', default=False, help='Set to search and use latest models/stats files')
@click.option('-v', '--version', default=7, help='Model version number')
def main(game_name, human_attacker, human_defender, interactive, train_attacker, train_defender, dynamic_train,
         cache_model_every, exit_after_cache, use_symmetry,
         attacker_load, defender_load, stats_load, load_latest, version):
    """Main training loop."""

    global king_is_special
    king_is_special = False

    log_level = 1

    # True to let human players play
    # human_attacker = False
    # human_defender = False
    # True to display the pygame screen to watch the game
    interactive = human_attacker or human_defender or interactive
    # True to Update the attacker/defender models as you go
    # train_attacker = True
    # train_defender = False

    # cache_model_every = 50 # games
    # use_symmetry = False

    if human_attacker and train_attacker:  # Sorry, I can't train humans
        print("Conflicting options human_attacker={} and train_attacker={}. Exiting.".format(human_attacker,
                                                                                             train_attacker))
        sys.exit(1)
    if human_defender and train_defender:
        print("Conflicting options human_defender={} and train_defender={}. Exiting.".format(human_defender,
                                                                                             train_defender))
        sys.exit(1)
    if attacker_load>0 or defender_load>0 and load_latest:
        print("Conflicting options attacker_load={} or defender_load={} with load_latest set. Exiting.".format(attacker_load,
                                                                                                               defender_load))
        sys.exit(1)

    train_attacker_orig = train_attacker
    train_defender_orig = train_defender

    if interactive:
        pygame.init()
        screen = pygame.display.set_mode(tafl.WINDOW_SIZE)
    else:
        screen = None

    tafl.initialize_groups()
    temp_board = tafl.Board(game_name)  # Just used to initialize global DIM for game_name...

    num_train_games_attacker = 0
    num_train_games_defender = 0
    # version         = 6  # Used to track major changes/restarts

    save_dir = 'models_{}_v{}'.format(game_name.lower(), version)
    if not human_attacker or not human_defender: os.makedirs(save_dir, exist_ok=True)

    stats_tracker_loaded = False
    attacker_model = None
    if not human_attacker:
        if load_latest:
            a_model_files = glob(save_dir + '/attacker_model_*_games.h5')
            if len(a_model_files) > 0:
                attacker_load = max([int(f.split("_")[4]) for f in a_model_files])  # Parse filenames and get latest
            else:
                attacker_load = 0
        if attacker_load == 0:
            attacker_model = initialize_random_cnn_model_3d_v1()
        else:
            attacker_model = load_model('{}/attacker_model_{}_games.h5'.format(save_dir, attacker_load))
            num_train_games_attacker = attacker_load

    defender_model = None
    if not human_defender:
        if load_latest:
            d_model_files = glob(save_dir + '/defender_model_*_games.h5')
            if len(d_model_files) > 0:
                defender_load = max([int(f.split("_")[4]) for f in d_model_files])  # Parse filenames and get latest
            else:
                defender_load = 0
        if defender_load == -1:
            defender_model = None  # Defaults to mostly random + some extra King movements
        elif defender_load == 0:
            defender_model = initialize_random_cnn_model_3d_v1()
        else:
            defender_model = load_model('{}/defender_model_{}_games.h5'.format(save_dir, defender_load))
            num_train_games_defender = defender_load

    stats = None
    if load_latest:
        stats_files = glob(save_dir + '/StatsTracker_*_games.pkl')
        if len(stats_files) > 0:
            stats_load = max([int(f.split("_")[3]) for f in stats_files])
        else:
            stats_load = 0
    if stats_load > 0 and (not human_attacker or not human_defender):
        stats = pickle.load(open('{}/StatsTracker_{}_games.pkl'.format(save_dir, stats_load), 'rb'))
    else:  # no previous stats
        stats = StatsTracker(200)

    play = True
    while play:
        if train_attacker: num_train_games_attacker += 1
        if train_defender: num_train_games_defender += 1

        start = timer()

        sample_frac = 0.90
        # TODO: Separate & based on number of trained games.  Then add noise to decision.
        sample_frac_attacker = 0.90
        sample_frac_defender = 0.90

        frac_attackers_to_remove = 0.00
        frac_defenders_to_remove = 0.00
        # Make a decision about whether or not to keep training Defender
        a_win_rate = (stats.a_win_rate_window() + stats.draw_rate_window() / 2.)  # Draws count half
        d_win_rate = 1 - a_win_rate
        if dynamic_train and a_win_rate < 0.40:
            train_defender = False  # Too smart, pause training
            if a_win_rate < 0.30:  # experimental
                frac_defenders_to_remove = 0.50 - a_win_rate
                if log_level>0:
                    print(f"a_win_rate is {a_win_rate}, removing {frac_defenders_to_remove} of defenders")
        else:
            train_defender = train_defender_orig

        if dynamic_train and d_win_rate < 0.40:
            train_attacker = False  # Too smart, pause training
            if d_win_rate < 0.30:  # experimental
                frac_attackers_to_remove = 0.50 - d_win_rate
                if log_level>0:
                    print(f"d_win_rate is {d_win_rate}, removing {frac_attackers_to_remove} of attackers")
        else:
            train_attacker = train_attacker_orig

        play, a_game_states, a_corrected_scores, d_game_states, d_corrected_scores = \
            run_game(attacker_model, defender_model, human_attacker, human_defender, screen, game_name,
                     sample_frac, frac_attackers_to_remove, frac_defenders_to_remove)

        end = timer()
        game_duration = end - start

        # Just some basic debugging to monitor how the training is progressing:
        if log_level>0:
            print(
                """Attacker has played:     {} games,\nDefender has played:     {} games,\nNum moves this game:     {} ({:0.3f} sec)"""
                .format(num_train_games_attacker, num_train_games_defender,
                        len(a_corrected_scores) + len(d_corrected_scores), game_duration))

        # Add Attacker outcome to the StatsTracker
        if len(a_corrected_scores) > 0:  # AI Attacker and/or defender
            stats.add_game_results(a_corrected_scores[-1], len(a_corrected_scores) + len(d_corrected_scores),
                                   game_duration)
            if log_level>0:
                print(stats)
        elif len(d_corrected_scores) > 0:  # AI defender only
            stats.add_game_results(-1 * d_corrected_scores[-1], len(a_corrected_scores) + len(d_corrected_scores),
                                   game_duration)
            if log_level>0:
                print(stats)
        # else: # PvP stats not tracked

        if train_attacker and attacker_model is not None and len(a_corrected_scores) > 0:
            # print(np.array_repr( a_game_states[-2] ).replace('\n', ''))
            # print(np.array_repr( a_game_states[-1] ).replace('\n', ''))
            if log_level>0:
                print("""Last Attacker states: {}""".format(
                      ' '.join(['{:+0.4f}'.format(entry) for entry in a_corrected_scores[-16:-1]])))
            smooth_corrected_scores_exp(a_corrected_scores)
            if log_level>0:
                print("""            Smoothed: {}""".format(
                      ' '.join(['{:+0.4f}'.format(entry) for entry in a_corrected_scores[-15:]])))
            a_game_states, a_corrected_scores = unison_shuffled_copies(np.array(a_game_states),
                                                                       np.array(a_corrected_scores))
            if use_symmetry:
                a_game_states = expand_game_states_symmetries(a_game_states)
                a_corrected_scores = np.tile(a_corrected_scores, 8)
            # attacker_model.fit(a_game_states.reshape(-1,11*11),a_corrected_scores,epochs=1,batch_size=1,verbose=0)
            # attacker_model.fit(a_game_states.reshape(-1,11,11,1),a_corrected_scores,epochs=1,batch_size=1,verbose=0)
            attacker_model.fit(a_game_states.reshape(-1, tafl.DIM, tafl.DIM, 3), #.reshape(-1, tafl.DIM * tafl.DIM * 3),
                               a_corrected_scores,
                               epochs=1,
                               batch_size=1,
                               verbose=0)

        if train_defender and defender_model is not None and len(d_corrected_scores) > 0:
            # print(np.array_repr( d_game_states[-2] ).replace('\n', ''))
            # print(np.array_repr( d_game_states[-1] ).replace('\n', ''))
            if log_level>0:
                print("""Last Defender states: {}""".format(
                      ' '.join(['{:+0.4f}'.format(entry) for entry in d_corrected_scores[-16:-1]])))
            smooth_corrected_scores_exp(d_corrected_scores)
            if log_level>0:
                print("""            Smoothed: {}""".format(
                ' '.join(['{:+0.4f}'.format(entry) for entry in d_corrected_scores[-15:]])))
            d_game_states, d_corrected_scores = unison_shuffled_copies(np.array(d_game_states),
                                                                       np.array(d_corrected_scores))
            if use_symmetry:
                d_game_states = expand_game_states_symmetries(d_game_states)
                d_corrected_scores = np.tile(d_corrected_scores, 8)
            # defender_model.fit(d_game_states.reshape(-1,11*11),d_corrected_scores,epochs=1,batch_size=1,verbose=0)
            # defender_model.fit(d_game_states.reshape(-1,11,11,1),d_corrected_scores,epochs=1,batch_size=1,verbose=0)
            defender_model.fit(d_game_states.reshape(-1, tafl.DIM, tafl.DIM, 3), #.reshape(-1, tafl.DIM * tafl.DIM * 3),
                               d_corrected_scores,
                               epochs=1,
                               batch_size=1,
                               verbose=0,)

        if (stats.num_games_total() % cache_model_every == 0):  # Save every cache_model_every games
            # print('--- num games played: {}'.format(stats.num_games_total()))
            if num_train_games_attacker > 0: attacker_model.save(
                '{}/attacker_model_{}_games.h5'.format(save_dir, num_train_games_attacker))
            if num_train_games_defender > 0: defender_model.save(
                '{}/defender_model_{}_games.h5'.format(save_dir, num_train_games_defender))
            if train_attacker or train_defender: pickle.dump(stats, open(
                '{}/StatsTracker_{}_games.pkl'.format(save_dir, stats.num_games_total()), 'wb'))
            if exit_after_cache:  # To avoid possible memory leak for long training sessions
                sys.exit()

        if interactive:
            time.sleep(2)
        if max(num_train_games_attacker,
               num_train_games_defender) >= 1000000: play = False  # Hardcoded cutoff just to make sure things don't go too crazy.

        tafl.cleanup()


if __name__ == '__main__':
    main()
