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
import tensorflow as tf
from keras import Sequential, Model, Input
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Add, Concatenate
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.initializers import TruncatedNormal

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
             game_name='Hnefatafl', sample_frac=1.0, attacker_temp=0.1, defender_temp=0.1, frac_attackers_to_remove=0, frac_defenders_to_remove=0, epsilon=0.0):
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
        if num_moves >= 100 or (game_name.lower() == "simple" and num_moves > 4):
            print("--- Draw game after {} moves".format(num_moves))
            # Replace last predictions with draw values instead of appending
            if len(a_predicted_scores) > 0:
                a_predicted_scores[-1] = -0.5  # Slight penalty for draws
            if len(d_predicted_scores) > 0:
                d_predicted_scores[-1] = -0.5  # Slight penalty for draws
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores

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
                                                           temperature=attacker_temp,
                                                           enable_remove=True if game_name.lower() != 'simple' else False,
                                                           screen=screen,
                                                           board=board,
                                                           epsilon=epsilon)
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
                                                           temperature=defender_temp,
                                                           enable_remove=True if game_name.lower() != 'simple' else False,
                                                           screen=screen,
                                                           board=board,
                                                           epsilon=epsilon)
                d_game_states.append(game_state)
                d_predicted_scores.append(predicted_score)

        if move.escaped:
            text = "--- King escaped! Defenders win!"
            print(text)
            text2 = "Play again? y/n"
            # Scale reward based on number of moves - faster wins get higher rewards
            
            # Replace last predictions instead of appending
            if len(a_predicted_scores) > 0:
                a_predicted_scores[-1] = -1.0
            if len(d_predicted_scores) > 0:
                d_predicted_scores[-1] = +1.0
                
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
        if move.king_killed:
            text = "--- King killed! Attackers win!"
            print(text)
            text2 = "Play again? y/n"
            # Scale reward based on number of moves - faster wins get higher rewards
            
            # Replace last predictions instead of appending
            if len(a_predicted_scores) > 0:
                a_predicted_scores[-1] = +1.0 
            if len(d_predicted_scores) > 0:
                d_predicted_scores[-1] = -1.0
                
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
        if move.restart:
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores


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


def do_best_move(move, model, game_state_cache, sample_frac=1.0, screen=None, board=None, enable_remove=True, temperature=0.1, epsilon=0.0):
    """ Function to try all possible moves and select the best according to the model provided.

    Uses batch inference for efficiency: collects all candidate states first, then
    makes a single batched prediction call instead of one call per move.

    Args:
        epsilon: Probability of making a random move (exploration)
    """
    game_state = game_state_to_3d_array()
    game_state_cache.append(deepcopy(game_state))

    # For simple game, we want to be more focused in our exploration
    simple_game = (tafl.DIM == 5)
    if simple_game:
        sample_frac = 1.0  # Always consider all moves in simple game

    if move.a_turn:
        pieces = tafl.Attackers
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

    # Epsilon-greedy: random move with probability epsilon
    if epsilon > 0 and random.random() < epsilon:
        # Select a random piece with valid moves
        valid_pieces = []
        for piece in pieces:
            move.select(piece)
            tafl.Current.add(piece)
            if len(move.vm) > 0:
                valid_pieces.append((piece, list(move.vm)))
            move.select(piece)
            tafl.Current.empty()

        if valid_pieces:
            piece, valid_moves = random.choice(valid_pieces)
            m = random.choice(valid_moves)
            move.select(piece)
            tafl.Current.add(piece)
            if move.is_valid_move(m, tafl.Current.sprites()[0], True):
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
                return game_state, 0.0  # Return neutral score for random moves
            tafl.Current.empty()

    # Phase 1: Collect all candidate (piece, move, state) tuples
    candidates = []  # List of (piece, move_pos, game_state, vm_snapshot)
    king_kill_candidate = None  # Track if we find a king-killing move

    for piece in pieces:
        if random.random() > sample_frac:
            continue
        move.select(piece)
        tafl.Current.add(piece)
        if len(move.vm) == 0:
            move.select(piece)
            tafl.Current.empty()
            continue

        vm_snapshot = set(move.vm)  # Snapshot valid moves for this piece

        for m in move.vm:
            if random.random() > sample_frac:
                continue

            # Try candidate move
            move.is_valid_move(m, tafl.Current.sprites()[0], True)
            if enable_remove:
                if move.a_turn:
                    move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                else:
                    move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)

            candidate_state = game_state_to_3d_array()

            # Check for immediate win (king killed)
            if move.a_turn and move.king_killed:
                king_kill_candidate = (piece, m, candidate_state, vm_snapshot)
                # Don't undo yet - we'll execute this move
                break

            # Check for move repetition before adding to candidates
            is_repeat = next(
                (True for elem in itertools.islice(game_state_cache, 0, game_state_cache.maxlen) if
                 np.array_equal(elem, candidate_state)), False)

            if not is_repeat:
                candidates.append((piece, m, candidate_state, vm_snapshot))

            # Reverse candidate move
            move.undo(tafl.Current.sprites()[0])

        move.select(piece)
        tafl.Current.empty()

        # If we found a king-killing move, execute it immediately
        if king_kill_candidate:
            piece, m, candidate_state, _ = king_kill_candidate
            best_score = 1.0

            if screen and board:
                tafl.update_image(screen, board, move,
                                f"Selected move ({piece.x_tile}, {piece.y_tile})->({m[0]}, {m[1]})",
                                f"Score: {best_score:.2f}",
                                highlight_pos=m,
                                highlight_score=best_score)
                pygame.display.flip()
                time.sleep(0.8)

            return candidate_state, best_score

    # No valid non-repeating moves found
    if len(candidates) == 0:
        print("NO BEST MOVE! No moves at all or a Draw?")
        return game_state, 0.0

    # Phase 2: Single batch prediction for all candidates
    states_batch = np.array([c[2] for c in candidates])
    scores = model.predict(states_batch, verbose=0).flatten()

    # Add temperature noise for exploration
    if temperature > 0:
        scores = scores + np.random.normal(0, temperature, size=scores.shape)

    # Phase 3: Select best move
    best_idx = np.argmax(scores)
    best_piece, best_move_pos, best_game_state, best_vm = candidates[best_idx]
    best_score = scores[best_idx]

    # Execute the best move
    move.select(best_piece)
    tafl.Current.add(best_piece)

    if best_vm != move.vm:
        print('best vm: ', best_vm)
        print('move vm: ', move.vm)

    if move.is_valid_move(best_move_pos, tafl.Current.sprites()[0], True):
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

        # Display final selected move if in interactive mode
        if screen and board:
            tafl.update_image(screen, board, move,
                            f"Selected move ({best_piece.x_tile}, {best_piece.y_tile})->({best_move_pos[0]}, {best_move_pos[1]})",
                            f"Score: {best_score:.2f}",
                            highlight_pos=best_move_pos,
                            highlight_score=best_score)
            pygame.display.flip()
            time.sleep(0.8)

        return game_state, best_score
    else:
        print("ERROR: Best move logic failed... Fix! Debugging info follows:")
        print("BEST MOVE", best_move_pos)
        print("Current", tafl.Current.sprites()[0], (best_piece.x_tile, best_piece.y_tile), move.row, move.col)
        print("Valid moves", move.vm)
        print("reValid moves", move.valid_moves(best_piece.special_sqs, debug=True))
        time.sleep(30)
        sys.exit(1)


def initialize_random_cnn_model_3d_sonnet(num_channels=3, use_batchnorm=True):
    """ Initialize Keras CNN model optimized for 7x7 board game learning.

    Architecture features:
    - Multiple channels to represent different piece types and board positions
    - Residual connections to help learn complex spatial relationships
    - Multiple convolutional layers with different kernel sizes to capture both local and broader patterns
    - Batch normalization for training stability
    - Dropout for regularization

    Args:
        num_channels: Number of input channels (3 for legacy, 6 for enhanced encoding)
        use_batchnorm: Whether to use batch normalization layers
    """
    print(f"Initializing CNN model v2 for board game learning ({num_channels} channels, batchnorm={use_batchnorm})")

    input_shape = (tafl.DIM, tafl.DIM, num_channels)
    std = 0.1  # Small std to prevent score explosion through deep network

    inputs = Input(shape=input_shape)

    # Initial convolution block
    x = Conv2D(64, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First residual block
    res = x
    x = Conv2D(64, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Add()([x, res])
    x = Activation('relu')(x)

    # Pattern recognition block - different kernel sizes
    # 5x5 for broader patterns like surrounding threats
    branch1 = Conv2D(32, (5, 5), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    # 3x3 for local patterns
    branch2 = Conv2D(32, (3, 3), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)

    # 1x1 for point-wise patterns
    branch3 = Conv2D(32, (1, 1), padding='same', use_bias=not use_batchnorm,
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)

    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(0.2)(x)

    # Final convolution to reduce channels
    x = Conv2D(32, (3, 3), padding='same', use_bias=not use_batchnorm,
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    # Final layer with tanh to bound outputs between -1 and 1
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)

    model = Model(inputs=inputs, outputs=x)

    optimizer = Adam(learning_rate=0.001)  # Keep learning rate moderate
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.summary()
    return model


def initialize_cnn_model_claude(input_shape):
    std = 0.1
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    
    # Residual blocks
    for _ in range(4):
        res = x
        x = Conv2D(64, (3, 3), padding='same', activation='relu',
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        x = Conv2D(64, (3, 3), padding='same',
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
        x = Add()([x, res])
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
    
    # Pattern recognition block
    branch1 = Conv2D(32, (5, 5), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    branch3 = Conv2D(32, (1, 1), padding='same', activation='relu',
                     kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(0.2)(x)
    
    # Final convolution
    x = Conv2D(32, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Dense layers
    x = Flatten()(x)
    x = Dense(256, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', 
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Output layer
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    model = Model(inputs=inputs, outputs=x)
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


A_DIM = 0
D_DIM = 2
K_DIM = 1

# Global encoding setting (set by main())
USE_ENHANCED_ENCODING = False


def get_exploration_temp(num_games, max_temp=0.5, min_temp=0.02, decay=5000):
    """Calculate exploration temperature with exponential decay.

    Args:
        num_games: Number of games trained so far
        max_temp: Starting temperature (high exploration)
        min_temp: Minimum temperature (exploitation)
        decay: Decay constant (higher = slower decay)

    Returns:
        Temperature value for move selection noise
    """
    return max(min_temp, max_temp * math.exp(-num_games / decay))


def get_epsilon(num_games, max_eps=0.3, min_eps=0.01, decay=10000):
    """Calculate epsilon for epsilon-greedy exploration with exponential decay.

    Args:
        num_games: Number of games trained so far
        max_eps: Starting epsilon (high random move probability)
        min_eps: Minimum epsilon (low random move probability)
        decay: Decay constant (higher = slower decay)

    Returns:
        Probability of making a random move
    """
    return max(min_eps, max_eps * math.exp(-num_games / decay))


def game_state_to_3d_array(is_attacker_turn=True):
    """ 3D Numpy array representation of game state for ML model.
        2 spatial dimensions + channels for piece types and board features.

        Legacy 3-channel encoding:
        - Ch 0: Attackers
        - Ch 1: King
        - Ch 2: Defenders

        Enhanced 6-channel encoding:
        - Ch 0: Attackers
        - Ch 1: King
        - Ch 2: Defenders
        - Ch 3: Corners (escape squares)
        - Ch 4: Center (throne)
        - Ch 5: Turn indicator (1=attacker turn)
    """
    global USE_ENHANCED_ENCODING

    if tafl.Attackers is None or tafl.Defenders is None or tafl.Kings is None:
        print("Game not properly initialized.  Exiting.")
        sys.exit(1)

    num_channels = 6 if USE_ENHANCED_ENCODING else 3
    arr = np.zeros((tafl.DIM, tafl.DIM, num_channels), dtype=np.float32)

    # Basic piece channels
    for p in tafl.Attackers:
        arr[p.x_tile][p.y_tile][A_DIM] = 1
    for p in tafl.Kings:
        arr[p.x_tile][p.y_tile][K_DIM] = 1
    for p in tafl.Defenders:
        arr[p.x_tile][p.y_tile][D_DIM] = 1

    # Enhanced encoding: add board features
    if USE_ENHANCED_ENCODING:
        # Channel 3: Corners (escape squares for king)
        corners = [(0, 0), (0, tafl.DIM-1), (tafl.DIM-1, 0), (tafl.DIM-1, tafl.DIM-1)]
        for x, y in corners:
            arr[x][y][3] = 1

        # Channel 4: Center (throne - hostile to attackers)
        center = (tafl.DIM - 1) // 2
        arr[center][center][4] = 1

        # Channel 5: Turn indicator
        if is_attacker_turn:
            arr[:, :, 5] = 1

    return arr


def get_num_channels():
    """Get the number of channels based on current encoding setting."""
    global USE_ENHANCED_ENCODING
    return 6 if USE_ENHANCED_ENCODING else 3


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


def apply_random_symmetry(game_state):
    """Apply one random symmetry transform to a game state.

    Compatible with both legacy 3-channel and enhanced 6-channel encoding.
    Works because corners stay at corners and center stays at center
    under any rotation/mirror of the board.
    """
    k = random.randint(0, 3)  # 0, 1, 2, or 3 rotations of 90 degrees
    state = np.rot90(game_state, k)
    if random.random() < 0.5:  # 50% chance to mirror
        state = np.flip(state, axis=0)
    return state


def apply_random_symmetries_to_batch(game_states):
    """Apply independent random symmetry to each state in a batch."""
    return np.array([apply_random_symmetry(gs) for gs in game_states])


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
        Add intermediate rewards based on game state advantages and spatial understanding.
        Especially tuned for the simple game mode to understand positioning.
    """
    if dynamic: 
        decay_constant = float(len(corrected_scores)) / 2.  # 1/2 game length
    
    # Get final outcome for proper reward shaping
    final_outcome = corrected_scores[-1]
    final_magnitude = abs(final_outcome)  # Preserve the magnitude of win/loss
    
    # For simple game mode, add strong positional rewards
    if tafl.DIM == 5:  # Simple game mode
        king_pos = None
        attacker_pos = None
        for p in tafl.Kings:
            king_pos = (p.x_tile, p.y_tile)
        for p in tafl.Attackers:
            attacker_pos = (p.x_tile, p.y_tile)
            
        if king_pos and attacker_pos:
            # Calculate distance to nearest corner
            corners = [(0,0), (0,4), (4,0), (4,4)]
            min_corner_dist = min(abs(king_pos[0] - c[0]) + abs(king_pos[1] - c[1]) for c in corners)
            
            # Calculate if attacker is between king and nearest corner
            blocking_bonus = 0
            for corner in corners:
                if (min(king_pos[0], corner[0]) <= attacker_pos[0] <= max(king_pos[0], corner[0]) and
                    min(king_pos[1], corner[1]) <= attacker_pos[1] <= max(king_pos[1], corner[1])):
                    blocking_bonus = 0.3
                    break
    
    # Shape rewards to encourage progress toward goal
    for i in range(len(corrected_scores) - 1):
        # Blend between immediate state value and final outcome
        position = len(corrected_scores) - i - 1
        alpha = math.exp(-position / decay_constant)
        
        # Add positional rewards for simple game
        if tafl.DIM == 5:
            if final_outcome > 0:  # Attacker won
                corrected_scores[i] += blocking_bonus
            else:  # Defender won
                if min_corner_dist < 3:  # Reward being closer to corners
                    corrected_scores[i] += (3 - min_corner_dist) * 0.2
        
        # Add small positive reward for maintaining advantage, scaled by final magnitude
        if final_outcome > 0 and corrected_scores[i] > 0:
            advantage_reward = 0.1 * final_magnitude
        elif final_outcome < 0 and corrected_scores[i] < 0:
            advantage_reward = 0.1 * final_magnitude
        else:
            advantage_reward = 0
            
        # Scale intermediate rewards by final magnitude
        corrected_scores[i] = (corrected_scores[i] + alpha * final_outcome + advantage_reward) / (1. + alpha)


def compute_td_targets(game_states, final_reward, gamma=0.95, model=None):
    """Compute TD(0) targets with bootstrapped next-state values.

    Args:
        game_states: List of game state arrays from the game
        final_reward: The actual reward at game end (+1 win, -1 loss, -0.5 draw)
        gamma: Discount factor for future rewards
        model: The model to use for bootstrapping (if None, uses pure MC returns)

    Returns:
        numpy array of target values for each state
    """
    n = len(game_states)
    if n == 0:
        return np.array([])

    targets = np.zeros(n)
    targets[n-1] = final_reward  # Terminal state gets actual reward

    if model is not None and n >= 2:
        # Get value predictions for all states
        states_array = np.array(game_states)
        # Reshape appropriately for the model
        num_channels = states_array.shape[-1]
        states_reshaped = states_array.reshape(-1, tafl.DIM, tafl.DIM, num_channels)
        values = model.predict(states_reshaped, verbose=0).flatten()

        # TD(0): target[i] = gamma * (-V(s_{i+1}))
        # Negate because opponent's good position is bad for us
        for i in range(n-2, -1, -1):
            next_value = -values[i+1]  # Negate for opponent's perspective
            targets[i] = gamma * next_value
    else:
        # Fallback to Monte Carlo returns if no model
        for i in range(n-2, -1, -1):
            targets[i] = gamma * targets[i+1]

    return targets


def update_model(model, states, rewards, batch_size=32, use_td=False, gamma=0.95):
    """Train the model on a batch of state-reward pairs.

    Args:
        use_td: If True, use TD learning targets instead of smoothed rewards
        gamma: Discount factor for TD learning
    """
    if len(states) == 0:
        return

    # Convert to numpy arrays
    states = np.array(states)
    rewards = np.array(rewards)

    # Get number of channels from the states shape
    num_channels = states.shape[-1] if len(states.shape) == 4 else get_num_channels()

    # Get predictions before training for last few states
    num_debug_states = min(3, len(states))
    debug_states = states[-num_debug_states:]
    debug_states_reshaped = debug_states.reshape(-1, tafl.DIM, tafl.DIM, num_channels)
    before_preds = model.predict(debug_states_reshaped, verbose=0)

    # Use smaller batch size for draw games to prevent conflicting gradients
    actual_batch_size = 8 if any(abs(rewards + 0.5) < 0.01) else batch_size

    # Train model
    history = model.fit(
        states.reshape(-1, tafl.DIM, tafl.DIM, num_channels),
        rewards,
        batch_size=actual_batch_size,
        epochs=1,
        verbose=0
    )

    # Get predictions after training
    after_preds = model.predict(debug_states_reshaped, verbose=0)

    # Print debug info
    print("\nModel prediction changes after training:")
    print("State | Before  | After   | Target  | Change")
    print("-" * 45)
    for i in range(num_debug_states):
        target = rewards[-num_debug_states + i]
        before = before_preds[i][0]
        after = after_preds[i][0]
        change = after - before
        direction = "+" if (target > before and after > before) or (target < before and after < before) else "-"
        print(f"{i:5d} | {before:7.4f} | {after:7.4f} | {target:7.4f} | {change:+7.4f} {direction}")

    return history


def initialize_compact_model_simple(num_channels=3):
    """ Initialize an extremely compact model specifically for 5x5 simple game mode.
    The model focuses purely on spatial relationships between pieces, particularly
    the relative positions of attacker vs king and potential blocking positions.

    Args:
        num_channels: Number of input channels (3 for legacy, 6 for enhanced encoding)
    """
    print(f"Initializing minimal model for simple game mode ({num_channels} channels)")

    input_shape = (tafl.DIM, tafl.DIM, num_channels)
    std = 0.1
    
    inputs = Input(shape=input_shape)
    
    # Single conv layer to detect piece positions and basic spatial patterns
    # Using 8 filters to keep it very simple
    x = Conv2D(8, (3, 3), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(inputs)
    
    # Direct 1x1 convolution to focus on piece presence
    x = Conv2D(4, (1, 1), padding='same', activation='relu',
               kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Flatten and minimal dense layer
    x = Flatten()(x)
    x = Dense(16, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    # Output with tanh activation
    x = Dense(1, activation='tanh',
              kernel_initializer=TruncatedNormal(mean=0.0, stddev=std))(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    # Higher learning rate for faster adaptation
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.summary()
    return model


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
              help='Set to train using symmetrical board states (8x expansion)')
@click.option('--probabilistic-symmetry/--no-probabilistic-symmetry', default=False,
              help='Apply random symmetry transform to each state (alternative to --use-symmetry)')
@click.option('-al', '--attacker-load', default=0, help='Attacker model file num to load')
@click.option('-dl', '--defender-load', default=0, help='Defender model file num to load')
@click.option('-sl', '--stats-load', default=0, help='Stats model file num to load')
@click.option('-ll/-nl', '--load-latest/--not-latest', default=False, help='Set to search and use latest models/stats files')
@click.option('-v', '--version', default=7, help='Model version number')
# New training improvement options
@click.option('--use-td/--no-td', default=True, help='Use TD learning instead of smoothed rewards')
@click.option('--gamma', default=0.95, help='Discount factor for TD learning')
@click.option('--initial-temp', default=0.5, help='Initial exploration temperature')
@click.option('--final-temp', default=0.02, help='Final exploration temperature')
@click.option('--temp-decay', default=5000, help='Temperature decay constant (games)')
@click.option('--initial-epsilon', default=0.3, help='Initial epsilon for random moves')
@click.option('--final-epsilon', default=0.01, help='Final epsilon for random moves')
@click.option('--epsilon-decay', default=10000, help='Epsilon decay constant (games)')
@click.option('--benchmark/--no-benchmark', default=False, help='Output timing statistics')
@click.option('--legacy-encoding/--enhanced-encoding', default=True, help='Use legacy 3-channel encoding (default) or enhanced 6-channel')
@click.option('--batchnorm/--no-batchnorm', default=True, help='Use batch normalization in model (default: True)')
def main(game_name, human_attacker, human_defender, interactive, train_attacker, train_defender, dynamic_train,
         cache_model_every, exit_after_cache, use_symmetry, probabilistic_symmetry,
         attacker_load, defender_load, stats_load, load_latest, version,
         use_td, gamma, initial_temp, final_temp, temp_decay,
         initial_epsilon, final_epsilon, epsilon_decay, benchmark, legacy_encoding, batchnorm):
    """Main training loop."""

    global king_is_special
    global USE_ENHANCED_ENCODING
    king_is_special = False

    # Set encoding mode based on flag
    USE_ENHANCED_ENCODING = not legacy_encoding
    num_channels = 6 if USE_ENHANCED_ENCODING else 3
    if USE_ENHANCED_ENCODING:
        print(f"Using enhanced 6-channel encoding (corners, center, turn indicator)")
    else:
        print(f"Using legacy 3-channel encoding")

    # Warn if both symmetry options are enabled
    if use_symmetry and probabilistic_symmetry:
        print("Warning: Both --use-symmetry and --probabilistic-symmetry are enabled.")
        print("         Using --use-symmetry (8x expansion). Disable one for clarity.")
    elif probabilistic_symmetry:
        print("Using probabilistic symmetry augmentation (random transform per state)")
    elif use_symmetry:
        print("Using deterministic symmetry augmentation (8x expansion)")

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
            a_model_files = glob(save_dir + '/attacker_model_*_games.keras')
            if len(a_model_files) > 0:
                attacker_load = max([int(f.split("_")[4]) for f in a_model_files])  # Parse filenames and get latest
            else:
                attacker_load = 0
        if attacker_load == 0:
            if game_name.lower() == "simple":
                attacker_model = initialize_compact_model_simple(num_channels=num_channels)
            else:
                attacker_model = initialize_random_cnn_model_3d_sonnet(num_channels=num_channels, use_batchnorm=batchnorm)
        else:
            attacker_model = load_model('{}/attacker_model_{}_games.keras'.format(save_dir, attacker_load))
            # Recompile with fresh optimizer
            optimizer = Adam(learning_rate=0.001)
            attacker_model.compile(optimizer=optimizer, loss='mean_squared_error')
            num_train_games_attacker = attacker_load

    defender_model = None
    if not human_defender:
        if load_latest:
            d_model_files = glob(save_dir + '/defender_model_*_games.keras')
            if len(d_model_files) > 0:
                defender_load = max([int(f.split("_")[4]) for f in d_model_files])  # Parse filenames and get latest
            else:
                defender_load = 0
        if defender_load == -1:
            defender_model = None  # Defaults to mostly random + some extra King movements
        elif defender_load == 0:
            if game_name.lower() == "simple":
                defender_model = initialize_compact_model_simple(num_channels=num_channels)
            else:
                defender_model = initialize_random_cnn_model_3d_sonnet(num_channels=num_channels, use_batchnorm=batchnorm)
        else:
            defender_model = load_model('{}/defender_model_{}_games.keras'.format(save_dir, defender_load))
            # Recompile with fresh optimizer
            optimizer = Adam(learning_rate=0.001)
            defender_model.compile(optimizer=optimizer, loss='mean_squared_error')
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

        # Calculate exploration parameters with decay
        num_games = max(num_train_games_attacker, num_train_games_defender)
        attacker_temp = get_exploration_temp(num_games, initial_temp, final_temp, temp_decay)
        defender_temp = get_exploration_temp(num_games, initial_temp, final_temp, temp_decay)
        current_epsilon = get_epsilon(num_games, initial_epsilon, final_epsilon, epsilon_decay)

        if log_level > 0 and benchmark:
            print(f"Exploration: temp={attacker_temp:.4f}, epsilon={current_epsilon:.4f}")

        play, a_game_states, a_corrected_scores, d_game_states, d_corrected_scores = \
            run_game(attacker_model, defender_model, human_attacker, human_defender, screen, game_name,
                     sample_frac, attacker_temp, defender_temp, frac_attackers_to_remove, frac_defenders_to_remove,
                     epsilon=current_epsilon)

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
            if log_level>0:
                print("\nTraining attacker model:")
                print("""Last Attacker states: {}""".format(
                      ' '.join(['{:+0.4f}'.format(entry) for entry in a_corrected_scores[-15:]])))

            # Use TD learning or legacy smoothing
            if use_td:
                final_reward = a_corrected_scores[-1]
                a_targets = compute_td_targets(a_game_states, final_reward, gamma=gamma, model=attacker_model)
                if log_level>0:
                    print("""          TD targets: {}""".format(
                          ' '.join(['{:+0.4f}'.format(entry) for entry in a_targets[-15:]])))
            else:
                smooth_corrected_scores_exp(a_corrected_scores)
                a_targets = np.array(a_corrected_scores)
                if log_level>0:
                    print("""            Smoothed: {}""".format(
                          ' '.join(['{:+0.4f}'.format(entry) for entry in a_corrected_scores[-15:]])))

            # Convert to numpy arrays without shuffling
            a_game_states = np.array(a_game_states)

            # Symmetry augmentation (mutually exclusive options)
            if use_symmetry:
                a_game_states = expand_game_states_symmetries(a_game_states)
                a_targets = np.tile(a_targets, 8)
            elif probabilistic_symmetry:
                a_game_states = apply_random_symmetries_to_batch(a_game_states)
                # targets unchanged - same size batch

            update_model(attacker_model, a_game_states, a_targets, use_td=use_td, gamma=gamma)

        if train_defender and defender_model is not None and len(d_corrected_scores) > 0:
            if log_level>0:
                print("\nTraining defender model:")
                print("""Last Defender states: {}""".format(
                      ' '.join(['{:+0.4f}'.format(entry) for entry in d_corrected_scores[-15:]])))

            # Use TD learning or legacy smoothing
            if use_td:
                final_reward = d_corrected_scores[-1]
                d_targets = compute_td_targets(d_game_states, final_reward, gamma=gamma, model=defender_model)
                if log_level>0:
                    print("""          TD targets: {}""".format(
                          ' '.join(['{:+0.4f}'.format(entry) for entry in d_targets[-15:]])))
            else:
                smooth_corrected_scores_exp(d_corrected_scores)
                d_targets = np.array(d_corrected_scores)
                if log_level>0:
                    print("""            Smoothed: {}""".format(
                          ' '.join(['{:+0.4f}'.format(entry) for entry in d_corrected_scores[-15:]])))

            # Convert to numpy arrays without shuffling
            d_game_states = np.array(d_game_states)

            # Symmetry augmentation (mutually exclusive options)
            if use_symmetry:
                d_game_states = expand_game_states_symmetries(d_game_states)
                d_targets = np.tile(d_targets, 8)
            elif probabilistic_symmetry:
                d_game_states = apply_random_symmetries_to_batch(d_game_states)
                # targets unchanged - same size batch

            update_model(defender_model, d_game_states, d_targets, use_td=use_td, gamma=gamma)

        if (stats.num_games_total() % cache_model_every == 0):  # Save every cache_model_every games
            # print('--- num games played: {}'.format(stats.num_games_total()))
            if num_train_games_attacker > 0: attacker_model.save(
                '{}/attacker_model_{}_games.keras'.format(save_dir, num_train_games_attacker))
            if num_train_games_defender > 0: defender_model.save(
                '{}/defender_model_{}_games.keras'.format(save_dir, num_train_games_defender))
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
