"""
An extension of Hnefatafl to include AI training, AI vs AI, and Player vs AI (attacker or defender) modes.

A full description of the game can be found here: https://en.wikipedia.org/wiki/Tafl_games

Learning Rate Features:
- Use --initial-lr to set the starting learning rate (default 0.001)
- Use --lr-decay to enable automatic learning rate decay during training
  (decays based on number of games the model has been trained on)

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
from models import simple_model, sonnet_model, claude_model


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


def check_for_winning_move(move, is_attacker=True):
    """Check if there's an immediate winning move available.
    
    Args:
        move: The Move object
        is_attacker: True if checking for attacker (king kill), False if defender (king escape)
        
    Returns:
        tuple of (piece, move_coords) if winning move exists, None otherwise
    """
    if is_attacker:
        # For attacker: Check for moves that kill the king
        for piece in tafl.Attackers:
            move.select(piece)
            tafl.Current.add(piece)
            if len(move.vm) > 0:
                for m_coords in move.vm:
                    # Try candidate move
                    old_x, old_y = piece.x_tile, piece.y_tile
                    move.is_valid_move(m_coords, piece, True)
                    
                    # Check if this move would kill the king
                    move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                    
                    if move.king_killed:
                        # Found a winning move! Undo and return the move
                        move.undo(piece)
                        move.select(piece)
                        tafl.Current.empty()
                        return (piece, m_coords)
                    
                    # Undo trial move
                    move.undo(piece)
            
            # Clean up after checking this piece
            move.select(piece)
            tafl.Current.empty()
    else:
        # For defender: Check for king escape moves
        for king in tafl.Kings:
            move.select(king)
            tafl.Current.add(king)
            if len(move.vm) > 0:
                for m_coords in move.vm:
                    # Check if this is an escape square
                    if m_coords in tafl.SPECIALSQS.difference([((tafl.DIM - 1) // 2, (tafl.DIM - 1) // 2)]):
                        move.select(king)
                        tafl.Current.empty()
                        return (king, m_coords)
            
            # Clean up
            move.select(king)
            tafl.Current.empty()
                
    return None


def run_game(attacker_model=None, defender_model=None, human_attacker=False, human_defender=False, screen=None,
             game_name='Hnefatafl', sample_frac=1.0, attacker_temp=0.1, defender_temp=0.1, 
             frac_attackers_to_remove=0, frac_defenders_to_remove=0, epsilon=0.1, log_level=0):
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
           epsilon: Probability of taking a random move (epsilon-greedy exploration)
           log_level: Controls verbosity of output (0=minimal, 1=normal, 2=debug)
    """
    # Define scaled reward values - using 0.8 instead of 1.0 to avoid extreme targets
    WIN_REWARD = 0.8
    LOSS_REWARD = -0.8
    DRAW_REWARD = -0.4  # Slight penalty for draws

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
                    
        # Check for no-pieces condition before trying to make a move
        if len(tafl.Attackers.sprites()) == 0:
            text = "--- All attackers eliminated! Defenders win!"
            if log_level > 0:
                print(text)
            text2 = "Play again? y/n"
            
            # Replace last predictions with win/loss values
            if len(a_predicted_scores) > 0:
                a_predicted_scores[-1] = LOSS_REWARD
            if len(d_predicted_scores) > 0:
                d_predicted_scores[-1] = WIN_REWARD
                
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
            
        if len(tafl.Defenders.sprites()) == 0:
            text = "--- All defenders eliminated! Attackers win!"
            if log_level > 0:
                print(text)
            text2 = "Play again? y/n"
            
            # Replace last predictions with win/loss values
            if len(a_predicted_scores) > 0:
                a_predicted_scores[-1] = WIN_REWARD
            if len(d_predicted_scores) > 0:
                d_predicted_scores[-1] = LOSS_REWARD
                
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
        
        # Check if the king is still alive, if not, attackers win
        if len(tafl.Kings.sprites()) == 0:
            text = "--- King killed! Attackers win!"
            if log_level > 0:
                print(text)
            text2 = "Play again? y/n"
            
            # Replace last predictions with win/loss values
            if len(a_predicted_scores) > 0:
                a_predicted_scores[-1] = WIN_REWARD
            if len(d_predicted_scores) > 0:
                d_predicted_scores[-1] = LOSS_REWARD
                
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
            
        if num_moves >= 100 or (game_name.lower() == "simple" and num_moves > 4):
            if log_level > 0:
                print("--- Draw game after {} moves".format(num_moves))
            # Replace last predictions with draw values instead of appending
            if len(a_predicted_scores) > 0:
                a_predicted_scores[-1] = DRAW_REWARD  # Slight penalty for draws
            if len(d_predicted_scores) > 0:
                d_predicted_scores[-1] = DRAW_REWARD  # Slight penalty for draws
            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores

        if move.a_turn:
            if game_name.lower() == "simple":
                move.a_turn = False
                continue
                # print("Attacker's Turn: Move {}".format(num_moves))
            if human_attacker:
                play = do_human_turn(screen, board, move)
            elif attacker_model is None:
                do_random_move(move)
                game_state = game_state_to_3d_array()
                predicted_score = (random.random() - 0.5) * 2
                a_game_states.append(game_state)
                a_predicted_scores.append(predicted_score)
            else:
                if human_defender: time.sleep(0.5)
                
                # First check for winning moves (priority over both best move and random move)
                winning_move = check_for_winning_move(move, is_attacker=True)
                if winning_move:
                    if log_level > 0:
                        print(f"--- Attacker found immediate winning move (king kill) ---")
                    piece, m_coords = winning_move
                    move.select(piece)
                    tafl.Current.add(piece)
                    
                    # Execute the winning move
                    if move.is_valid_move(m_coords, piece, True):
                        move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                        game_state = game_state_to_3d_array()
                        predicted_score = WIN_REWARD  # Use scaled win reward instead of 1.0
                        
                        move.end_turn(piece)
                        tafl.Current.empty()
                        
                        # Record state and predicted score
                        a_game_states.append(game_state)
                        a_predicted_scores.append(predicted_score)
                        
                        # Check if this winning move killed the king - if so, end game immediately
                        if move.king_killed:
                            text = "--- King killed! Attackers win!"
                            if log_level > 0:
                                print(text)
                            text2 = "Play again? y/n"
                            
                            # Replace last predictions with win/loss values
                            a_predicted_scores[-1] = WIN_REWARD
                            if len(d_predicted_scores) > 0:
                                d_predicted_scores[-1] = LOSS_REWARD
                                
                            if screen:
                                tafl.update_image(screen, board, move, text, text2)
                                pygame.display.flip()
                            if human_attacker or human_defender: play = end_game_loop(move)
                            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
                        
                        continue
                
                # Epsilon-greedy: randomly choose between exploration (random move) and exploitation (best move)
                if random.random() < epsilon:
                    # Exploration: Take a random move
                    if log_level > 1:
                        print(f"--- Epsilon-Greedy ({epsilon:.2f}): Taking random move for Attacker ---")
                    do_random_move(move)
                    game_state = game_state_to_3d_array()
                    # Still predict the score for the resulting state for training consistency
                    predicted_score = attacker_model.predict(game_state.reshape(1, tafl.DIM, tafl.DIM, 3), verbose=0)[0][0]
                else:
                    # Exploitation: Use the model to find the best move
                    if log_level > 1:
                        print(f"--- Epsilon-Greedy: Using best move for Attacker ---")
                    game_state, predicted_score = do_best_move(move,
                                                               attacker_model,
                                                               game_state_cache,
                                                               sample_frac=sample_frac,
                                                               temperature=attacker_temp,
                                                               enable_remove=True if game_name.lower() != 'simple' else False,
                                                               screen=screen,
                                                               board=board)
                
                # Record state and predicted score regardless of how the move was chosen
                a_game_states.append(game_state)
                a_predicted_scores.append(predicted_score)
            
            # Check for game-ending conditions after attacker's move
            if move.king_killed:
                text = "--- King killed! Attackers win!"
                if log_level > 0:
                    print(text)
                text2 = "Play again? y/n"
                
                # Replace last predictions with win/loss values
                if len(a_predicted_scores) > 0:
                    a_predicted_scores[-1] = WIN_REWARD 
                if len(d_predicted_scores) > 0:
                    d_predicted_scores[-1] = LOSS_REWARD
                    
                if screen:
                    tafl.update_image(screen, board, move, text, text2)
                    pygame.display.flip()
                if human_attacker or human_defender: play = end_game_loop(move)
                return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
            
        else:
            # print("Defender's Turn: Move {}".format(num_moves))
            if human_defender:
                play = do_human_turn(screen, board, move)
            elif defender_model is None:
                # game_state = do_mostly_random_but_strike_to_kill_move(move)
                do_dummy_1_defender_move(move)
                game_state = game_state_to_3d_array()
                predicted_score = (random.random() - 0.5) * 2
                d_game_states.append(game_state)
                d_predicted_scores.append(predicted_score)
            else:
                if human_attacker: time.sleep(0.5)
                
                # First check for winning moves (priority over both best move and random move)
                winning_move = check_for_winning_move(move, is_attacker=False)
                if winning_move:
                    if log_level > 0:
                        print(f"--- Defender found immediate winning move (king escape) ---")
                    piece, m_coords = winning_move
                    move.select(piece)
                    tafl.Current.add(piece)
                    
                    # Execute the winning move
                    if move.is_valid_move(m_coords, piece, True):
                        move.king_escaped(tafl.Kings)
                        move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)
                        game_state = game_state_to_3d_array()
                        predicted_score = WIN_REWARD  # Use scaled win reward instead of 1.0
                        
                        move.end_turn(piece)
                        tafl.Current.empty()
                        
                        # Record state and predicted score
                        d_game_states.append(game_state)
                        d_predicted_scores.append(predicted_score)
                        
                        # Check if this winning move caused escape - if so, end game immediately
                        if move.escaped:
                            text = "--- King escaped! Defenders win!"
                            if log_level > 0:
                                print(text)
                            text2 = "Play again? y/n"
                            
                            # Replace last predictions with win/loss values
                            if len(a_predicted_scores) > 0:
                                a_predicted_scores[-1] = LOSS_REWARD
                            d_predicted_scores[-1] = WIN_REWARD
                                
                            if screen:
                                tafl.update_image(screen, board, move, text, text2)
                                pygame.display.flip()
                            if human_attacker or human_defender: play = end_game_loop(move)
                            return play, a_game_states, a_predicted_scores, d_game_states, d_predicted_scores
                        
                        continue
                
                # Epsilon-greedy: randomly choose between exploration (random move) and exploitation (best move)
                if random.random() < epsilon:
                    # Exploration: Take a random move
                    if log_level > 1:
                        print(f"--- Epsilon-Greedy ({epsilon:.2f}): Taking random move for Defender ---")
                    do_random_move(move)
                    game_state = game_state_to_3d_array()
                    # Still predict the score for the resulting state for training consistency
                    predicted_score = defender_model.predict(game_state.reshape(1, tafl.DIM, tafl.DIM, 3), verbose=0)[0][0]
                else:
                    # Exploitation: Use the model to find the best move
                    if log_level > 1:
                        print(f"--- Epsilon-Greedy: Using best move for Defender ---")
                    game_state, predicted_score = do_best_move(move,
                                                               defender_model,
                                                               game_state_cache,
                                                               sample_frac=sample_frac,
                                                               temperature=defender_temp,
                                                               enable_remove=True if game_name.lower() != 'simple' else False,
                                                               screen=screen,
                                                               board=board)
                
                # Record state and predicted score regardless of how the move was chosen
                d_game_states.append(game_state)
                d_predicted_scores.append(predicted_score)
            
            # Check for game-ending conditions after defender's move
            if move.escaped:
                text = "--- King escaped! Defenders win!"
                if log_level > 0:
                    print(text)
                text2 = "Play again? y/n"
                
                # Replace last predictions with win/loss values
                if len(a_predicted_scores) > 0:
                    a_predicted_scores[-1] = LOSS_REWARD
                if len(d_predicted_scores) > 0:
                    d_predicted_scores[-1] = WIN_REWARD
                    
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


def do_best_move(move, model, game_state_cache, sample_frac=1.0, screen=None, board=None, enable_remove=True, temperature=0.1):
    """ Function to try all possible moves and select the best according to the model provided.
        Uses batch prediction for better performance.
    """
    # Store the current game state before any moves
    current_game_state = game_state_to_3d_array()
    game_state_cache.append(deepcopy(current_game_state))

    # For simple game, we want to be more focused in our exploration
    simple_game = (tafl.DIM == 5)
    if simple_game:
        sample_frac = 1.0  # Always consider all moves in simple game
        
    if move.a_turn:
        pieces = tafl.Attackers
    else:
        pieces = tafl.Defenders
        # If King can win, do it immediately (no need to predict)
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
                    return game_state_to_3d_array(), 1.0
            move.select(king)
            tafl.Current.empty()

    if len(pieces) == 0:
        return current_game_state, 0.0
    
    # Create containers to collect all valid moves and their resulting states
    all_pieces = []
    all_moves = []
    all_game_states = []
    
    # Collect all valid moves and their resulting states
    for piece in pieces:
        if random.random() > sample_frac:
            continue
            
        move.select(piece)
        tafl.Current.add(piece)
        
        if len(move.vm) == 0:
            move.select(piece)
            tafl.Current.empty()
            continue
            
        for m in move.vm:
            if random.random() > sample_frac:
                continue
                
            # Try candidate move
            move.is_valid_move(m, tafl.Current.sprites()[0], True)
            
            # Check for immediate win for attacker
            if move.a_turn and move.king_killed:  
                # Move would kill king, do it without further evaluation
                if enable_remove:
                    move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                
                game_state = game_state_to_3d_array()
                
                # Display final selected move if in interactive mode
                if screen and board:
                    tafl.update_image(screen, board, move, 
                                    f"Selected move ({piece.x_tile}, {piece.y_tile})->({m[0]}, {m[1]})",
                                    f"Score: 1.0",
                                    highlight_pos=m,
                                    highlight_score=1.0)
                    pygame.display.flip()
                    time.sleep(0.8)
                
                move.end_turn(tafl.Current.sprites()[0])
                tafl.Current.empty()
                return game_state, 1.0
                
            if enable_remove:
                if move.a_turn:
                    move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
                else:
                    move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)
                    
            # Get state after move
            game_state = game_state_to_3d_array()
            
            # Store the move information
            all_pieces.append(piece)
            all_moves.append(m)
            all_game_states.append(game_state)
            
            # Undo the move
            move.undo(tafl.Current.sprites()[0])
            
        # Clean up after checking this piece
        move.select(piece)
        tafl.Current.empty()
    
    # If no valid moves were found
    if len(all_game_states) == 0:
        print("NO VALID MOVES! No moves at all or a Draw?")
        return current_game_state, 0.0
    
    # Convert collected states to numpy array for batch prediction
    batch_states = np.array(all_game_states)
    
    # Get predictions for all states in a single batch
    batch_predictions = model.predict(
        batch_states.reshape(-1, tafl.DIM, tafl.DIM, 3), 
        verbose=0
    ).flatten()
    
    # Add temperature-scaled random noise to predictions
    if temperature > 0:
        noise = np.random.normal(0, temperature, size=batch_predictions.shape)
        batch_predictions += noise
    
    # Break ties with small random noise
    batch_predictions += np.random.uniform(-0.01, 0.01, size=batch_predictions.shape)
    
    # Check for state repetition
    repetition_penalty = np.zeros_like(batch_predictions)
    for i, state in enumerate(all_game_states):
        # Check if this state would repeat a previous state
        if next((True for elem in itertools.islice(game_state_cache, 0, game_state_cache.maxlen) 
                 if np.array_equal(elem, state)), False):
            repetition_penalty[i] = -1.0  # Apply penalty for repeating a state
    
    # Apply repetition penalty
    final_scores = batch_predictions + repetition_penalty
    
    # Find the move with the highest score
    best_index = np.argmax(final_scores)
    best_score = float(batch_predictions[best_index])  # Use the original prediction as the score
    best_piece = all_pieces[best_index]
    best_move = all_moves[best_index]
    
    # Now execute the best move
    move.select(best_piece)
    tafl.Current.add(best_piece)
    
    if move.is_valid_move(best_move, tafl.Current.sprites()[0], True):
        if tafl.Current.sprites()[0] in tafl.Kings:
            move.king_escaped(tafl.Kings)
        if enable_remove:
            if move.a_turn:
                move.remove_pieces(tafl.Defenders, tafl.Attackers, tafl.Kings, king_is_special)
            else:
                move.remove_pieces(tafl.Attackers, tafl.Defenders, tafl.Kings, king_is_special)

        final_game_state = game_state_to_3d_array()

        # Display final selected move if in interactive mode
        if screen and board:
            tafl.update_image(screen, board, move, 
                            f"Selected move ({best_piece.x_tile}, {best_piece.y_tile})->({best_move[0]}, {best_move[1]})",
                            f"Score: {best_score:.2f}",
                            highlight_pos=best_move,
                            highlight_score=best_score)
            pygame.display.flip()
            time.sleep(0.8)

        move.end_turn(tafl.Current.sprites()[0])
        tafl.Current.empty()
        
        return final_game_state, best_score
    else:
        print("ERROR: Best move logic failed... Fix! Debugging info:")
        print(f"Best move: {best_move}, Best piece: ({best_piece.x_tile}, {best_piece.y_tile})")
        print(f"Current: {tafl.Current.sprites()[0]}, row: {move.row}, col: {move.col}")
        print(f"Valid moves: {move.vm}")
        
        # Clean up
        move.select(best_piece)
        tafl.Current.empty()
        
        return current_game_state, 0.0


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
        arr[p.x_tile][p.y_tile][A_DIM] = 1
    for p in tafl.Kings:
        arr[p.x_tile][p.y_tile][K_DIM] = 1 
    for p in tafl.Defenders:
        arr[p.x_tile][p.y_tile][D_DIM] = 1 

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


def update_model(model, states, rewards, batch_size=32, learning_rate=None, log_level=0):
    """Train the model on a batch of state-reward pairs."""
    if len(states) == 0:
        return
        
    # Update learning rate if provided
    if learning_rate is not None:
        current_lr = tf.keras.backend.get_value(model.optimizer.learning_rate)
        if current_lr != learning_rate:
            if log_level > 0:
                print(f"Updating learning rate: {current_lr:.6f} -> {learning_rate:.6f}")
            tf.keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    
    # Convert to numpy arrays
    states = np.array(states)
    rewards = np.array(rewards)
    
    # Only calculate debug predictions if we're going to print them
    if log_level > 1:
        # Get predictions before training for last few states
        num_debug_states = min(3, len(states))
        debug_states = states[-num_debug_states:]
        debug_states_reshaped = debug_states.reshape(-1, tafl.DIM, tafl.DIM, 3)
        before_preds = model.predict(debug_states_reshaped, verbose=0)
    
    # Use smaller batch size for draw games to prevent conflicting gradients
    actual_batch_size = 8 if any(abs(rewards + 0.5) < 0.01) else batch_size
    
    # Train model
    history = model.fit(
        states.reshape(-1, tafl.DIM, tafl.DIM, 3), 
        rewards,
        batch_size=actual_batch_size,
        epochs=1,
        verbose=0
    )
    
    # Only calculate and print debug information if log_level > 1
    if log_level > 1:
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
            direction = "✓" if (target > before and after > before) or (target < before and after < before) else "✗"
            print(f"{i:5d} | {before:7.4f} | {after:7.4f} | {target:7.4f} | {change:+7.4f} {direction}")
    
    return history


@click.command()
@click.option('-g', '--game-name', default='Hnefatafl', help='Name of Tafl variant to play')
@click.option('-ha/-aa', '--human-attacker/--ai-attacker', default=False, help='Set to play attacker manually')
@click.option('-hd/-ad', '--human-defender/--ai-defender', default=False, help='Set to play defender manually')
@click.option('-i/-b', '--interactive/--batch', default=False, help='Set to watch AI vs AI matches')
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
@click.option('--initial-lr', default=0.001, help='Initial learning rate for model training')
@click.option('-ld/-nld', '--lr-decay/--no-lr-decay', default=False, help='Enable learning rate decay during training')
@click.option('--epsilon', default=0.1, type=float, help='Probability of taking a random move (epsilon-greedy, 0.0 to 1.0)')
@click.option('--log-level', default=0, type=int, help='Log verbosity: 0=minimal, 1=normal, 2=debug')
def main(game_name, human_attacker, human_defender, interactive, train_attacker, train_defender, dynamic_train,
         cache_model_every, exit_after_cache, use_symmetry,
         attacker_load, defender_load, stats_load, load_latest, version,
         initial_lr, lr_decay, epsilon, log_level):
    """Main training loop."""

    global king_is_special
    king_is_special = False

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
                attacker_model = simple_model(tafl.DIM, learning_rate=initial_lr)
            else:
                attacker_model = sonnet_model(tafl.DIM, learning_rate=initial_lr)
        else:
            attacker_model = load_model('{}/attacker_model_{}_games.keras'.format(save_dir, attacker_load))
            # Recompile with fresh optimizer
            optimizer = Adam(learning_rate=initial_lr)
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
                defender_model = simple_model(tafl.DIM, learning_rate=initial_lr)
            else:
                defender_model = sonnet_model(tafl.DIM, learning_rate=initial_lr)
        else:
            defender_model = load_model('{}/defender_model_{}_games.keras'.format(save_dir, defender_load))
            # Recompile with fresh optimizer
            optimizer = Adam(learning_rate=initial_lr)
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

        attacker_temp = 0.1
        defender_temp = 0.1
        play, a_game_states, a_corrected_scores, d_game_states, d_corrected_scores = \
            run_game(attacker_model, defender_model, human_attacker, human_defender, screen, game_name,
                     sample_frac, attacker_temp, defender_temp, frac_attackers_to_remove, frac_defenders_to_remove, 
                     epsilon=epsilon, log_level=log_level)

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
            smooth_corrected_scores_exp(a_corrected_scores)
            if log_level>0:
                print("""            Smoothed: {}""".format(
                      ' '.join(['{:+0.4f}'.format(entry) for entry in a_corrected_scores[-15:]])))
            # Convert to numpy arrays without shuffling
            a_game_states = np.array(a_game_states)
            a_corrected_scores = np.array(a_corrected_scores)
            if use_symmetry:
                a_game_states = expand_game_states_symmetries(a_game_states)
                a_corrected_scores = np.tile(a_corrected_scores, 8)
            
            # Calculate learning rate with decay if enabled
            current_lr = initial_lr
            if lr_decay:
                decay_factor = 0.1
                current_lr = initial_lr * (1.0 / (1.0 + decay_factor * num_train_games_attacker / 1000))
            
            update_model(attacker_model, a_game_states, a_corrected_scores, learning_rate=current_lr, log_level=log_level)

        if train_defender and defender_model is not None and len(d_corrected_scores) > 0:
            if log_level>0:
                print("\nTraining defender model:")
                print("""Last Defender states: {}""".format(
                      ' '.join(['{:+0.4f}'.format(entry) for entry in d_corrected_scores[-15:]])))
            smooth_corrected_scores_exp(d_corrected_scores)
            if log_level>0:
                print("""            Smoothed: {}""".format(
                ' '.join(['{:+0.4f}'.format(entry) for entry in d_corrected_scores[-15:]])))
            # Convert to numpy arrays without shuffling
            d_game_states = np.array(d_game_states)
            d_corrected_scores = np.array(d_corrected_scores)
            if use_symmetry:
                d_game_states = expand_game_states_symmetries(d_game_states)
                d_corrected_scores = np.tile(d_corrected_scores, 8)
            
            # Calculate learning rate with decay if enabled
            current_lr = initial_lr
            if lr_decay:
                decay_factor = 0.1
                current_lr = initial_lr * (1.0 / (1.0 + decay_factor * num_train_games_defender / 1000))
            
            update_model(defender_model, d_game_states, d_corrected_scores, learning_rate=current_lr, log_level=log_level)

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
