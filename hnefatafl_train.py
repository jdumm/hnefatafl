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
from keras.optimizers import Adam
from keras.models import load_model

import hnefatafl as tafl
from models import (
    initialize_model_for_game,
    resolve_model_preset,
    validate_model_channels,
)
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
                            if move.is_valid_move(m, tafl.Current.sprites()[0], True):
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


def run_game(model=None, human_attacker=False, human_defender=False, screen=None,
             game_name='Hnefatafl', sample_frac=1.0, attacker_temp=0.1, defender_temp=0.1,
             frac_attackers_to_remove=0, frac_defenders_to_remove=0, epsilon=0.0, log_level=1):
    """Run one game and return states plus terminal reward from attacker perspective."""
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
    game_states = []
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
            if log_level >= 2:
                print("--- Draw game after {} moves".format(num_moves))
            return play, game_states, -0.5, num_moves

        if move.a_turn:
            if game_name.lower() == "simple":
                move.a_turn = False
                continue
                # print("Attacker's Turn: Move {}".format(num_moves))
            if human_attacker:
                play = do_human_turn(screen, board, move)
            elif model is None:
                do_random_move(move)
            else:
                if human_defender: time.sleep(0.5)
                game_state, _ = do_best_move(move,
                                             model,
                                             game_state_cache,
                                             sample_frac=sample_frac,
                                             temperature=attacker_temp,
                                             enable_remove=True if game_name.lower() != 'simple' else False,
                                             screen=screen,
                                             board=board,
                                             epsilon=epsilon,
                                             prefer_max=True)
                game_states.append(game_state)
        else:
            # print("Defender's Turn: Move {}".format(num_moves))
            if human_defender:
                play = do_human_turn(screen, board, move)
            elif model is None:
                do_dummy_1_defender_move(move)
            else:
                if human_attacker: time.sleep(0.5)
                game_state, _ = do_best_move(move,
                                             model,
                                             game_state_cache,
                                             sample_frac=sample_frac,
                                             temperature=defender_temp,
                                             enable_remove=True if game_name.lower() != 'simple' else False,
                                             screen=screen,
                                             board=board,
                                             epsilon=epsilon,
                                             prefer_max=False)
                game_states.append(game_state)

        if move.escaped:
            text = "--- King escaped! Defenders win!"
            if log_level >= 2:
                print(text)
            text2 = "Play again? y/n"
            # Scale reward based on number of moves - faster wins get higher rewards
            
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, game_states, -1.0, num_moves
        if move.king_killed:
            text = "--- King killed! Attackers win!"
            if log_level >= 2:
                print(text)
            text2 = "Play again? y/n"
            if screen:
                tafl.update_image(screen, board, move, text, text2)
                pygame.display.flip()
            if human_attacker or human_defender: play = end_game_loop(move)
            return play, game_states, +1.0, num_moves
        if move.restart:
            return play, game_states, 0.0, num_moves


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


def do_best_move(move, model, game_state_cache, sample_frac=1.0, screen=None, board=None, enable_remove=True,
                 temperature=0.1, epsilon=0.0, prefer_max=True):
    """ Function to try all possible moves and select the best according to the model provided.

    Uses batch inference for efficiency: collects all candidate states first, then
    makes a single batched prediction call instead of one call per move.

    Args:
        epsilon: Probability of making a random move (exploration)
    """
    game_state = game_state_to_3d_array(is_attacker_turn=move.a_turn)
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
                    return game_state, 1.0 if prefer_max else -1.0
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
                game_state = game_state_to_3d_array(is_attacker_turn=move.a_turn)
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

            candidate_state = game_state_to_3d_array(is_attacker_turn=move.a_turn)

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
    best_idx = np.argmax(scores) if prefer_max else np.argmin(scores)
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

        game_state = game_state_to_3d_array(is_attacker_turn=move.a_turn)

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
        print("reValid moves", move.valid_moves(best_piece.special_sqs))
        time.sleep(30)
        sys.exit(1)


A_DIM = 0
D_DIM = 2
K_DIM = 1


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

        8-channel encoding:
        - Ch 0: Attackers
        - Ch 1: King
        - Ch 2: Defenders
        - Ch 3: Corners (escape squares)
        - Ch 4: Center (throne)
        - Ch 5: Turn indicator (1=attacker turn)
        - Ch 6: Attacker piece count normalized by board area
        - Ch 7: Defender piece count normalized by board area
    """
    if tafl.Attackers is None or tafl.Defenders is None or tafl.Kings is None:
        print("Game not properly initialized.  Exiting.")
        sys.exit(1)

    num_channels = 8
    arr = np.zeros((tafl.DIM, tafl.DIM, num_channels), dtype=np.float32)

    # Basic piece channels
    for p in tafl.Attackers:
        arr[p.x_tile][p.y_tile][A_DIM] = 1
    for p in tafl.Kings:
        arr[p.x_tile][p.y_tile][K_DIM] = 1
    for p in tafl.Defenders:
        arr[p.x_tile][p.y_tile][D_DIM] = 1

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

    # Channels 6-7: global team piece counts (broadcast)
    board_area = float(tafl.DIM * tafl.DIM)
    arr[:, :, 6] = len(tafl.Attackers) / board_area
    arr[:, :, 7] = len(tafl.Defenders) / board_area

    return arr


def get_num_channels():
    """Return the number of channels for game-state encoding."""
    return 8


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

    Works with the fixed 8-channel encoding because corners stay at corners,
    center stays at center, and global count/turn channels are spatially
    uniform under symmetry.
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


def compute_td_targets(game_states, final_reward, gamma=0.95, model=None, negate_next_value=True):
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

        # TD(0): target[i] = gamma * V(s_{i+1}) for attacker-perspective value.
        # For side-to-move value functions, set negate_next_value=True.
        for i in range(n-2, -1, -1):
            next_value = -values[i+1] if negate_next_value else values[i+1]
            targets[i] = gamma * next_value
    else:
        # Fallback to Monte Carlo returns if no model
        for i in range(n-2, -1, -1):
            targets[i] = gamma * targets[i+1]

    return targets


def update_model(model, states, rewards, batch_size=32, use_td=False, gamma=0.95, log_level=1):
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

    if log_level >= 2:
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

    if log_level >= 2:
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


@click.command()
@click.option('-g', '--game-name', default='Hnefatafl', help='Name of Tafl variant to play')
@click.option('-ha/-aa', '--human-attacker/--ai-attacker', default=False, help='Set to play attacker manually')
@click.option('-hd/-ad', '--human-defender/--ai-defender', default=False, help='Set to play defender manually')
@click.option('-i/-b', '--interactive/--batch', default=False, help='Set true in order to watch AI vs AI matches')
@click.option('-ta/-na', '--train-attacker/--no-train-attacker', default=False,
              help='Set to update attacker AI after each game')
@click.option('-td/-nd', '--train-defender/--no-train-defender', default=False,
              help='Set to update defender AI after each game')
@click.option('--train-model/--no-train-model', default=False,
              help='Train the shared model after each game')
@click.option('-dt/-st', '--dynamic-train/--static-train', default=False, help='Set to pause defender AI when lopsided')
@click.option('-c', '--cache-model-every', default=100, help='Cache the Keras DNN model every so many games')
@click.option('-e/-ne', '--exit-after-cache/--no-exit-after-cache', default=False, help='Exit after model cache step to allow restart')
@click.option('-s/-ns', '--use-symmetry/--no-symmetry', default=False,
              help='Set to train using symmetrical board states (8x expansion)')
@click.option('--probabilistic-symmetry/--no-probabilistic-symmetry', default=False,
              help='Apply random symmetry transform to each state (alternative to --use-symmetry)')
@click.option('-al', '--attacker-load', default=0, help='Attacker model file num to load')
@click.option('-dl', '--defender-load', default=0, help='Defender model file num to load')
@click.option('-ml', '--model-load', default=0, help='Shared model file num to load')
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
@click.option('--batchnorm/--no-batchnorm', default=True, help='Use batch normalization in model (default: True)')
@click.option('--learning-rate', default=0.001, help='Learning rate for optimizer (try 0.0001 if models saturate)')
@click.option('--model-preset', type=click.Choice(['auto', 'simple', 'brandubh', 'hnefatafl'], case_sensitive=False),
              default='auto', help='Model preset override. Defaults to auto selection from game mode.')
@click.option('--log-level', default=1, type=click.IntRange(0, 2),
              help='Logging verbosity: 0=minimal, 1=periodic summaries, 2=per-game debug')
@click.option('--log-every', default=100, type=click.IntRange(1, 1000000),
              help='At log-level 1, print summary every N games')
def main(game_name, human_attacker, human_defender, interactive, train_attacker, train_defender, train_model, dynamic_train,
         cache_model_every, exit_after_cache, use_symmetry, probabilistic_symmetry,
         attacker_load, defender_load, model_load, stats_load, load_latest, version,
         use_td, gamma, initial_temp, final_temp, temp_decay,
         initial_epsilon, final_epsilon, epsilon_decay, benchmark, batchnorm, learning_rate, model_preset,
         log_level, log_every):
    """Main training loop."""

    global king_is_special
    king_is_special = False

    # Fixed encoding mode
    num_channels = 8
    print("Using fixed 8-channel encoding (corners, center, turn indicator, attacker count, defender count)")
    effective_preset = resolve_model_preset(game_name, model_preset=model_preset)
    print(f"Using model preset '{effective_preset}' (requested: {model_preset}, game: {game_name.lower()})")

    # Warn if both symmetry options are enabled
    if use_symmetry and probabilistic_symmetry:
        print("Warning: Both --use-symmetry and --probabilistic-symmetry are enabled.")
        print("         Using --use-symmetry (8x expansion). Disable one for clarity.")
    elif probabilistic_symmetry:
        print("Using probabilistic symmetry augmentation (random transform per state)")
    elif use_symmetry:
        print("Using deterministic symmetry augmentation (8x expansion)")

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

    # Backward-compatible aliases: legacy split train flags now map to shared train flag.
    if train_attacker or train_defender:
        if not train_model:
            print("Info: --train-attacker/--train-defender are deprecated. Using shared --train-model.")
        train_model = train_model or train_attacker or train_defender

    # Backward-compatible aliases: split load flags map to shared model-load.
    legacy_loads = [v for v in (attacker_load, defender_load) if v > 0]
    if len(set(legacy_loads)) > 1:
        print("Conflicting options attacker_load and defender_load with different values. Exiting.")
        sys.exit(1)
    if defender_load == -1:
        print("defender_load=-1 is not supported with shared model mode. Exiting.")
        sys.exit(1)
    if model_load == 0 and legacy_loads:
        model_load = legacy_loads[0]
        print(f"Info: using shared --model-load {model_load} from legacy split load options.")

    if model_load > 0 and load_latest:
        print(f"Conflicting options model_load={model_load} with load_latest set. Exiting.")
        sys.exit(1)

    if human_attacker and human_defender and train_model:
        print("Conflicting options: cannot train model when both sides are human. Exiting.")
        sys.exit(1)

    if interactive:
        pygame.init()
        screen = pygame.display.set_mode(tafl.WINDOW_SIZE)
    else:
        screen = None

    tafl.initialize_groups()
    temp_board = tafl.Board(game_name)  # Just used to initialize global DIM for game_name...

    num_train_games_model = 0

    save_dir = 'models_{}_v{}'.format(game_name.lower(), version)
    if not human_attacker or not human_defender: os.makedirs(save_dir, exist_ok=True)

    has_ai_player = (not human_attacker) or (not human_defender)
    model = None
    if has_ai_player:
        if load_latest:
            model_files = glob(save_dir + '/shared_model_*_games.keras')
            if len(model_files) > 0:
                model_load = max([int(f.split("_")[2]) for f in model_files])  # Parse filenames and get latest
            else:
                model_load = 0
        if model_load == 0:
            model = initialize_model_for_game(
                game_name=game_name,
                num_channels=num_channels,
                use_batchnorm=batchnorm,
                learning_rate=learning_rate,
                model_preset=effective_preset,
            )
        else:
            model = load_model('{}/shared_model_{}_games.keras'.format(save_dir, model_load))
            validate_model_channels(model, num_channels, "Shared")
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            print(f"Shared model reloaded with learning_rate={learning_rate}")
            num_train_games_model = model_load

    stats = None
    if load_latest:
        stats_files = glob(save_dir + '/StatsTracker_*_games.pkl')
        if len(stats_files) > 0:
            stats_load = max([int(f.split("_")[3]) for f in stats_files])
        else:
            stats_load = 0
    if stats_load > 0 and has_ai_player:
        stats = pickle.load(open('{}/StatsTracker_{}_games.pkl'.format(save_dir, stats_load), 'rb'))
    else:  # no previous stats
        stats = StatsTracker(200)

    play = True
    while play:
        if train_model:
            num_train_games_model += 1

        start = timer()

        frac_attackers_to_remove = 0.00
        frac_defenders_to_remove = 0.00
        # Dynamic balancing by piece removal only (shared model always trains together).
        a_win_rate = (stats.a_win_rate_window() + stats.draw_rate_window() / 2.)  # Draws count half
        d_win_rate = 1 - a_win_rate
        if dynamic_train and a_win_rate < 0.40:
            if a_win_rate < 0.30:  # experimental
                frac_defenders_to_remove = 0.50 - a_win_rate
                if log_level >= 2:
                    print(f"a_win_rate is {a_win_rate}, removing {frac_defenders_to_remove} of defenders")

        if dynamic_train and d_win_rate < 0.40:
            if d_win_rate < 0.30:  # experimental
                frac_attackers_to_remove = 0.50 - d_win_rate
                if log_level >= 2:
                    print(f"d_win_rate is {d_win_rate}, removing {frac_attackers_to_remove} of attackers")

        # Calculate exploration parameters with decay
        num_games = num_train_games_model
        attacker_temp = get_exploration_temp(num_games, initial_temp, final_temp, temp_decay)
        defender_temp = get_exploration_temp(num_games, initial_temp, final_temp, temp_decay)
        current_epsilon = get_epsilon(num_games, initial_epsilon, final_epsilon, epsilon_decay)

        if log_level >= 2 and benchmark:
            print(f"Exploration: temp={attacker_temp:.4f}, epsilon={current_epsilon:.4f}")

        play, game_states, final_reward, num_moves_game = run_game(
            model=model,
            human_attacker=human_attacker,
            human_defender=human_defender,
            screen=screen,
            game_name=game_name,
            sample_frac=0.90,
            attacker_temp=attacker_temp,
            defender_temp=defender_temp,
            frac_attackers_to_remove=frac_attackers_to_remove,
            frac_defenders_to_remove=frac_defenders_to_remove,
            epsilon=current_epsilon,
            log_level=log_level,
        )

        end = timer()
        game_duration = end - start

        if has_ai_player:
            stats.add_game_results(final_reward, num_moves_game, game_duration)

        should_print_summary = (
            log_level >= 2 or
            (log_level >= 1 and stats.num_games_total() % log_every == 0)
        )
        if should_print_summary:
            print(
                """Model has played:        {} games,\nNum moves this game:     {} ({:0.3f} sec)"""
                .format(num_train_games_model, num_moves_game, game_duration))
            print(stats)

        if train_model and model is not None and len(game_states) > 0:
            if log_level >= 2:
                print("\nTraining shared model:")

            if use_td:
                targets = compute_td_targets(
                    game_states, final_reward, gamma=gamma, model=model, negate_next_value=False
                )
                if log_level >= 2:
                    print("""          TD targets: {}""".format(
                          ' '.join(['{:+0.4f}'.format(entry) for entry in targets[-15:]])))
            else:
                targets = compute_td_targets(
                    game_states, final_reward, gamma=gamma, model=None, negate_next_value=False
                )
                if log_level >= 2:
                    print("""  Discounted returns: {}""".format(
                          ' '.join(['{:+0.4f}'.format(entry) for entry in targets[-15:]])))
            game_states = np.array(game_states)

            # Symmetry augmentation (mutually exclusive options)
            if use_symmetry:
                game_states = expand_game_states_symmetries(game_states)
                targets = np.tile(targets, 8)
            elif probabilistic_symmetry:
                game_states = apply_random_symmetries_to_batch(game_states)
                # targets unchanged - same size batch

            update_model(model, game_states, targets, use_td=use_td, gamma=gamma, log_level=log_level)

        if (stats.num_games_total() % cache_model_every == 0):  # Save every cache_model_every games
            if num_train_games_model > 0 and model is not None:
                model.save('{}/shared_model_{}_games.keras'.format(save_dir, num_train_games_model))
            if train_model and has_ai_player:
                pickle.dump(stats, open(
                '{}/StatsTracker_{}_games.pkl'.format(save_dir, stats.num_games_total()), 'wb'))
            if exit_after_cache:  # To avoid possible memory leak for long training sessions
                sys.exit()

        if interactive:
            time.sleep(2)
        if num_train_games_model >= 1000000:
            play = False  # Hardcoded cutoff just to make sure things don't go too crazy.

        tafl.cleanup()


if __name__ == '__main__':
    main()
