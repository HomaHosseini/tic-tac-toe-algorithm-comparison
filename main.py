# Import necessary libraries
import numpy as np
import random
import time
from typing import List, Tuple, Optional

algorithm_time = {"minimax": [0,0], "alpha_beta": [0,0], "evaluation_based":[0,0], "monte_carlo_tree_search": [0,0]}
# Tic-Tac-Toe game environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0: empty, 1: player 1, -1: player 2

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)

    def is_valid_move(self, x: int, y: int) -> bool:
        return self.board[x, y] == 0

    def make_move(self, x: int, y: int, player: int) -> bool:
        if self.is_valid_move(x, y):
            self.board[x, y] = player
            return True
        return False

    def check_winner(self) -> Optional[int]:
        # Check rows, columns, and diagonals
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 3:
                return np.sign(sum(self.board[:, i]))
        if abs(self.board.trace()) == 3:
            return np.sign(self.board.trace())
        if abs(np.fliplr(self.board).trace()) == 3:
            return np.sign(np.fliplr(self.board).trace())
        return None if 0 in self.board else 0  # 0: draw, None: ongoing game

    def get_available_moves(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def display(self):
        display_board = self.board.astype(str)
        display_board[display_board == "0"] = "."
        display_board[display_board == "1"] = "X"
        display_board[display_board == "-1"] = "O"
        print("\n".join([" ".join(row) for row in display_board]))
        print()

def best_available_move(state: TicTacToe, player: int, algorithm_value) -> Tuple[int, int]:
    best_m = None
    best_val = float('-inf') if player == 1 else float('inf')
    for m in state.get_available_moves():
        x, y = m
        state.make_move(x, y, player)
        v = algorithm_value(state, -player)
        state.board[x,y] = 0
        if (player == 1 and v > best_val) or (player == -1 and v < best_val):
            best_val = v
            best_m = m
    return best_m

# Minimax method
def minimax(state: TicTacToe, player: int) -> Tuple[int, int]:
    def minimax_value(state: TicTacToe, player: int) -> int:
        terminal = state.check_winner()
        if terminal is not None:
            return terminal
        best = float('-inf') if player == 1 else float('inf')
        for m in state.get_available_moves():
            x, y = m
            state.make_move(x, y, player)
            val = minimax_value(state, -player)
            state.board[x,y] = 0
            best = max(best, val) if player == 1 else min(best, val)
        return best
    return best_available_move(state, player, minimax_value)

# Alpha-Beta Pruning
def alpha_beta(state: TicTacToe, player: int) -> Tuple[int, int]:
    def alpha_beta_value(state: TicTacToe, player: int, alpha: int, beta: int) -> int:
        terminal = state.check_winner()
        if terminal is not None:
            return terminal
        v = float('-inf') if player == 1 else float('inf')
        for m in state.get_available_moves():
            x, y = m
            state.make_move(x, y, player)
            v = max(v, alpha_beta_value(state, -player, alpha, beta)) if player == 1 else \
                min(v, alpha_beta_value(state, -player, alpha, beta))
            state.board[x,y] = 0
            if player == 1:
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            else:
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v
        
    best_m = None
    best_val = float('-inf') if player == 1 else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    for m in state.get_available_moves():
        x, y = m
        state.make_move(x, y, player)
        v = alpha_beta_value(state, -player, alpha, beta)
        state.board[x,y] = 0

        if (player == 1 and v > best_val) or (player == -1 and v < best_val):
            best_val = v
            best_m = m

    return best_m

def best_available_move_eval(state: TicTacToe, player: int, algorithm_value) -> Tuple[int, int]:
    best_m = None
    best_val = float('-inf') if player == 1 else float('inf')
    for m in state.get_available_moves():
        x, y = m
        state.make_move(x, y, player)
        v = algorithm_value(state, player)
        state.board[x,y] = 0
        if (player == 1 and v > best_val) or (player == -1 and v < best_val):
            best_val = v
            best_m = m
    return best_m

def evaluation_based(state: TicTacToe, player: int) -> Tuple[int, int]:
    def heuristic_value(state: TicTacToe, player: int) -> int:
        board = state.board
        score = 0

        lines = [
            board[i, :] for i in range(3)  
        ] + [
            board[:, j] for j in range(3)  
        ] + [
            board.diagonal(), 
            np.fliplr(board).diagonal() 
        ]

        for line in lines:
            p = np.count_nonzero(line == player)
            o = np.count_nonzero(line == -player)

            if p > 0 and o == 0:
                if p == 3:
                    score += 1000  
                elif p == 2:
                    score += 100  
                else:
                    score += 10 
            elif o > 0 and p == 0:
                if o == 2:
                    score -= 100  
                else:
                    score -= 10  

        if board[1, 1] == player:
            score += 5
        elif board[1, 1] == -player:
            score -= 5

        for x, y in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            if board[x, y] == player:
                score += 3
            elif board[x, y] == -player:
                score -= 3

        return score

    return best_available_move_eval(state, player, heuristic_value)



simulation_count = 10
# Monte Carlo Tree Search (MCTS)
def monte_carlo_tree_search(state: TicTacToe, player: int) -> Tuple[int, int]:
    
    def simulate(state: TicTacToe, player: int) -> int:
        while state.check_winner() is None:
            move = random.choice(state.get_available_moves())
            state.make_move(*move, player)
            player = -player
        return state.check_winner()

    best_m = None
    best_val = float('-inf')  if player == 1 else float('inf')
    for m in state.get_available_moves():
        wins = 0
        for _ in range(simulation_count):
            t = TicTacToe()
            t.board = state.board.copy()
            t.make_move(*m, player)
            result = simulate(t, -player)
            wins += result
            if (player == 1 and wins > best_val) or (player == -1 and wins < best_val):
                best_val = wins
                best_m = m
    return best_m

# Simulate a match between two methods
def simulate_game(method1, method2):
    game = TicTacToe()
    players = [1, -1]
    methods = {1: method1, -1: method2}
    while game.check_winner() is None:
        current_player = players[0]
        start_time = time.time()
        move = methods[current_player](game, current_player)
        global algorithm_time
        algorithm_time[methods[current_player].__name__][0] += time.time() - start_time
        algorithm_time[methods[current_player].__name__][1] += 1
        game.make_move(*move, current_player)
        players.reverse()
    game.display()
    return game.check_winner()


if __name__ == "__main__":
    game_count = 30
    result = [0,0,0]
    for i in range (game_count):
        winner = simulate_game(alpha_beta, monte_carlo_tree_search)
        if winner == 1:
            print("Player 1 (Method 1) wins!")
            result[0] += 1
        elif winner == -1:
            print("Player 2 (Method 2) wins!")
            result[2] += 1
        else:
            print("It's a draw!")
            result[1] += 1
    print(result)
    
    print(algorithm_time)
