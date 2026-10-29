# Tic-Tac-Toe AI: Algorithm Comparison

An educational Python project exploring classical AI decision-making through Tic-Tac-Toe. The project implements Minimax, Alpha-Beta Pruning, heuristic-based evaluation, and Monte Carlo rollout simulations, and runs matches between selected algorithms.

## Algorithms

| Algorithm                | Description                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| **Minimax**              | Explores possible game outcomes to select a move based on optimal play.                    |
| **Alpha-Beta Pruning**   | Enhances Minimax by pruning branches that cannot influence the final decision.             |
| **Heuristic Evaluation** | Evaluates board positions using potential winning lines, the center, and corner positions. |
| **Monte Carlo Rollouts** | Estimates candidate moves by simulating random games and accumulating their outcomes.      |

**Note:** The Monte Carlo implementation is a simplified rollout-based method rather than a complete Monte Carlo Tree Search (MCTS) implementation.

## Features

- NumPy-based Tic-Tac-Toe board representation.
- Legal move generation and winner detection.
- Multiple AI decision-making strategies.
- Automated matches between selected algorithms.
- Basic execution-time and move-count tracking.
- Win, loss, and draw reporting.

## Requirements

- Python 3.9 or newer
- NumPy

## Running Experiments

The current script runs 30 simulated games between Alpha-Beta Pruning and the Monte Carlo rollout method.

The output includes the final board of each game, the winner or draw, aggregate game results, and accumulated execution times and move counts.
