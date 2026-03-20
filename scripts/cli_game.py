"""
CueZero CLI Game - Command line interface for agent battles

Features:
- Support multiple agent types (MCTS, Policy, Basic, Human, Random)
- Display score and win rate statistics
- Support switching first/second turn and ball types
- Show match progress and results
"""

import argparse
import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from cuezero.env.billiards_env import BilliardsEnv
from cuezero.models.dual_network import DualNetwork
from cuezero.mcts.search import MCTS
from cuezero.inference.agent import (
    Agent, MCTSAgent, PolicyAgent,
    HumanAgent, BasicAgent, RandomAgent
)


def create_agent(agent_type: str, env, model, device) -> Agent:
    """Create an agent based on type string.

    Args:
        agent_type: One of 'human', 'mcts_fast', 'mcts_full', 'policy', 'basic', 'random'
        env: BilliardsEnv instance
        model: DualNetwork instance
        device: torch device

    Returns:
        Agent instance
    """
    agent_type = agent_type.lower()

    if agent_type == 'human':
        return HumanAgent(name="Human")
    elif agent_type == 'mcts_fast':
        mcts = MCTS(model=model, mode="fast")
        agent = MCTSAgent(mcts=mcts, name="MCTS(Fast)")
        agent.set_env(env)
        return agent
    elif agent_type == 'mcts_full':
        mcts = MCTS(model=model, mode="full")
        agent = MCTSAgent(mcts=mcts, name="MCTS(Full)")
        agent.set_env(env)
        return agent
    elif agent_type == 'policy':
        return PolicyAgent(policy_network=model, device=device, name="Policy")
    elif agent_type == 'basic':
        return BasicAgent(name="Basic")
    elif agent_type == 'random':
        return RandomAgent(name="Random")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def print_current_stats(current_game: int, total_games: int, results: dict,
                        agent_a_name: str, agent_b_name: str):
    """Print current statistics before each game.

    Args:
        current_game: Current game number (1-indexed)
        total_games: Total games to play
        results: Results dictionary with 'A', 'B', 'DRAW' keys
        agent_a_name: Name of agent A
        agent_b_name: Name of agent B
    """
    if current_game == 1:
        print("\n" + "="*60)
        print("【初始状态】比赛即将开始")
        print("="*60)
        return

    played_games = current_game - 1
    a_wins = results['A']
    b_wins = results['B']
    draws = results['DRAW']

    # Calculate win rates
    a_win_rate = a_wins / played_games * 100 if played_games > 0 else 0
    b_win_rate = b_wins / played_games * 100 if played_games > 0 else 0
    draw_rate = draws / played_games * 100 if played_games > 0 else 0

    print("\n" + "-"*60)
    print(f"【累计比分】已完成 {played_games}/{total_games} 局")
    print(f"{agent_a_name}: {a_wins} 胜 | {agent_b_name}: {b_wins} 胜 | 平局：{draws}")
    print(f"胜率：{agent_a_name} {a_win_rate:.1f}% | {agent_b_name} {b_win_rate:.1f}% | 平局 {draw_rate:.1f}%")
    print("-"*60)


def print_game_result(game_num: int, winner: str, agent_a_name: str, agent_b_name: str):
    """Print the result of a single game.

    Args:
        game_num: Game number
        winner: 'A', 'B', or 'DRAW'
        agent_a_name: Name of agent A
        agent_b_name: Name of agent B
    """
    print(f"\n------- 第 {game_num} 局比赛结束 -------")
    if winner == 'DRAW':
        print("结果：平局！")
    elif winner == 'A':
        print(f"结果：{agent_a_name} 获胜！")
    else:
        print(f"结果：{agent_b_name} 获胜！")
    print("-"*60)


def run_battle(agent_a: Agent, agent_b: Agent, env: BilliardsEnv,
               target_ball: str = 'solid', verbose: bool = True):
    """Run a single battle between two agents.

    Args:
        agent_a: First agent
        agent_b: Second agent
        env: PoolEnv instance
        target_ball: 'solid', 'stripe', or 'random'
        verbose: Print game progress

    Returns:
        winner: 'A', 'B', or 'DRAW'
    """
    # Reset environment
    if target_ball == 'random':
        target_ball = random.choice(['solid', 'stripe'])
    env.reset(target_ball=target_ball)

    # Clear agent buffers
    if hasattr(agent_a, 'clear_buffer'):
        agent_a.clear_buffer()
    if hasattr(agent_b, 'clear_buffer'):
        agent_b.clear_buffer()

    players = [agent_a, agent_b]

    if verbose:
        print(f"\n------- 第 1 局比赛开始 -------")
        print(f"Player A: {agent_a.name} (目标：{target_ball})")
        print(f"Player B: {agent_b.name}")

    while True:
        player = env.get_curr_player()
        obs = env.get_observation(player)
        balls, my_targets, table = obs

        if verbose:
            print(f"\n[第{env.hit_count}次击球] Player {player} 回合")

        # Get action from appropriate agent
        if player == 'A':
            agent = agent_a
        else:
            agent = agent_b

        try:
            action = agent.decision(balls, my_targets, table)
        except Exception as e:
            print(f"[ERROR] {agent.name} decision error: {e}")
            action = None

        if action is None:
            # Fallback to random action
            action = {
                'V0': round(random.uniform(0.5, 8.0), 2),
                'phi': round(random.uniform(0, 360), 2),
                'theta': round(random.uniform(0, 90), 2),
                'a': round(random.uniform(-0.5, 0.5), 3),
                'b': round(random.uniform(-0.5, 0.5), 3),
            }

        # Take shot
        step_info = env.take_shot(action)

        # Check if done
        done, info = env.get_done()

        if not done:
            if step_info.get('ENEMY_INTO_POCKET'):
                if verbose:
                    print(f"对方球入袋：{step_info['ENEMY_INTO_POCKET']}")
        else:
            # Game over
            winner = info.get('winner', 'DRAW')
            return winner


def main():
    parser = argparse.ArgumentParser(description="CueZero CLI Battle")
    parser.add_argument("--agent-a", type=str, default="mcts_fast",
                        choices=['human', 'mcts_fast', 'mcts_full', 'policy', 'basic', 'random'],
                        help="Agent A type (default: mcts_fast)")
    parser.add_argument("--agent-b", type=str, default="basic",
                        choices=['human', 'mcts_fast', 'mcts_full', 'policy', 'basic', 'random'],
                        help="Agent B type (default: basic)")
    parser.add_argument("--games", type=int, default=5,
                        help="Number of games to play (default: 5)")
    parser.add_argument("--model", type=str, default="dual_network_final.pt",
                        help="Path to model file (default: dual_network_final.pt)")
    parser.add_argument("--no-verbose", action="store_true",
                        help="Disable verbose output")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None)")

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.model}")
    model = DualNetwork()

    try:
        model.load(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Using randomly initialized model")

    model.to(device)
    model.eval()

    # Create environment
    env = BilliardsEnv()

    # Create agents
    print(f"\nCreating agents...")
    agent_a = create_agent(args.agent_a, env, model, device)
    agent_b = create_agent(args.agent_b, env, model, device)

    print(f"Agent A: {agent_a.name} ({args.agent_a})")
    print(f"Agent B: {agent_b.name} ({args.agent_b})")

    # Initialize results
    results = {'A': 0, 'B': 0, 'DRAW': 0}

    # Ball type rotation
    ball_types = ['solid', 'solid', 'stripe', 'stripe']

    print(f"\n开始 {args.games} 局对战，{agent_a.name} vs {agent_b.name}")
    print("="*60)

    # Main game loop
    for i in range(args.games):
        current_game = i + 1

        # Print current stats
        print_current_stats(current_game, args.games, results,
                           agent_a.name, agent_b.name)

        # Get ball type for this game
        ball_type = ball_types[i % 4]

        # Run battle
        winner = run_battle(agent_a, agent_b, env,
                           target_ball=ball_type,
                           verbose=not args.no_verbose)

        # Update results
        results[winner] += 1

        # Print game result
        print_game_result(current_game, winner, agent_a.name, agent_b.name)

    # Print final results
    print("\n" + "="*60)
    print("最终对战结果")
    print("="*60)
    print(f"总对局数：{args.games}")
    print(f"{agent_a.name} 获胜：{results['A']} 局 ({results['A']/args.games*100:.1f}%)")
    print(f"{agent_b.name} 获胜：{results['B']} 局 ({results['B']/args.games*100:.1f}%)")
    print(f"平局：{results['DRAW']} 局 ({results['DRAW']/args.games*100:.1f}%)")
    print("="*60)

    if results['A'] > results['B']:
        print(f"🏆 {agent_a.name} 最终获胜！")
    elif results['B'] > results['A']:
        print(f"🏆 {agent_b.name} 最终获胜！")
    else:
        print("双方最终战平！")

    print("\n对战结束，感谢观看！")


if __name__ == '__main__':
    main()
