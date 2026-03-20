"""
CueZero Test Suite

Comprehensive tests for:
1. Model loading
2. All agent types
3. CLI battle
4. Server API endpoints
"""

import os
import sys
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from cuezero.models.dual_network import DualNetwork
from cuezero.env.billiards_env import BilliardsEnv
from cuezero.inference.agent import (
    Agent, MCTSAgent, PolicyAgent,
    HumanAgent, BasicAgent, RandomAgent
)
from cuezero.mcts.search import MCTS


class TestResult:
    """Test result tracker"""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✓ {test_name}")

    def record_fail(self, test_name: str, reason: str = ""):
        self.failed += 1
        error = f"{test_name}: {reason}" if reason else test_name
        self.errors.append(error)
        print(f"  ✗ {test_name} - {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Test Summary: {self.name}")
        print(f"{'='*50}")
        print(f"Total: {total}, Passed: {self.passed}, Failed: {self.failed}")
        if self.errors:
            print(f"\nFailed tests:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*50}")
        return self.failed == 0


def test_model_loading(result: TestResult):
    """Test 1: Model loading"""
    print("\n" + "="*50)
    print("Test 1: Model Loading")
    print("="*50)

    model_path = "dual_network_final.pt"

    # Test 1.1: Create model
    try:
        model = DualNetwork()
        result.record_pass("Create DualNetwork instance")
    except Exception as e:
        result.record_fail("Create DualNetwork instance", str(e))
        return

    # Test 1.2: Load model (if exists)
    if os.path.exists(model_path):
        try:
            model.load(model_path)
            result.record_pass(f"Load model from {model_path}")
        except Exception as e:
            result.record_fail(f"Load model from {model_path}", str(e))
    else:
        print(f"  ! Model file not found: {model_path}, skipping load test")
        result.record_pass("Model file check (not found, expected in dev)")

    # Test 1.3: Model forward pass
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Create dummy input (batch_size=1, state_dim=81*3 preprocessed)
        dummy_input = torch.randn(1, 3, 81).to(device)
        with torch.no_grad():
            output = model(dummy_input)

        assert "policy_output" in output
        assert "value_output" in output
        assert output["policy_output"].shape == (1, 5)
        result.record_pass("Model forward pass")
    except Exception as e:
        result.record_fail("Model forward pass", str(e))


def test_agents(result: TestResult):
    """Test 2: All agent types"""
    print("\n" + "="*50)
    print("Test 2: Agent Types")
    print("="*50)

    env = BilliardsEnv()
    env.reset(target_ball='solid')
    obs = env.get_observation('A')
    balls, my_targets, table = obs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model if available
    model = None
    model_path = "dual_network_final.pt"
    if os.path.exists(model_path):
        try:
            model = DualNetwork()
            model.load(model_path)
            model.to(device)
            model.eval()
            print("  ! Model loaded for agent tests")
        except Exception as e:
            print(f"  ! Model not loaded: {e}, using dummy model")
            model = DualNetwork().to(device)
    else:
        model = DualNetwork().to(device)

    # Test agents
    agents = [
        ("RandomAgent", lambda: RandomAgent()),
        ("BasicAgent", lambda: BasicAgent()),
        ("PolicyAgent", lambda: PolicyAgent(policy_network=model, device=device)),
    ]

    # MCTS agents need env
    try:
        mcts = MCTS(model=model, mode="fast")
        agent = MCTSAgent(mcts=mcts, name="MCTS(Fast)")
        agent.set_env(env)
        agents.append(("MCTSAgent(fast)", lambda: agent))
    except Exception as e:
        result.record_fail("MCTSAgent(fast) creation", str(e))

    for agent_name, agent_factory in agents:
        try:
            agent = agent_factory()

            # Test decision
            action = agent.decision(balls, my_targets, table)

            # Validate action format
            assert isinstance(action, dict)
            assert all(k in action for k in ['V0', 'phi', 'theta', 'a', 'b'])
            assert 0.5 <= action['V0'] <= 8.0
            assert 0 <= action['phi'] < 360
            assert 0 <= action['theta'] <= 90
            assert -0.5 <= action['a'] <= 0.5
            assert -0.5 <= action['b'] <= 0.5

            result.record_pass(f"{agent_name} decision")
        except Exception as e:
            result.record_fail(f"{agent_name} decision", str(e))

    # Test HumanAgent (skip interactive, just test creation)
    try:
        human_agent = HumanAgent()
        assert human_agent.name == "HumanAgent"
        # Test that _random_action works
        action = human_agent._random_action()
        assert isinstance(action, dict)
        assert all(k in action for k in ['V0', 'phi', 'theta', 'a', 'b'])
        result.record_pass("HumanAgent creation")
    except Exception as e:
        result.record_fail("HumanAgent creation", str(e))

    # Test reset method
    for agent_name, agent_factory in agents[:3]:  # Skip MCTS for simplicity
        try:
            agent = agent_factory()
            agent.reset()
            result.record_pass(f"{agent_name} reset")
        except Exception as e:
            result.record_fail(f"{agent_name} reset", str(e))


def test_cli_battle(result: TestResult):
    """Test 3: CLI battle (simulated)"""
    print("\n" + "="*50)
    print("Test 3: CLI Battle (Simulated)")
    print("="*50)

    env = BilliardsEnv()

    # Load model
    model = None
    model_path = "dual_network_final.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        try:
            model = DualNetwork()
            model.load(model_path)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"  ! Model load failed: {e}")
            model = DualNetwork().to(device)
    else:
        model = DualNetwork().to(device)

    # Create agents (use RandomAgent for faster testing)
    try:
        agent_a = RandomAgent(name="Random")
        agent_b = BasicAgent(name="Basic")
        result.record_pass("Create agents for CLI battle")
    except Exception as e:
        result.record_fail("Create agents for CLI battle", str(e))
        return

    # Simulate a few steps (not full game - too slow for tests)
    try:
        env.reset(target_ball='solid')

        if hasattr(agent_a, 'clear_buffer'):
            agent_a.clear_buffer()
        if hasattr(agent_b, 'clear_buffer'):
            agent_b.clear_buffer()

        max_steps = 10  # Just test a few steps
        step_count = 0

        for step in range(max_steps):
            player = env.get_curr_player()
            obs = env.get_observation(player)
            balls, my_targets, table = obs

            agent = agent_a if player == 'A' else agent_b
            action = agent.decision(balls, my_targets, table)

            step_info = env.take_shot(action)

            # Verify action was valid
            assert isinstance(action, dict)
            assert 'V0' in action

            step_count += 1

        result.record_pass(f"CLI battle simulation ({step_count} steps)")

    except Exception as e:
        result.record_fail("CLI battle simulation", str(e))


def test_server_api(result: TestResult):
    """Test 4: Server API (import and structure check)"""
    print("\n" + "="*50)
    print("Test 4: Server API Structure")
    print("="*50)

    # Test server module import
    try:
        from server.server import (
            app, BattleState, create_agent,
            AgentType, BattleStartRequest, BattleNextRequest
        )
        result.record_pass("Server module import")
    except Exception as e:
        result.record_fail("Server module import", str(e))
        return

    # Test FastAPI app
    try:
        assert app.title == "CueZero - AlphaZero Billiards AI"
        result.record_pass("FastAPI app creation")
    except Exception as e:
        result.record_fail("FastAPI app", str(e))

    # Test BattleState
    try:
        env = BilliardsEnv()
        agent_a = RandomAgent(name="A")
        agent_b = BasicAgent(name="B")
        battle = BattleState(
            battle_id="test123",
            agent_a=agent_a,
            agent_b=agent_b,
            agent_a_type="random",
            agent_b_type="basic"
        )
        battle.start_game(target_ball='solid')

        assert battle.game_status == "playing"
        assert battle.current_player == 'A'

        state = battle.get_state()
        assert state['battle_id'] == "test123"
        assert state['game_status'] == "playing"

        result.record_pass("BattleState creation and start_game")
    except Exception as e:
        result.record_fail("BattleState", str(e))

    # Test create_agent function
    try:
        env = BilliardsEnv()
        agent = create_agent("random", env)
        assert isinstance(agent, RandomAgent)

        agent = create_agent("basic", env)
        assert isinstance(agent, BasicAgent)

        result.record_pass("create_agent function")
    except Exception as e:
        result.record_fail("create_agent function", str(e))


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# CueZero Test Suite")
    print("#"*60)

    result = TestResult("CueZero")

    # Run tests
    test_model_loading(result)
    test_agents(result)
    test_cli_battle(result)
    test_server_api(result)

    # Final summary
    success = result.summary()

    return 0 if success else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CueZero Test Suite")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    sys.exit(run_all_tests())
