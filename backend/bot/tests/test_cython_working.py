from src.cfr.information_set import InformationSet
import time
import numpy as np


def test_cython_acceleration():
    """Test if Cython acceleration is working"""

    info_set = InformationSet()
    legal_actions = ['fold', 'call', 'raise_large']

    # Setup some regrets for testing
    info_set.cumulative_regrets = {
        'fold': 0.5,
        'call': -0.8,
        'raise_large': 1.2
    }

    # Test get_strategy performance
    iterations = 10000
    reach_prob = 0.5

    print("🏃 Testing Cython acceleration...")

    start_time = time.time()
    for _ in range(iterations):
        strategy = info_set.get_strategy(legal_actions, reach_prob)
    duration = time.time() - start_time

    print(f"📊 Results ({iterations} iterations):")
    print(
        f"   Duration: {duration:.3f}s ({duration/iterations*1000:.3f}ms per call)")
    print(f"   Strategy result: {[f'{x:.3f}' for x in strategy]}")

    # Test get_average_strategy
    start_time = time.time()
    for _ in range(iterations):
        avg_strategy = info_set.get_average_strategy(legal_actions)
    avg_duration = time.time() - start_time

    print(
        f"   Average strategy duration: {avg_duration:.3f}s ({avg_duration/iterations*1000:.3f}ms per call)")
    print(f"   Average strategy result: {[f'{x:.3f}' for x in avg_strategy]}")

    # Check if using Cython
    from src.cfr.information_set import CYTHON_AVAILABLE
    print(
        f"\n✅ Cython Status: {'ENABLED' if CYTHON_AVAILABLE else 'FALLBACK TO PYTHON'}")


if __name__ == "__main__":
    test_cython_acceleration()
