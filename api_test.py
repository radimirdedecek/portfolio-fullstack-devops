######################################################################################
#                    automated testing using the pytest framework                    #
#                     test fibonacci functions in python section                     #
#                    automated testing using the pytest framework                    #
######################################################################################
import pytest
import time
from api_utils import fast_fibonacci, slow_fibonacci, loop_fibonacci 

# --- Constants for Tests ---
TEST_N = 10 
EXPECTED_RESULT = 55 # F(10) is 55
N_PERF = 30 # Used for caching performance test

# --- Fixture to ensure cache is clear before performance tests ---
@pytest.fixture(autouse=True)
def clear_fib_cache():
    """Clears the cache for fast_fibonacci before running tests."""
    try:
        # Note: We rely on the decorator adding the clear_cache method
        fast_fibonacci.clear_cache()
    except AttributeError:
        # Pass silently if the decorator structure is modified or cache isn't available
        pass

# --- Correctness Tests ---
def test_01_slow_recursive_correctness():
    """Tests the Naive Recursive function (O(2^n))."""
    # Functions return (result, time, string). We only care about the result [0].
    result, _, _ = slow_fibonacci(TEST_N, 0)
    assert result == EXPECTED_RESULT, f"Slow recursive failed: Expected {EXPECTED_RESULT}, got {result}"

def test_02_fast_recursive_correctness():
    """Tests the Memoized Recursive function (O(n))."""
    result, _, _ = fast_fibonacci(TEST_N, 0)
    assert result == EXPECTED_RESULT, f"Fast recursive failed: Expected {EXPECTED_RESULT}, got {result}"

def test_03_loop_iterative_correctness():
    """Tests the Iterative Loop function (O(n))."""
    result, _, _ = loop_fibonacci(TEST_N, 0)
    assert result == EXPECTED_RESULT, f"Loop iterative failed: Expected {EXPECTED_RESULT}, got {result}"

# --- Performance Test ---
def test_04_fast_recursive_performance():
    """
    Tests if the fast_fibonacci function is significantly faster on a second run 
    (proving the O(1) cache hit).
    """
    # 1. First run (populates cache)
    start_time_1 = time.time()
    fast_fibonacci(N_PERF, 0)
    duration_1 = time.time() - start_time_1
    
    # 2. Second run (cache hit)
    start_time_2 = time.time()
    fast_fibonacci(N_PERF, 0)
    duration_2 = time.time() - start_time_2
    
    # Assert the second run is much faster (e.g., at least 10x faster)
    assert duration_2 < duration_1 * 0.1 + 0.001, \
        f"Cache failed: Second run ({duration_2:.6f}s) was not significantly faster than first ({duration_1:.6f}s)."

# --- Edge Case / Validation Test ---
# We use parametrize to run the same test logic with different inputs/expected outcomes
@pytest.mark.parametrize("n_input, expected_result", [
    (-2, 0),   # F(-2) should be 0
    (0, 0),    # F(0)  should be 0
    (1, 1),    # F(1)  should be 1
    (2, 1),    # F(2)  should be 1
    (13, 233)  # F(13) should be 233
])
def test_05_boundary_correctness(n_input, expected_result):
    """Test boundary and small inputs for loop function."""
    result, _, _ = loop_fibonacci(n_input, 0)
    assert result == expected_result, f"F({n_input}) failed. Expected {expected_result}, got {result}"

def test_06_negative_input_loop():
    """Tests that the improved loop_fibonacci correctly returns 0 for n < 1."""
    # Based on your fix: If n < 1, the function should return 0
    result, _, _ = loop_fibonacci(-5, 0)
    assert result == 0, "Loop function did not handle negative input correctly (expected 0)."

#  command line:
#  pytest api_test.py 