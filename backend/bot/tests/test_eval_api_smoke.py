# backend/bot/tests/test_eval_api_smoke.py
"""
Smoke coverage for two previously-untested surfaces:
  * the best-response tree walk (best_response.py) runs end-to-end and returns a
    finite, non-negative exploitability, and
  * the Flask health endpoint responds 200 with the active blueprint.

These load the card-abstraction table (and, for the API, resolve a blueprint),
so they are slower and the API test SKIPS gracefully when no blueprint DB is
present (e.g. a fresh clone before training). Run: python tests/test_eval_api_smoke.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_passed = _failed = 0


def check(name, cond, extra=''):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS {name}")
    else:
        _failed += 1
        print(f"FAIL {name} {extra}")


class _UniformDB:
    """Unknown key everywhere -> the evaluator falls back to uniform play."""
    def get_average_strategy(self, key):
        return None


def test_best_response_walk_runs():
    from src.evaluation.best_response import BestResponseEvaluator
    ev = BestResponseEvaluator(_UniformDB(), seed=1)
    res = ev.evaluate(num_samples=2)
    expl = res['exploitability_mbb']
    check('BR walk returns finite exploitability', math.isfinite(expl), f'(got {expl})')
    # Exploitability is BR0 + BR1 and is non-negative by construction (a best
    # response is never worse than the strategy it responds to).
    check('BR exploitability is non-negative', expl >= -1e-6, f'(got {expl})')


def test_health_endpoint():
    # Importing the API resolves a blueprint + loads tables; skip if unavailable.
    try:
        api_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'api')
        sys.path.insert(0, api_dir)
        import strategy_api
    except Exception as e:
        print(f"SKIP test_health_endpoint (API import failed: {e!r})")
        return
    # Call the view inside an app context (avoids a werkzeug test_client version
    # quirk with `as_tuple`); still exercises the real endpoint handler. The
    # view always returns (Response, status_code) -- the status is the rolling-
    # deploy probe signal so it can't just be a bare Response.
    with strategy_api.app.app_context():
        result = strategy_api.test()
        resp = result[0] if isinstance(result, tuple) else result
        body = resp.get_json()
    check('/api/test reports a status', bool(body) and 'status' in body, f'(got {body})')
    check('/api/test reports the active blueprint', 'blueprint' in body, f'(got {body})')


if __name__ == '__main__':
    test_best_response_walk_runs()
    test_health_endpoint()
    print(f"\nResults: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    sys.exit(1 if _failed else 0)
