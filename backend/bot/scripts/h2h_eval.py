# backend/bot/scripts/h2h_eval.py
"""
Head-to-head with AIVAT on the engine-driven match (src/evaluation/match.py):
candidate A vs B, where B is a blueprint DB or the fitted human POPULATION model
('model'). A's perspective, >0 = A wins. Writes a JSON summary.

Two departures from tests/run_match.py, both to play blueprints the way prod does:

* Untrained faced all-ins, any street: a blueprint facing a jam it never trained
  (a capped-menu v2 facing a capped_pj cap jam; an spr5 key at an SPR the blueprint
  never saw a jam) plays the deployed guard's rule
  (RiverSubgameSolver._facing_deep_raise_guard, uninformed read): call iff equity vs
  the top 20% of hands >= pot odds. Preflop the 20% are ranked by preflop bucket;
  postflop by the bucket's centroid-mean equity in the player's own abstraction
  (prod's _combo_strength). Trained nodes are untouched.

* The human model (HumanModel population layer, v2 keys, root abstraction) samples
  from the fitted human frequencies on the capped engine WITH a voluntary all-in, so
  humans can shove anywhere, as they can live. Where only the voluntary all-in makes
  a shove legal, the model's shove probability is capped at the rate humans actually
  shove (decisions.jsonl: preflop 0.23%, flop 1.7%, turn 1.7%, river 2.0%): the
  population model backs off to the v2 blueprint's prior, whose 'allin' mass (learned
  at low-SPR stack clamps) sits on deep keys too and would otherwise make the model
  shove ~5x more than humans do.

* --jam-read-a <jam_read.json> (V3.11): after the match, re-score every flop/turn jam A
  faced (calling commits A's stack, so nothing follows but the runout) with the pooled
  jam read instead of A's own call probability. The difference is exact per spot (both
  hands are known): (P_read(call) - P_A(call)) * (equity * final pot - call cost), with
  P_read = A's own probability where the read is not confident (the served guard then
  falls through to the same path). Reported as jam_read_a (mbb/hand, A's perspective).
  The match itself is unchanged, so this needs no second run and no pairing.

Run from backend/bot/:
    python scripts/h2h_eval.py --db-a A.db --db-b {B.db|model} --hands 40000 --seed 101 --out x.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np  # noqa: E402

from src.abstractions.postflop_features import load_centroids  # noqa: E402
from src.abstractions.sizing import db_menu_mode, game_kwargs  # noqa: E402
from src.cfr.poker_game import PokerGame  # noqa: E402
from src.evaluation.aivat import AIVATEstimator  # noqa: E402
from src.evaluation.match import BlueprintPlayer, HeadToHeadMatch  # noqa: E402
from src.game.range_tracker import RangeTracker  # noqa: E402
from src.storage.blueprint_db import BlueprintDB  # noqa: E402

_BOT = os.path.join(os.path.dirname(__file__), '..')
V2 = os.path.join(_BOT, 'analysis', 'blueprints', 'blueprint_final_v2.db')
JAM_FRAC = 0.20              # RiverSubgameSolver.JAM_RANGE_FRACTION (and the postflop default)
HUMAN_JAM_RATE = {0: 0.0023, 1: 0.017, 2: 0.017, 3: 0.020}


class GuardedPlayer(BlueprintPlayer):
    """A blueprint player that answers an UNTRAINED faced all-in with prod's guard."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_guard = [0, 0, 0, 0]
        self.n_guard_call = [0, 0, 0, 0]
        self._jam_cache = {}
        self._means = {}
        self._ctx = None

    def act(self, seat, hand, vis, street, legal, pattern, street_pot,
            pot=None, to_call=None, stack=None, state=None):
        self._ctx = (hand, vis, street, pot, to_call, stack, pattern, state, seat)
        return super().act(seat, hand, vis, street, legal, pattern, street_pot,
                           pot=pot, to_call=to_call, stack=stack, state=state)

    def _policy(self, key, legal):
        return BlueprintPlayer._restricted(self, key, legal)

    def _untrained(self, key, legal):
        stored = self._raw(key)
        return (not stored) or sum(max(0.0, stored.get(a, 0.0)) for a in legal) <= 1e-12

    def _strength(self, combo, board):
        if not board:
            return float(int(self.cards.get_bucket(list(combo), None).split('_')[1]))
        n = len(board)
        if n not in self._means:
            street = {3: 'flop', 4: 'turn', 5: 'river'}[n]
            c, bins = load_centroids(street, self.cards.postflop.dir)
            c = np.asarray(c, dtype=float)
            centers = (np.arange(bins) + 0.5) / bins
            sums = c.sum(axis=1)
            self._means[n] = (c @ centers) / np.where(sums > 0, sums, 1.0)
        b = self.cards.get_bucket(list(combo), list(board))
        idx = int(b.split('_')[1]) if isinstance(b, str) else int(b)
        return float(self._means[n][idx])

    def _jam_eq(self, hand, board):
        ck = (frozenset(hand), tuple(board))
        if ck not in self._jam_cache:
            t = RangeTracker(list(hand), self.cards)
            if board:
                t.reveal(list(board))
            scored = sorted(((self._strength(h, board), i)
                             for i, h in enumerate(t.hands) if t.w[i] > 0), reverse=True)
            keep = {i for _, i in scored[:max(1, int(round(JAM_FRAC * len(scored))))]}
            for i in range(len(t.w)):
                if i not in keep:
                    t.w[i] = 0.0
            self._jam_cache[ck] = float(t.hero_equity(list(hand), list(board), n_runouts=200))
        return self._jam_cache[ck]

    def _restricted(self, key, legal):
        hand, vis, street, pot, to_call, stack, pattern, _state, _seat = self._ctx
        if (pattern.endswith('a') and to_call and to_call > 0 and 'call' in legal
                and 'fold' in legal and self._untrained(key, legal)):
            eq = self._jam_eq(hand, vis)
            call_cost = min(to_call, stack)
            final_pot = pot + to_call if to_call <= stack else pot - to_call + 2.0 * stack
            act = 'call' if eq * final_pot - call_cost >= 0.0 else 'fold'
            self.n_guard[street] += 1
            self.n_guard_call[street] += act == 'call'
            return np.array([1.0 if a == act else 0.0 for a in legal])
        return self._policy(key, legal)


class ModelPlayer(GuardedPlayer):
    """The fitted human population model as a player (v2 keys, root abstraction)."""

    def __init__(self, src, cards, rng, fn):
        super().__init__(src, cards, rng, menu_mode='capped')
        self.game = PokerGame(**{**game_kwargs('capped'), 'voluntary_allin': True})
        self._capped = PokerGame(**game_kwargs('capped'))
        self._fn = fn
        self._mcache = {}
        self.n_capped = 0

    def _policy(self, key, legal):
        ck = (key, legal)
        if ck not in self._mcache:
            self._mcache[ck] = np.asarray(self._fn(key, legal), dtype=float)
        probs = self._mcache[ck]
        _hand, _vis, street, _pot, _to_call, stack, _pattern, state, seat = self._ctx
        if 'allin' in legal and state is not None:
            if 'allin' not in self._capped.state_legal_actions(street, state, seat, stack):
                i = legal.index('allin')
                cap = HUMAN_JAM_RATE[street]
                if probs[i] > cap:
                    self.n_capped += 1
                    rest = probs.sum() - probs[i]
                    probs = probs * ((1.0 - cap) / rest if rest > 1e-12 else 0.0)
                    probs[i] = cap
        return probs


def score_jam_read(rec, path, db_a, hands):
    """The jam read's exact counterfactual gain at A's faced flop/turn jams (see the
    module docstring), in mbb/hand over all `hands`."""
    from src.cfr.keys import spr_bucket
    from src.cfr.poker_game import STARTING_STACK
    from src.exploitation.jam_read import JamRead, pot_type
    from src.subgame.river_subgame_solver import RiverSubgameSolver
    from eval_facing_jams import exact_equity
    read = JamRead.load(path, db_a)
    strength = RiverSubgameSolver(db_a)._combo_strength
    per_hand = []
    spots = changed = 0
    for r in rec:
        a = r['seat_of_A']
        b = 1 - a
        gain = 0.0
        pre = [e['action'] for e in r['events'] if e['type'] == 'action' and e['street'] == 0]
        street_pot = {}
        for e in r['events']:
            if e['type'] == 'street_start':
                street_pot[e['street']] = e['pot']
                continue
            if (e['type'] != 'action' or e.get('seat') != a or e['street'] not in (1, 2)
                    or not e['pattern'].endswith('a') or e['action'] not in ('call', 'fold')):
                continue
            inv = e['invested']
            to_call = inv[b] - inv[a]
            stack = STARTING_STACK - inv[a]
            if to_call <= 0 or to_call < stack - 1e-6:
                continue                              # money behind: not the jam guard's spot
            pot = inv[0] + inv[1]
            cost = min(to_call, stack)
            final_pot = pot + cost if to_call <= stack else pot - to_call + 2.0 * stack
            legal = e['legal']
            probs = np.asarray(e['probs'], dtype=float)
            p_a = float(probs[legal.index('call')] / probs.sum())
            street = ('flop', 'turn')[e['street'] - 1]
            act, _ = read.decide(list(r['hand_a']), list(e['vis']), street, pot_type(pre),
                                 spr_bucket('spr5', e['street'], street_pot[e['street']]),
                                 final_pot, cost, strength)
            spots += 1
            if act is None:
                continue
            p_read = 1.0 if act == 'call' else 0.0
            if abs(p_read - p_a) > 1e-12:
                changed += 1
                eq = exact_equity(list(r['hand_a']), list(r['hand_b']), list(e['vis']), seed='h2h')
                gain += (p_read - p_a) * (eq * final_pot - cost)
        per_hand.append(gain)
    x = np.asarray(per_hand + [0.0] * (hands - len(per_hand))) * 500.0     # chips -> mbb
    return {'spots': spots, 'changed': changed, 'delta_mbb': float(x.mean()),
            'se_mbb': float(x.std(ddof=1) / np.sqrt(len(x)))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--db-a', required=True)
    p.add_argument('--db-b', required=True, help="blueprint path, or 'model'")
    p.add_argument('--model-dir', default=os.path.join(_BOT, 'analysis', 'opponent_models'),
                   help='fitted opponent models (model_population.json)')
    p.add_argument('--hands', type=int, default=40000)
    p.add_argument('--seed', type=int, default=101)
    p.add_argument('--out', required=True)
    p.add_argument('--jam-read-a', default=None,
                   help="a fitted jam read for --db-a: re-score A's faced flop/turn jams with it")
    args = p.parse_args()

    da = BlueprintDB(args.db_a, read_only=True)
    is_model = args.db_b == 'model'
    db = BlueprintDB(V2 if is_model else args.db_b, read_only=True)
    mm_a, mm_b = db_menu_mode(da), db_menu_mode(db)
    print(f"A {args.db_a} ({mm_a})\nB {'human population model' if is_model else args.db_b} "
          f"({mm_b})\nhands {args.hands} seed {args.seed}", flush=True)
    m = HeadToHeadMatch(da, db, seed=args.seed, menu_mode_a=mm_a, menu_mode_b=mm_b)
    m.pa = GuardedPlayer(da, m.pa.cards, m.rng, menu_mode=mm_a)
    if is_model:
        from src.exploitation.opponent_model import HumanModel
        if not os.path.exists(os.path.join(args.model_dir, 'model_population.json')):
            # HumanModel would silently back off to the blueprint prior alone.
            sys.exit(f"no model_population.json in {args.model_dir} (pass --model-dir)")
        hm = HumanModel(args.model_dir, db)
        m.pb = ModelPlayer(db, m.pb.cards, m.rng, hm.strategy_fn_for('__population__'))
    else:
        m.pb = GuardedPlayer(db, m.pb.cards, m.rng, menu_mode=mm_b)
    rec = []
    t0 = time.time()
    m.evaluate(num_hands=args.hands, record=rec, progress_every=max(1, args.hands // 10))
    res = AIVATEstimator(db, seed=args.seed).estimate(rec)
    jams = [0, 0, 0, 0]
    for r in rec:
        for e in r['events']:
            if e['type'] == 'action' and e['action'] == 'allin':
                jams[e['street']] += 1
    out = dict(a=args.db_a, b=args.db_b, hands=args.hands, seed=args.seed,
               aivat_mbb=res['aivat_mbb'], aivat_se=res['aivat_stderr_mbb'],
               raw_mbb=res['raw_mbb'], raw_se=res['raw_stderr_mbb'],
               var_reduction=res['var_reduction'], jams_by_street=jams,
               guard_a=[m.pa.n_guard, m.pa.n_guard_call],
               guard_b=[m.pb.n_guard, m.pb.n_guard_call],
               model_jam_capped=getattr(m.pb, 'n_capped', 0), seconds=time.time() - t0)
    if args.jam_read_a:
        out['jam_read_a'] = score_jam_read(rec, args.jam_read_a, da, args.hands)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
