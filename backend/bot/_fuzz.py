import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))
from src.cfr.poker_game import PokerGame, STARTING_STACK
from src.cfr.blueprint_trainer import BlueprintTrainer

game = PokerGame()

def walk(history, street, sp, p0_inv, p1_inv, p0_stack, p1_stack, depth, log):
    """Random tree walk asserting chip invariants at every node."""
    if depth > 60:
        return
    # invariants before acting
    pot = game.calculate_current_pot(sp, history, street, p0_inv, p1_inv)
    p0_this = game.get_player_contribution_this_round(history, street, sp, 0, p0_inv, p1_inv)
    p1_this = game.get_player_contribution_this_round(history, street, sp, 1, p0_inv, p1_inv)
    p0_total = p0_inv + p0_this
    p1_total = p1_inv + p1_this
    ctx = f"st={street} hist={history} sp={sp} inv=({p0_inv},{p1_inv})"
    # preflop street-0 contributions include the posted blinds (sum to pot);
    # postflop they sum to the pot increment for that street.
    expected = pot if street == 0 else (pot - sp)
    assert abs((p0_this + p1_this) - expected) < 1e-6, f"contrib!=potinc {ctx} p0_this={p0_this} p1_this={p1_this} pot={pot}"
    assert p0_total <= STARTING_STACK + 1e-6, f"p0 over stack {p0_total} {ctx}"
    assert p1_total <= STARTING_STACK + 1e-6, f"p1 over stack {p1_total} {ctx}"
    assert p0_total >= -1e-6 and p1_total >= -1e-6, f"neg total {ctx}"
    assert pot <= 2*STARTING_STACK + 1e-6, f"pot too big {pot} {ctx}"

    if game.is_terminal(history, street):
        return
    cur = game._acting_player(len(history), street)
    legal = game.get_legal_actions(street, history, sp, cur, p0_stack, p1_stack, p0_inv, p1_inv)
    if not legal:
        if street < 3:
            new_p0 = STARTING_STACK - p0_total
            new_p1 = STARTING_STACK - p1_total
            assert new_p0 >= -1e-6 and new_p1 >= -1e-6, f"neg stack on street transition {ctx} {new_p0} {new_p1}"
            walk([], street+1, pot, p0_total, p1_total, new_p0, new_p1, depth+1, log)
        return
    for a in legal:
        cost = game._action_cost(a, street, history, sp, cur, p0_inv, p1_inv)
        rem = p0_stack if cur == 0 else p1_stack
        assert cost <= rem + 1e-6, f"action {a} cost {cost} > remaining {rem} {ctx}"
        assert cost >= -1e-6, f"neg cost {a} {cost} {ctx}"
        if cur == 0:
            np0, np1 = p0_stack - cost, p1_stack
        else:
            np0, np1 = p0_stack, p1_stack - cost
        walk(history+[a], street, sp, p0_inv, p1_inv, np0, np1, depth+1, log)

random.seed(1)
for trial in range(3000):
    game._calc_cache.clear()
    walk([], 0, 3, 0.0, 0.0, STARTING_STACK-1, STARTING_STACK-2, 0, [])
print("tree-walk invariants OK")

# cfr return-value fuzz
random.seed(2)
trainer = BlueprintTrainer()
ca = trainer.game_adapter.card_abstractions
for i in range(4000):
    p0, p1, comm = trainer.deal_random_hand()
    trainer._p0_preflop = ca.get_bucket(p0, None)
    trainer._p1_preflop = ca.get_bucket(p1, None)
    trainer._p0_postflop = {s: ca.get_bucket(p0, comm[:2+s]) for s in (1,2,3)}
    trainer._p1_postflop = {s: ca.get_bucket(p1, comm[:2+s]) for s in (1,2,3)}
    trainer.game._calc_cache.clear()
    for up in (0,1):
        v = trainer.cfr(p0, p1, comm, [], 0, up, 0, i, 3)
        assert abs(v) <= STARTING_STACK + 1e-6, f"cfr value {v} out of range"
for k, info in trainer.info_sets.items():
    for a, r in info.cumulative_regrets.items():
        assert r >= -1e-9, f"neg regret {k} {a} {r}"
print("cfr return + regret invariants OK")
