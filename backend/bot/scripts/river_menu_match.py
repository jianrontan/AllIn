"""Solver-on menu H2H through two synchronized live GameSession views.

A is the candidate; seats alternate on independent seeded deals. Uses existing
AIVAT chance controls c1/c3 only: c2 is disabled because the opponent is another
solver, not the blueprint model. No blueprint action-probability control is used.
Smoke runs are execution checks; a menu verdict requires many hands/seeds.
"""
import argparse
import copy
from contextlib import closing
import json
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage.blueprint_db import BlueprintDB, FrozenBlueprint
from src.game.game_session import GameSession, session_kwargs, advance_bot_turns
from src.game.range_tracker import RangeTracker
from src.subgame.river_subgame_solver import RiverSubgameSolver
from src.evaluation.aivat import AIVATEstimator
from src.evaluation.policy_identity import policy_identity
from measure_turn_match import _aivat_record

_SHARED=('p0_stack','p1_stack','p0_invested','p1_invested','street','starting_pot',
         'history','bet_pattern','status')


def play_hand(db,bots,seed,index):
    state=random.getstate()
    try:
        return _play_hand(db,bots,seed,index)
    finally:
        random.setstate(state)


def _play_hand(db,bots,seed,index):
    random.seed(f'{seed}|{index}')
    kw=session_kwargs(db)
    common=GameSession.new('match','offline',**kw).to_dict()
    candidate_seat=index%2
    by_seat=[bots[0] if seat==candidate_seat else bots[1] for seat in (0,1)]
    sessions=[]
    for seat,bot in enumerate(by_seat):
        bot._rng=np.random.default_rng(np.random.SeedSequence([seed,index,seat]))
        data=copy.deepcopy(common)
        data['human_seat']=1-seat
        session=GameSession.from_dict(data,strategy_fn=bot.range_model_fn(),
                                      hero_strategy_fn=bot.hero_range_model_fn(),**kw)
        session.data['opp_range']=RangeTracker(data[f'p{seat}_cards'],session.cards).to_dict()
        session.data['bot_range']=RangeTracker((),session.cards).to_dict()
        sessions.append(session)
    originals=[s.apply_action for s in sessions]
    log={'streets':[],'folder':None}
    quality=[]
    def apply(seat,action,solved_hero_probs=None):
        active=sessions[seat]
        if action=='fold':
            log['folder']=active.current_player()
        before=active.data['street']
        result=originals[seat](action,solved_hero_probs=solved_hero_probs)
        # Record only successful actions; advance_bot_turns can retry a rejected size.
        log['streets'].append(before)
        debug=getattr(by_seat[seat],'last_debug',None) or {}
        if debug.get('solved'):
            quality.append(dict(candidate=seat==candidate_seat,iters=debug.get('iters',0),
                                converged=debug.get('converged',False),
                                budget_truncated=(not debug.get('converged',False)
                                                  and debug.get('iters',0)<by_seat[seat].max_iters)))
        originals[1-seat](action)
        for field in _SHARED:
            if sessions[0].data[field]!=sessions[1].data[field]:
                raise RuntimeError(f'GameSession views diverged: {field}')
        return result
    for seat,session in enumerate(sessions):
        session.apply_action=lambda action,solved_hero_probs=None,seat=seat: apply(seat,action,solved_hero_probs)
    steps=0
    while sessions[0].data['status']=='in_hand':
        seat=sessions[0].current_player()
        advance_bot_turns(sessions[seat],by_seat[seat])
        steps+=1
        if steps>120:
            raise RuntimeError('Match failed to finish; discard no partial hands')
    candidate=sessions[candidate_seat]
    record=_aivat_record(candidate,candidate_seat,log)
    record['river_range']=None  # no model-dependent river chance control
    record['result']=-candidate.data['human_net']
    record['solve_quality']=quality
    if abs(sessions[0].data['human_net']+sessions[1].data['human_net'])>1e-8:
        raise RuntimeError('Match is not zero sum')
    return record


def load_checkpoint(path, config, sources, identity=None):
    saved=json.loads(path.read_text())
    if identity is None or saved.get('policy_identity') != identity:
        raise ValueError('Checkpoint policy identity missing or changed; legacy checkpoints require a fresh run')
    if saved['source_sha256']!=sources:
        raise ValueError('Checkpoint source hashes differ; refusing mixed-policy results')
    for key in ('db','a','b','hands','iters','budget','seed','depth_a','depth_b'):
        if saved['config'].get(key)!=config.get(key):
            raise ValueError(f'Checkpoint configuration differs: {key}')
    records=saved['records']
    if saved['completed']!=len(records) or not 0<=len(records)<=config['hands']:
        raise ValueError('Invalid checkpoint hand count')
    if any(r['seat_of_A']!=i%2 for i,r in enumerate(records)):
        raise ValueError('Checkpoint seat order differs')
    for key,value in (('backend',os.environ.get('ALLIN_RIVER_BACKEND','python')),
                      ('algorithm',os.environ.get('ALLIN_RIVER_ALGORITHM','cfrplus'))):
        if key in saved and saved[key]!=value:
            raise ValueError(f'Checkpoint {key} differs')
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--a',default='F',choices=['A','F','F_EXTENDED'])
    parser.add_argument('--b',default='A',choices=['A','F','F_EXTENDED'])
    parser.add_argument('--hands',type=int,default=20)
    parser.add_argument('--iters',type=int,default=275)
    parser.add_argument('--budget',type=float,default=24.)
    parser.add_argument('--seed',type=int,default=39)
    parser.add_argument('--resume',action='store_true',help='reuse validated complete-hand checkpoint')
    parser.add_argument('--progress-every',type=int,default=1000)
    parser.add_argument('--depth-a',type=int)
    parser.add_argument('--depth-b',type=int)
    args=parser.parse_args()
    if args.hands<2 or args.iters<1 or not math.isfinite(args.budget) or args.budget<=0 or args.seed<0:
        parser.error('hands >=2, iters >0, finite budget >0 and seed >=0 required')
    if any(d is not None and not 0<=d<=6 for d in (args.depth_a,args.depth_b)):
        parser.error('depth overrides must be between 0 and 6')
    output=Path(args.out)
    if output.exists():
        parser.error('output already exists; choose a new result path')
    output.parent.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).resolve().parents[1]/'src'
    sources={name:hashlib.sha256((src/name).read_bytes()).hexdigest()
             for name in ('subgame/river_tree.py','subgame/river_subgame_solver.py',
                          'game/game_session.py','evaluation/aivat.py')}
    started=time.perf_counter()
    with closing(BlueprintDB(args.db,read_only=True)) as db:
        identity=policy_identity(args.db, session_kwargs(db)['abstraction_dir'])
        bots=[]
        frozen=FrozenBlueprint(db)
        for menu,depth in [(args.a,args.depth_a),(args.b,args.depth_b)]:
            bot=RiverSubgameSolver(db,river_menu=menu,max_aggressions=depth,
                                   max_iters=args.iters,check_every=40,time_budget=args.budget,
                                   safe_gadget=True,gadget_anchor='auto',purify_threshold=.01)
            bot.db=frozen
            bots.append(bot)
        records=[]
        checkpoint=output.with_suffix(output.suffix+'.partial')
        stats_complete=True
        if args.resume and checkpoint.exists():
            saved=load_checkpoint(checkpoint,vars(args),sources,identity)
            records=saved['records']
            stats_complete=saved.get('stats_complete','solver_stats' in saved)
            if 'solver_stats' in saved:
                for bot,stats in zip(bots,saved['solver_stats']):
                    bot.stats.update(stats)
        elif checkpoint.exists():
            parser.error('checkpoint exists; use --resume or choose another output')
        resumed_hands=len(records)
        def save_checkpoint():
            value=dict(config=vars(args),source_sha256=sources,policy_identity=identity,completed=len(records),
                       records=records,solver_stats=[b.stats for b in bots],
                       stats_complete=stats_complete,
                       backend=os.environ.get('ALLIN_RIVER_BACKEND','python'),
                       algorithm=os.environ.get('ALLIN_RIVER_ALGORITHM','cfrplus'))
            temporary=checkpoint.with_suffix(checkpoint.suffix+'.writing')
            temporary.write_text(json.dumps(value,allow_nan=False))
            temporary.replace(checkpoint)
        if resumed_hands:
            print(json.dumps(dict(event='resume',hands=resumed_hands,
                                  stats_complete=stats_complete)),flush=True)
        try:
            for i in range(resumed_hands,args.hands):
                records.append(play_hand(db,bots,args.seed,i))
                if args.progress_every>0 and (i+1)%args.progress_every==0:
                    save_checkpoint()
                    print(json.dumps(dict(progress=i+1,hands=args.hands,seed=args.seed,
                                          a=args.a,b=args.b,seconds=time.perf_counter()-started,
                                          solves=[b.stats.get('solved',0) for b in bots])),flush=True)
        except BaseException:
            save_checkpoint()
            raise
        estimate=AIVATEstimator(db,seed=args.seed).estimate(records)
        if not math.isfinite(estimate['var_reduction']):
            estimate['var_reduction']=None  # undefined when raw payoff variance is zero
        payload=dict(config=vars(args),backend=os.environ.get('ALLIN_RIVER_BACKEND','python'),
                     algorithm=os.environ.get('ALLIN_RIVER_ALGORITHM','cfrplus'),
                     seconds=time.perf_counter()-started,estimate=estimate,
                     resumed_hands=resumed_hands,stats_complete=stats_complete,
                     timing_scope='current_segment' if resumed_hands else 'whole_run',
                     source_sha256=sources,policy_identity=identity,
                     menus=[dict(name=b.river_menu,bets=b.menu,raises=b.raise_menu,
                                 depth=b.max_aggressions) for b in bots],
                     solver_stats=[bot.stats for bot in bots],records=records)
        temporary=output.with_suffix(output.suffix+'.writing')
        temporary.write_text(json.dumps(payload,allow_nan=False))
        temporary.replace(output)
        checkpoint.unlink(missing_ok=True)
        print(json.dumps({k:v for k,v in payload.items() if k not in ('records','policy_identity')}),flush=True)


if __name__=='__main__':
    main()
