#!/usr/bin/env bash
# backend/bot/scripts/train_on_cloud.sh
#
# One-shot cloud training for the next blueprint. Bundles the three changes:
#   * LOSSLESS PREFLOP   -- already in code (card_abstractions.py, 169 fine buckets).
#   * gamma = 1          -- the trainer default now; passed explicitly below too.
#   * FINER POSTFLOP BUCKETS -- re-fit + re-baked HERE (flop/turn). This is the only
#     reason a re-bake is needed: changing bucket COUNTS. (Lossless preflop and gamma
#     do NOT touch the postflop tables.) River stays at its committed 10 buckets -- the
#     live river solver refines the river anyway, so it isn't a leak -- and isn't baked.
#
# Run on a FRESH Ubuntu box (as root) AFTER cloning the repo. See the README block the
# assistant gives you; in short:
#     nohup bash backend/bot/scripts/train_on_cloud.sh > ~/train.log 2>&1 &
#     tail -f ~/train.log
#
# Quick speed test first (recommended): set a tiny iteration count to gauge throughput:
#     ITERATIONS=200000 bash backend/bot/scripts/train_on_cloud.sh
#
# WHEN IT FINISHES it bundles EVERYTHING needed to serve into ~/result/ (one scp). You
# must ship FIVE same-generation artifacts -- blueprint + 2 centroids + 2 baked tables --
# NOT just the .db, or serving buckets hands wrong (silent) / the deploy 503s. The closing
# message spells out exactly what to commit vs upload to the release. It also runs BR/LBR
# every TRACK_EVERY iters and writes the curve to ~/result/progress.txt.
set -euo pipefail

# ---- knobs (override via env, e.g. ITERATIONS=200000 FLOP_BUCKETS=28 bash ...) -------
FLOP_BUCKETS=${FLOP_BUCKETS:-30}        # up from the committed 20
TURN_BUCKETS=${TURN_BUCKETS:-24}        # up from the committed 16
ITERATIONS=${ITERATIONS:-60000000}      # more buckets -> needs more iters to converge
WORKERS=${WORKERS:-$(nproc)}            # all cores on a dedicated training box
SITUATIONS=${SITUATIONS:-4000}          # centroid-fit sample size per street
TRACK_EVERY=${TRACK_EVERY:-7500000}     # train in chunks this size, BR/LBR after each
                                        # (snapshots kept for the post-run BR sweep too).
                                        # Set TRACK_EVERY=$ITERATIONS to train in one shot.
BR_SAMPLES=${BR_SAMPLES:-80}            # BR board samples per checkpoint (a trend, ~10min ea)
LBR_HANDS=${LBR_HANDS:-3000}            # LBR hands per checkpoint (0 = skip LBR)

# ---- system prereqs (fresh Ubuntu) ---------------------------------------------------
echo "==> [1/5] system packages"
apt-get update -qq
# build-essential + python3-dev are insurance: if a pinned wheel (e.g. phevaluator)
# lacks a cp312 build, pip falls back to compiling it -- without a C toolchain that
# fails mid-install on a fresh box (the "fail partway, waste money" case).
apt-get install -y -qq python3-venv python3-pip build-essential python3-dev

cd "$(dirname "$0")/.."                  # -> backend/bot
echo "==> [2/5] python venv + deps"
python3 -m venv .venv
set +u; source .venv/bin/activate; set -u   # activate may read unset vars under `set -u`
pip install --quiet --upgrade pip
pip install --quiet -r ../requirements.txt

# [3-4/5] Re-fit centroids + bake tables -- the slow fixed cost. Skipped if the baked
# turn table already exists (a re-run after a kill), unless REBAKE=force. The git-ignored
# tables are absent on a fresh clone, so a first run always bakes.
if [[ "${REBAKE:-auto}" == "force" || ! -f analysis/abstractions/postflop_table_turn.npz ]]; then
    echo "==> [3/5] re-fitting flop/turn centroids ($FLOP_BUCKETS / $TURN_BUCKETS buckets)"
    python scripts/compute_postflop_buckets.py --street flop --buckets "$FLOP_BUCKETS" --situations "$SITUATIONS"
    python scripts/compute_postflop_buckets.py --street turn --buckets "$TURN_BUCKETS" --situations "$SITUATIONS"
    echo "==> [4/5] baking flop/turn lookup tables (the TURN bake is the slow step)"
    python scripts/bake_postflop_table.py --street flop
    python scripts/bake_postflop_table.py --street turn
else
    echo "==> [3-4/5] baked tables already present -- skipping (set REBAKE=force to redo)"
fi

echo "==> [5/5] training $ITERATIONS iters in ${TRACK_EVERY}-iter chunks (BR/LBR each)"
mkdir -p analysis/blueprints/snapshots
PROGRESS="$HOME/progress.txt"
: > "$PROGRESS"
trained=0
DBFILE=""
while [ "$trained" -lt "$ITERATIONS" ]; do
    chunk=$(( (ITERATIONS - trained) < TRACK_EVERY ? (ITERATIONS - trained) : TRACK_EVERY ))
    if [ -z "$DBFILE" ]; then
        python tests/run_blueprint_trainer.py --iterations "$chunk" --workers "$WORKERS" \
            --menu-mode capped --gamma 1.0 --checkpoint-every 50000
        DBFILE=$(ls -1t analysis/blueprints/blueprint_par_*.db | head -1)
    else
        python tests/run_blueprint_trainer.py --iterations "$chunk" --workers "$WORKERS" \
            --menu-mode capped --gamma 1.0 --checkpoint-every 50000 \
            --resume "$(basename "$DBFILE")"
    fi
    trained=$((trained + chunk))
    snap="analysis/blueprints/snapshots/snap_${trained}.db"
    cp "$DBFILE" "$snap"
    echo "================ ${trained} iterations ================" | tee -a "$PROGRESS"
    # BR (in-abstraction exploitability, the convergence scoreboard -- watch for the
    # U-curve / a collapse: it should fall then flatten, NOT rise late).
    python tests/run_evaluation.py --db "$snap" --samples "$BR_SAMPLES" --workers "$WORKERS" \
        2>&1 | grep -E "Exploitability|BR as" | tee -a "$PROGRESS" || true
    # LBR (off-tree exploiter -- the practically-relevant robustness number).
    if [ "$LBR_HANDS" -gt 0 ]; then
        python tests/run_lbr.py --db "$snap" --hands "$LBR_HANDS" 2>&1 \
            | grep -iE "lbr|mbb" | tail -2 | tee -a "$PROGRESS" || true
    fi
done

# ---- bundle EVERY artifact needed to SERVE, into one folder for a single scp -------
echo "==> bundling result -> $HOME/result/"
mkdir -p "$HOME/result"
FINAL=$(ls -1t analysis/blueprints/blueprint_par_*.db | head -1)
cp "$FINAL" "$HOME/result/blueprint_final.db"     # RENAMED: prod loads this exact name
cp analysis/abstractions/postflop_centroids_flop.npz "$HOME/result/"
cp analysis/abstractions/postflop_centroids_turn.npz "$HOME/result/"
cp analysis/abstractions/postflop_table_flop.npz     "$HOME/result/"
cp analysis/abstractions/postflop_table_turn.npz     "$HOME/result/"
cp -r analysis/blueprints/snapshots "$HOME/result/snapshots"
cp "$PROGRESS" "$HOME/result/progress.txt"

cat <<'EOF'

==================== DONE -- READ THIS ====================
From your LAPTOP, pull the whole result folder down (one command):
    scp -r root@<box-ip>:~/result ./result

To SERVE this blueprint, all of these must be the SAME generation -- do NOT split them:
  1. COMMIT to git (binds the serving code to this abstraction):
       result/postflop_centroids_flop.npz -> backend/bot/analysis/abstractions/
       result/postflop_centroids_turn.npz -> backend/bot/analysis/abstractions/
  2. UPLOAD to the 'assets-v1' GitHub release (--clobber replaces the old 25M assets):
       gh release upload assets-v1 result/blueprint_final.db \
         result/postflop_table_flop.npz result/postflop_table_turn.npz --clobber
  3. Before pushing, sanity-check locally: put all 5 files in place and run
       cd backend/bot && python -m pytest tests/test_postflop_table_stamp.py -q
     then boot the API and confirm /api/healthz `iterations` matches your run.
WHY all five: the serving image ships PRE-BAKED tables (it does NOT re-bake) and loads
the blueprint by the exact name blueprint_final.db. Ship the .db without its matching
centroids+tables and the bot buckets every hand wrong (silent) or the deploy 503s.
==========================================================
Live BR/LBR curve across training: ~/result/progress.txt
EOF
