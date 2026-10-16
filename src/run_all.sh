#!/usr/bin/env bash
# run_all.sh - reproduce every result in the report, in order.  Resumable: a finished run
# (results/<tag>_result.json exists) is skipped, so if it is interrupted just start it again.
#
#   cd "Assignment 1"
#   <activate your Python environment>
#   bash src/run_all.sh                # everything
#   bash src/run_all.sh grid           # or one stage: prep | grid | seeds | part2 | eval
#   SMOKE=1 bash src/run_all.sh        # 1 epoch per run: quick end-to-end test of the whole pipeline
#
# Before the FIRST run on a new split, archive old outputs (never deletes anything):
#   mv results results_old && mv data/processed data/processed_old
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"   # keep the Mac awake
mkdir -p results/logs
STAGE=${1:-all}
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# train <tag> <train.py args...>
train() {
  local tag=$1; shift
  if [ -f "results/${tag}_result.json" ]; then echo "[skip] $tag"; return; fi
  echo "[run ] $tag  ($(date +%H:%M:%S))"
  $CAF $PY src/train.py "$@" ${SMOKE:+--epochs 1} --tag "$tag" 2>&1 | tee "results/logs/${tag}.log" | grep -E "params=|early stop|test_ppl|train_time_s" || true
  [ -f "results/${tag}_result.json" ] || { echo "FAILED: $tag (see results/logs/${tag}.log)"; exit 1; }
}
epochs_for() { [ "$1" = "transformer" ] && echo 250 || echo 100; }
stag() { [ "$2" = "42" ] && echo "$1" || echo "$1_s$2"; }          # seed 42 -> no suffix

if want prep; then
  echo "== data + n-gram =="
  $PY src/data.py 2>&1 | tee results/logs/data_stats.txt
  [ -f results/ngram_result.json ] || $PY src/ngram.py 2>&1 | tee results/logs/ngram.txt
fi

if want grid; then
  echo "== Part I: learning-rate grid (seed 42), chosen on validation perplexity =="
  for m in rnn lstm transformer; do
    for lr in 1e-3 3e-3; do
      train "${m}_lr${lr}" --model $m --lr $lr --epochs "$(epochs_for $m)" --patience 8 --seed 42
    done
  done
  $PY src/select_best.py 2>&1 | tee results/logs/select_best.txt
fi

if want seeds; then
  echo "== Part I: extra seeds for the chosen configuration of each architecture =="
  for m in rnn lstm transformer; do
    tag=$($PY src/select_best.py --get $m); lr=$($PY src/select_best.py --lr $m)
    for s in 43 44; do
      train "${tag}_s${s}" --model $m --lr $lr --epochs "$(epochs_for $m)" --patience 8 --seed $s
    done
  done
fi

if want part2; then
  echo "== Part II: embeddings (LSTM, same lr / budget as the chosen Part I LSTM) =="
  [ -f results/sgns_cost.json ] || $PY src/embeddings.py sgns 2>&1 | tee results/logs/embeddings_sgns.txt
  [ -f results/glove_coverage.json ] || $PY src/embeddings.py glove 2>&1 | tee results/logs/embeddings_glove.txt
  lr=$($PY src/select_best.py --lr lstm)
  for s in 42 43 44; do
    for v in sgns glove; do
      W="data/processed/emb_${v}100.npy"
      train "$(stag lstm_${v}_frozen $s)" --model lstm --lr $lr --epochs 100 --patience 8 --seed $s --emb_weights $W --freeze_emb
      train "$(stag lstm_${v}_init $s)"   --model lstm --lr $lr --epochs 100 --patience 8 --seed $s --emb_weights $W
    done
  done
fi

if want eval; then
  echo "== checks, downstream task, generation, benchmarks =="
  $PY src/extra_checks.py tests    2>&1 | tee results/logs/tests.txt
  $PY src/extra_checks.py params   2>&1 | tee results/logs/params.txt
  $PY src/extra_checks.py position 2>&1 | tee results/logs/position.txt
  $PY src/classify.py              2>&1 | tee results/logs/classify.txt
  $PY src/generate.py --set part1  2>&1 | tee results/logs/generate_part1.txt
  $PY src/generate.py --set part2  2>&1 | tee results/logs/generate_part2.txt
  $PY src/benchmark.py --device cpu 2>&1 | tee results/logs/benchmark_cpu.txt
  $PY src/benchmark.py --device mps 2>&1 | tee results/logs/benchmark_mps.txt || echo "(no MPS device - skipped)"
fi
echo "== done: stage '$STAGE' finished after ${SECONDS}s =="
