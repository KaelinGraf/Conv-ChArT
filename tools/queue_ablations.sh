#!/bin/bash
# Durable ablation supervisor: walks runs/queue.txt, holding concurrency at MAX_MODELS.
#
# REWRITTEN 2026-07-30 (Kaelin: "continue to iterate through the whole ablation queue. Once
# that's done, use the results and creativity/intuition to come up with other ablations to do
# till I tell you to stop"). The previous version was a one-shot walker over a hardcoded bash
# array with a blocking sentinel wait for the composite launch -- fine for a fixed pre-Friday
# ladder, useless for an open-ended queue that gets APPENDED TO while it runs. Two changes
# carry all the difference:
#
#   1. The queue lives in runs/queue.txt and is RE-READ on every pass, so appending an arm to
#      that file is all it takes to extend the campaign. No edit to this script, no restart.
#   2. It launches PAIRS via tools/train_pair.py, which shares one DataLoader across arms.
#      Measured 147 -> 275 model-samples/s (1.86x) with GPU 54% -> 98%: at 2 arms the second
#      arm is very nearly free, because generation, not the GPU, was the binding cost.
#
# Everything the old version learned the hard way is kept verbatim below: exact-argv trainer
# counting, the cpu19 pin, the done-marker idempotence, the resume-file existence check, and
# the VRAM settle after launch. The old hardcoded QUEUE array and the composite sentinel wait
# are in git history (commit 3215049 and earlier) if the pre-Friday ordering is ever needed.
#
# RESTART THIS SUPERVISOR AFTER EDITING IT. The queue FILE is re-read every pass, but the
# SCRIPT is not: a running bash holds its loop in memory, so a live supervisor keeps executing
# whatever logic it started with. On 2026-08-01 that silently stalled the campaign -- the
# oversized-group rule below was added at 18:19 while a supervisor from 08:44 was still running
# the old `n_models + size > MAX` check, which rejects a 4-arm group even on an idle GPU. The
# GPU sat idle with the queue "healthy" (process alive, log clean, dry-run correct) because the
# dry-run reads the NEW script and the daemon runs the OLD one. Kill and relaunch after any edit.
#
# QUEUE FILE FORMAT (runs/queue.txt) -- '#' comments and blank lines ignored:
#     name:config:steps[:resume]                          one arm, via train_detector.py
#     name:config:steps[:resume] | name2:config2:steps2   two+ arms, via train_pair.py
# The '|' groups arms into ONE process sharing ONE loader. train_pair asserts the arms agree
# on every loader-determining key and dies if they do not, so a bad grouping fails loudly at
# launch rather than silently training on mismatched streams. sigma_hm IS such a key: arms at
# different sigma CANNOT be paired.
#
# STEP BUDGET IS NOT COSMETIC. cosine_lr anchors the anneal to --steps (dcc/trainutil.py:53),
# so a 50k-scheduled run killed at 35k sits at 36% of peak LR with its weights still bouncing,
# while a 35k-SCHEDULED run has annealed to the 3.0e-6 floor. Same wall-clock, materially
# better model. Match the budget to the comparator you intend to beat.
PY=/home/kaelin/anaconda3/envs/MLWS/bin/python
REPO="/home/kaelin/p4p/dense deep charuco"
QUEUE_FILE="${QUEUE_FILE:-runs/queue.txt}"
MAX_MODELS="${MAX_MODELS:-2}"   # MODELS, not processes -- a 2-arm train_pair counts as 2.
                                # 2 not 3: with the shared loader the GPU is now the binding
                                # resource (97-99% busy on a full-width pair), so a third arm
                                # splits the same throughput three ways instead of adding to
                                # it. Trios only pay when generation-bound again.
WORKERS="${WORKERS:-12}"
cd "$REPO" || exit 1

# COUNTING IS THE BUG-PRONE PART. `pgrep -f train_detector.py` and `grep -c` both match this
# script's OWN argv and any bash -c wrapper carrying the string, which has already produced a
# 47-minute self-deadlock and killed the session shell three times (and did so again on
# 2026-07-31 via `pkill -f "bash tools/queue_ablations.sh"`). Match argv[0]/argv[1] EXACTLY:
# the awk process has $1=="awk", so it cannot match itself.
#
# train_pair.py holds N models' worth of GPU, so it counts as N -- but ONLY for arms STILL
# TRAINING. Counting every --arm occurrence over-counts a partly-finished pair: on 2026-07-31
# A-CE finished at 50k while its gates-off partner had 29k steps to go, and the supervisor
# went on reserving 2 slots for a process training 1 model. That idles a slot for hours, and
# the cost is real -- one arm alone runs ~147 samp/s where a shared-loader pair runs ~275.
# An arm is discounted only on its own graceful `done:` marker, the same criterion is_done
# uses, so a merely-slow arm is never mistaken for a finished one.
n_models() {
  ps -eo args= | awk -v p="$PY" -v repo="$REPO" '
    $1==p && $2=="tools/train_detector.py" {n++}
    $1==p && $2=="tools/train_pair.py" {
      for (i=3; i<=NF; i++) if ($i=="--arm") {
        split($(i+1), f, ":")
        armlog = repo "/runs/" f[1] ".log"     # NOT `log` -- that is an awk built-in (natural
        done_marker = 0                        # logarithm) and assigning to it is a syntax
        while ((getline line < armlog) > 0)    # error that `bash -n` cannot see.
          if (line ~ /train_detector\] done/) done_marker = 1
        close(armlog)
        if (!done_marker) n++
      }
    }
    END {print n+0}'
}

# Is this run name being trained RIGHT NOW, by either trainer? Scans for the token after
# --name rather than assuming a fixed argv position, and matches "name:" anywhere in a pair's
# argv (its arms are --arm NAME:CONFIG:STEPS). Errs toward reporting running: a false "yes"
# delays an arm, a false "no" gives one metrics.jsonl two writers.
is_running() {
  ps -eo args= | awk -v p="$PY" -v n="$1" '
    $1==p && $2=="tools/train_detector.py" {for (i=3; i<NF; i++) if ($i=="--name" && $(i+1)==n) f=1}
    $1==p && $2=="tools/train_pair.py" && index($0, n ":") > 0 {f=1}
    END {exit !f}'
}

# Finished CLEANLY -- the graceful-completion marker, never "the process is gone". That
# distinction is load-bearing: on 2026-07-29 two arms were pre-empted and their truncated
# results were nearly reported as converged. train_pair writes the same marker into each
# arm's own log for exactly this check.
is_done() { grep -q "train_detector\] done" "runs/$1.log" 2>/dev/null; }

if [ ! -f "$QUEUE_FILE" ]; then echo "[queue $(date -Is)] no $QUEUE_FILE -- nothing to do"; exit 0; fi

# MUTUAL EXCLUSION with the boot-resume service, which resumes ANY incomplete arm and would
# race this script for the same free slot (two trainers, one run dir, one metrics.jsonl with
# two authors). At boot that service runs before any login, so this guard is what keeps a
# hand-started queue from fighting it.
if ps -eo args= | awk '$2 ~ /resume_after_boot\.sh/' | grep -q .; then
  echo "[queue $(date -Is)] ABORT: resume_after_boot.sh is running -- it owns launch decisions."
  exit 0
fi

echo "[queue $(date -Is)] supervising $QUEUE_FILE (max $MAX_MODELS concurrent models, $WORKERS workers)"
while :; do
  launched=0 pending=0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"                                   # strip trailing comments
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -z "$line" ] && continue

    # Split the line into arm groups on '|' and keep only the arms that still need training.
    # PER-ARM, not per-group: an earlier version skipped the WHOLE line as soon as any one arm
    # was done or running, which silently orphaned its partner -- e.g. the A-CE/gates-off pair,
    # where A-CE finishes ~29k steps before gates-off and would have taken the gates-off resume
    # down with it. Arms are dropped individually; whatever remains launches together.
    IFS='|' read -r -a groups <<< "$line"
    names=() cfgs=() steps_a=() resumes=() bad=0
    for g in "${groups[@]}"; do
      g="$(echo "$g" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
      IFS=: read -r nm cf st rs <<< "$g"
      st="${st:-50000}"
      if [ -z "$nm" ] || [ -z "$cf" ]; then echo "[queue $(date -Is)] BAD LINE: $line"; bad=1; break; fi
      if [ ! -f "$cf" ]; then echo "[queue $(date -Is)] SKIP $nm: config $cf missing"; continue; fi
      if is_done "$nm"; then continue; fi                        # already finished
      if is_running "$nm"; then pending=1; continue; fi          # in flight; come back later
      # A resume entry must have its checkpoint, or we would silently restart from step 0.
      # Not yet written (an arm that has not reached its first val) is a WAIT, not a skip:
      # restarting such an arm from 0 is exactly the silent data loss the check exists to stop.
      if [ -n "$rs" ] && [ ! -f "$rs" ]; then
        echo "[queue $(date -Is)] WAIT $nm: resume checkpoint $rs not written yet"; pending=1; continue
      fi
      names+=("$nm"); cfgs+=("$cf"); steps_a+=("$st"); resumes+=("$rs")
    done
    [ "$bad" -eq 1 ] && continue
    [ "${#names[@]}" -eq 0 ] && continue

    # DRY_RUN=1 prints the launch decisions and exits without touching the GPU. Worth its three
    # lines: this queue gets APPENDED TO repeatedly, and a mistyped config path or an illegal
    # pairing is far cheaper to find here than four hours into a run.
    if [ -n "$DRY_RUN" ]; then
      echo "[queue dry-run] WOULD LAUNCH ${#names[@]} arm(s): ${names[*]}"
      for i in "${!names[@]}"; do
        echo "                  ${names[$i]}  cfg=${cfgs[$i]}  steps=${steps_a[$i]}${resumes[$i]:+  resume=${resumes[$i]}}"
      done
      continue
    fi

    # Slot check counts THIS group's arms, so a pair waits for two free slots.
    #
    # STRICT ORDER: if the head-of-queue line does not fit, STOP SCANNING -- do not let a later,
    # smaller line jump into the slot. This was `continue` until 2026-07-31, and the greedy
    # version reordered the campaign the first time it mattered: with one slot free it skipped
    # four pair-lines (each needing two) and launched the SOLO arm from the last batch, which
    # would then still have been running when the second slot freed -- delaying the highest
    # priority work by a full 35k-step arm. A briefly idle slot is cheaper than silently
    # reordering a queue whose order encodes what the user asked for first.
    # A group LARGER than MAX_MODELS is legal, but only onto an idle GPU. Rationale: at small
    # widths the shared loader, not the GPU, is the binding cost -- the 222k pair ran at 42-53%
    # GPU while a full-width pair sits at 97%. Four small arms on ONE loader therefore cost
    # barely more wall-clock than two, whereas two separate 2-arm lines would mean two loader
    # pools and the CPU starvation this design exists to avoid (147 vs 275 model-samples/s
    # measured). So: never start a big group alongside anything, but do let it have the machine.
    if [ "$(n_models)" -gt 0 ] && [ $(( $(n_models) + ${#names[@]} )) -gt "$MAX_MODELS" ]; then
      pending=1; break
    fi

    # CPU PIN: keep every trainer and its loader workers off cpu19. All four kernel GPFs on
    # 2026-07-29/30 landed on that core (core_id 35, 94C, 716,637 throttle events) inside
    # DataLoader processes; a killed worker wedges the run permanently. Since PL1 was dropped
    # 250W -> 125W the package sits at 79-81C, so this is belt-and-braces rather than the fix
    # -- but it is free: at a binding power cap throughput is set by watts, not core count
    # (measured: 134 samp/s on 14 cores AND on 23). Children inherit the mask.
    if [ "${#names[@]}" -eq 1 ]; then
      RES=(); [ -n "${resumes[0]}" ] && RES=(--resume "${resumes[0]}")
      echo "[queue $(date -Is)] launching ${names[0]} (${cfgs[0]}, ${steps_a[0]} steps)${resumes[0]:+ RESUMING from ${resumes[0]}}"
      env PYTHONPATH= taskset -c 0-18,20-23 "$PY" tools/train_detector.py --config "${cfgs[0]}" \
          --name "${names[0]}" --steps "${steps_a[0]}" --workers "$WORKERS" "${RES[@]}" \
          >> "runs/${names[0]}.log" 2>&1 &
    else
      ARMS=()
      for i in "${!names[@]}"; do
        ARMS+=(--arm "${names[$i]}:${cfgs[$i]}:${steps_a[$i]}${resumes[$i]:+:${resumes[$i]}}")
      done
      echo "[queue $(date -Is)] launching PAIR (${#names[@]} arms, shared loader): ${names[*]}"
      env PYTHONPATH= taskset -c 0-18,20-23 "$PY" tools/train_pair.py "${ARMS[@]}" \
          --workers "$WORKERS" >> "runs/pair_${names[0]}.log" 2>&1 &
    fi
    echo "[queue $(date -Is)]   pid $!"
    launched=1
    sleep 180        # let it claim VRAM before the next slot check
    break            # re-read the queue file from the top: it may have been appended to
  done < "$QUEUE_FILE"

  [ -n "$DRY_RUN" ] && exit 0                            # one pass, no waiting
  [ "$launched" -eq 1 ] && continue
  if [ "$pending" -eq 1 ]; then sleep 120; continue; fi   # arms left, but no free slot yet
  echo "[queue $(date -Is)] queue drained -- every entry is done, running, or skipped."
  exit 0
done
