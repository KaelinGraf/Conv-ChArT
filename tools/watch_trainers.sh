#!/bin/bash
# Trainer watchdog: detect a stalled run and the CPU-19 fault class that causes it.
#
# WHAT IT IS WATCHING FOR, and why this specific check. On 2026-07-29/30 four kernel
# general protection faults occurred, ALL on CPU 19, ALL inside PyTorch DataLoader
# processes (pt_data_worker x3, pt_data_pin x1). The kernel kills the faulting worker; a
# torch DataLoader cannot recover from a dead worker, so the run wedges forever with the
# main process spinning and metrics frozen. Trainers are now pinned off CPU 19
# (taskset -c 0-18,20-23); this watchdog verifies that fix holds and catches any
# recurrence on another core -- which would mean the fault is RAM, not one core.
#
# THREE INDEPENDENT SIGNALS, because each alone has a blind spot:
#   1. metrics.jsonl mtime  -- the definitive liveness test. A wedged run keeps its
#      process alive at high CPU, so `ps` alone never sees it.
#   2. zombie children      -- names the mechanism the instant it happens, before the
#      stall threshold trips.
#   3. new kernel Oops      -- names the CAUSE, with the CPU number, which decides
#      whether the taskset workaround is holding.
#
# Read-only: it reports, it does not restart. Automatic restart belongs to
# tools/resume_after_boot.sh (boot) and the main session (live), and a watchdog that
# also acts is a watchdog that can fight them.
PY=/home/kaelin/anaconda3/envs/MLWS/bin/python
REPO="/home/kaelin/p4p/dense deep charuco"
STALL_S=${STALL_S:-300}          # metrics silent this long = stalled (val passes take ~60-90 s)
POLL_S=${POLL_S:-60}
TEMP_WARN=${TEMP_WARN:-92}      # package C: log it
TEMP_CRIT=${TEMP_CRIT:-97}      # package C: act on it (crit is 105, instability well below)
cd "$REPO" || exit 1

# Kernel Oops lines already seen, so each is reported once. Seeded with the current count
# so a restart of this script does not re-report history.
SEEN_OOPS=$(journalctl -k -b 0 --no-pager 2>/dev/null | grep -c "general protection fault")
echo "[watch $(date -Is)] armed. stall>${STALL_S}s, poll ${POLL_S}s, temp warn/crit ${TEMP_WARN}/${TEMP_CRIT}C, ${SEEN_OOPS} pre-existing Oops this boot"

while :; do
  ALERT=""

  # --- 1. stalled metrics, per ACTIVE run dir
  # Keyed on the run dir, not on the process. tools/train_pair.py (2026-07-30) trains several
  # arms inside ONE process, so the old "one pid -> one run name from argv $7" mapping both
  # missed pair runs entirely and could not have named their arms. A run dir whose
  # metrics.jsonl moved recently HAS a live writer, whichever script that writer is -- which
  # is the property the check actually wants, and it needs no argv parsing at all.
  now=$(date +%s)
  for f in runs/*/metrics.jsonl; do
    [ -f "$f" ] || continue
    age=$(( now - $(stat -c %Y "$f") ))
    [ "$age" -gt 1800 ] && continue          # not an active run; finished or long dead
    if [ "$age" -gt "$STALL_S" ]; then
      name=$(basename "$(dirname "$f")")
      ALERT="${ALERT}STALLED: $name metrics ${age}s old\n"
    fi
  done

  # --- 2. zombie loader workers (the mechanism, visible before the stall threshold)
  # Only zombies owned by a LIVE trainer matter -- those are the ones blocking a
  # DataLoader. Orphaned zombies from already-killed runs linger unreaped (seen: pids
  # 27508 and 141978 reparented to 3010) and would otherwise alert forever.
  LIVE=$(ps -eo pid,args= | awk -v p="$PY" \
        '$2==p && ($3=="tools/train_detector.py" || $3=="tools/train_pair.py"){printf "%s ", $1}')
  Z=$(ps -eo ppid,pid,stat,comm --no-headers | awk -v live="$LIVE" \
        '$3 ~ /^Z/ && $4 ~ /pt_data/ && index(" "live, " "$1" ")>0 {print "  ppid "$1" pid "$2" "$4}')
  [ -n "$Z" ] && ALERT="${ALERT}ZOMBIE DataLoader worker(s):\n$Z\n"

  # --- 3. THERMAL. Added 2026-07-30 when Kaelin measured ~102C: this is the leading
  # indicator, and almost certainly the CAUSE of the CPU-19 faults rather than a
  # coincidence. cpu19 is core_id 35, measured at 94C with 716,637 throttle events while
  # the coolest cores sat at 78C. Package hit 96C against a 105C critical. Heat scales
  # with load, which is why the faults began immediately after workers went 6 -> 10.
  # Warn well below crit, because instability starts long before thermal shutdown.
  PKG=$(sensors 2>/dev/null | awk '/Package id 0:/{gsub(/[+C]/,"",$4); printf "%d", $4}')
  if [ -n "$PKG" ]; then
    if [ "$PKG" -ge "$TEMP_CRIT" ]; then
      HOT=$(sensors 2>/dev/null | awk '/^Core /{gsub(/[+C]/,"",$3); if ($3+0 >= 90) printf "%s%s=%dC ", $1, $2, $3}')
      ALERT="${ALERT}THERMAL: package ${PKG}C (>= ${TEMP_CRIT}C). Cores >=90C: ${HOT}\n"
      ALERT="${ALERT}  -> reduce --workers, or the fault class returns on whichever core is hottest.\n"
    elif [ "$PKG" -ge "$TEMP_WARN" ]; then
      ALERT="${ALERT}thermal warning: package ${PKG}C (>= ${TEMP_WARN}C)\n"
    fi
  fi

  # --- 4. new kernel Oops, with the CPU -- names the cause
  N=$(journalctl -k -b 0 --no-pager 2>/dev/null | grep -c "general protection fault")
  if [ "$N" -gt "$SEEN_OOPS" ]; then
    DET=$(journalctl -k -b 0 --no-pager 2>/dev/null | grep -A1 "general protection fault" \
          | grep "CPU:" | tail -n $(( N - SEEN_OOPS )) | sed 's/.*kernel: /  /')
    ALERT="${ALERT}NEW KERNEL OOPS ($(( N - SEEN_OOPS )) new):\n$DET\n"
    ALERT="${ALERT}  -> if CPU is 19, the taskset pin FAILED. If any OTHER cpu, the fault is\n"
    ALERT="${ALERT}     NOT core-localised and the suspect becomes RAM (this box has no ECC).\n"
    SEEN_OOPS=$N
  fi

  if [ -n "$ALERT" ]; then
    printf "\n[watch %s] === ALERT ===\n" "$(date -Is)"
    printf "$ALERT"
    printf "  live trainer pids: %s\n" "$LIVE"
    printf "  load: %s | RAM avail: %sGi\n" "$(cut -d' ' -f1-3 /proc/loadavg)" "$(free -g | awk 'NR==2{print $7}')"
  fi
  sleep "$POLL_S"
done
