#!/bin/bash
# Interleaved testspeed runs with opt.island_solve off/on; prints median ns per world-step.
# Usage: contrib/island_fastpath/ab_option.sh <label> <reps> <testspeed args...>
label=$1
reps=$2
shift 2
TS="$(dirname "$0")/../../.venv/bin/mjwarp-testspeed"
declare -A times
for rep in $(seq "$reps"); do
  for on in 0 1; do
    val=$([ "$on" = 1 ] && echo true || echo false)
    t=$("$TS" "$@" --override "opt.island_solve=$val" 2>&1 | grep "Total time per step" | awk '{print $5}' | tr -d ,)
    times[$on]+="$t "
  done
done
median() { tr ' ' '\n' | grep -v '^$' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
off=$(echo "${times[0]}" | median)
on=$(echo "${times[1]}" | median)
printf "%-22s off %12.0f ns  on %12.0f ns  ratio %5.2fx   (off: %s| on: %s)\n" \
  "$label" "$off" "$on" "$(echo "$off / $on" | bc -l)" "${times[0]}" "${times[1]}"
