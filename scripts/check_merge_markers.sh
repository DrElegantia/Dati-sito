#!/usr/bin/env bash
set -euo pipefail

status=0
git grep -nE '^(<<<<<<<|=======|>>>>>>>)' -- . > /tmp/merge_markers.txt || status=$?

if [ "$status" -eq 0 ]; then
  echo "Merge conflict markers found in tracked files:"
  cat /tmp/merge_markers.txt
  exit 1
elif [ "$status" -eq 1 ]; then
  echo "No merge conflict markers found in tracked files."
  exit 0
else
  echo "git grep failed with status $status"
  exit "$status"
fi
