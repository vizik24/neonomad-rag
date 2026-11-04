#!/usr/bin/env bash
# Run a handful of questions through the full query pipeline (retrieve + Claude) and save the output.
cd "$(dirname "$0")/.." || exit 1

questions=(
  "Can I fit an e-bike in the BikerBed?"
  "Does the BikerBed fit a Renault Trafic?"
  "What size Allen key do I need?"
  "Is a stove included with the kitchen?"
  "Will fitting a BikerBed let me register my van as a motorhome?"
  "Do you deliver to France?"
)

for q in "${questions[@]}"; do
  echo "=================================================="
  echo "Q: $q"
  .venv/bin/python query.py "$q" 2>/dev/null
done 2>&1 | tee smoke_test.log
echo "SMOKE TEST DONE"
