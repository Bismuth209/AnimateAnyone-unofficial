#!/bin/bash

max_workers=10

for ((phase=0; phase<max_workers; phase++)); do
    PHASE=$phase NPHASES=$max_workers python prepare_ubc.py &
done

# Wait for all background jobs to finish
wait
