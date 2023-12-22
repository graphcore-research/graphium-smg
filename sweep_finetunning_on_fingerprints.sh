#!/bin/bash

# Constants
ENTITY='graphcore'
PROJECT='scaling_mol_gnns'

# Set environment variables
export SWEEP_DATASET='bbb-martins'
export SWEEP_FINGERPRINTS_PATH='ids_to_fingerprint.pt'
export SWEEP_MODEL_SIZE='10M'

# Run wandb sweep and capture both stdout and stderr
SWEEP_OUTPUT=$(wandb sweep --project "$PROJECT" finetune_on_fingerprints.yaml 2>&1)
echo "Sweep command output:"
echo "$SWEEP_OUTPUT"

# Extract the sweep ID using grep and cut
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep 'Created sweep with ID:' | cut -d ' ' -f 6)

# Debug: Echo the captured sweep ID
echo "Captured Sweep ID: $SWEEP_ID"

# Check if the sweep ID was captured
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to capture the Sweep ID."
    exit 1
fi

# Use the captured sweep ID to start an agent
wandb agent "$ENTITY/$PROJECT/$SWEEP_ID"
