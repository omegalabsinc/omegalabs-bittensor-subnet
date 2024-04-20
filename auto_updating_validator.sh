#!/bin/bash

VALIDATOR_ARGS=$@

# first, git pull
git pull

# next, set up environment
pip install -e .

# finally, run the validator
python neurons/validator.py $VALIDATOR_ARGS --neuron.auto_update
