#!/bin/bash

set -e

MODEL="gpt-5-mini"
FIXED_ARGS="--k 15 --top-m 5 --alpha 0.8 --theta 0.6 --xi-llm 0.95 --L 20 --U 1000"

echo "=========================================="
echo "Running experiments (main.py)"
echo "Model: $MODEL"
echo "Fixed args: $FIXED_ARGS"
echo "=========================================="

echo "[1/8] Running Census (Budget 0.5)..."
python main.py --dataset census --budget 0.5 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "[2/8] Running Cora (Budget 0.5)..."
python main.py --dataset cora --budget 0.5 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "[3/8] Running AS [using 'affiliation'] (Budget 1)..."
python main.py --dataset affiliation --budget 1 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "[4/8] Running Amazon-GP [using 'amazonGp'] (Budget 1.5)..."
python main.py --dataset amazonGp --budget 1.5 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "[5/8] Running Song (Budget 1)..."
python main.py --dataset song --budget 1 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "[6/8] Running Alaska [using 'sigmod'] (Budget 3)..."
python main.py --dataset sigmod --budget 3 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "[7/8] Running Music [using 'music20K'] (Budget 2)..."
python main.py --dataset music20K --budget 2 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "[8/8] Running Movies (Budget 2)..."
python main.py --dataset movies --budget 2 --model $MODEL $FIXED_ARGS
echo "------------------------------------------"

echo "All experiments completed!"