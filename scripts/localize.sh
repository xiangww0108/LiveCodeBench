#!/bin/bash

python src/data_prep/localize_bug_spans.py \
  --in output/localizer-data-Qwen3/small.json \
  --out output/localizer-data-Qwen3/out_small_gpt-4o-mini.json \
  --model gpt-4o-mini