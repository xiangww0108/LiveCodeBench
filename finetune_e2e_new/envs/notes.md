in qwen25_requirements env, run:
    train, inference_intrinsic, inference_extrinsic, eval_intrinsic
in lcb_requirements env, run:
    eval_extrinsic 
    (via  ```
    python -m lcb_runner.runner.custom_evaluator \
    --custom_output_file /home/ubuntu/finetune_e2e_new/data/Qwen2-1_5B_repaired_extrinsic.json \
    --scenario codegeneration \
    --release_version release_v6 \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --num_process_evaluate 8
    ```)
