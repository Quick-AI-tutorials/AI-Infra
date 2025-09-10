set -x
NUM_GPUS=1
INFERENCE_BACKEND="vllm"   # or "sglang" (not really used since epochs=0)

CKPT="$HOME/ckpts/gsm8k_1.5B_ckpt/global_step_30"

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  trainer.resume_mode=from_path \
  trainer.resume_path="$CKPT" \
  trainer.epochs=0 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.hf_save_interval=1 \
  trainer.export_path="$HOME/exports" \
  trainer.logger=console
# HF export will land at: $HOME/exports/global_step_30/policy
