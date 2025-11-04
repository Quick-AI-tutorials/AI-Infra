# SGlang Server Args

1 prefill, 12 decode setup

Prefill
```
	export SGLANG_TORCH_PROFILER_DIR=${WORKSPACE}/../prefill_torch_profiler
	mkdir -p ${SGLANG_TORCH_PROFILER_DIR}	
	export SGLANG_DUMPER_DIR=${WORKSPACE}/../prefill_sglang_dump
	mkdir -p ${SGLANG_DUMPER_DIR}
	export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${WORKSPACE}/../prefill_sglang_expert_distribution_recorder
	mkdir -p ${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}
	export FLASHINFER_WORKSPACE_BASE=${WORKSPACE}/../prefill_flashinfer_workspace_base
	mkdir -p ${FLASHINFER_WORKSPACE_BASE}
	export SGL_DG_CACHE_DIR=${WORKSPACE}/../prefill_deepgemm_cache
	mkdir -p ${SGL_DG_CACHE_DIR}

	export SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1
	export SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1
	# export FLASHINFER_WORKSPACE_BASE=/data/numa0/tom/flashinfer_workspace_base
	export SGL_JIT_DEEPGEMM_PRECOMPILE=0
	export MC_TE_METRIC=true
	export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
	export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
	export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
	export SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1
	export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
	# export SGLANG_LOCAL_IP_NIC=eth0
	# export SGLANG_DUMPER_DIR=/data/numa0/tom/temp
	# export SGLANG_TORCH_PROFILER_DIR=/data/numa0/tom/temp_sglang_server2local
	# export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/data/numa0/tom/temp_sglang_server2local
	# export GLOO_SOCKET_IFNAME=eth0
	# export NCCL_SOCKET_IFNAME=eth0
	# export NCCL_MNNVL_ENABLE=1
	# export NCCL_CUMEM_ENABLE=1
	export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
	export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
	export PYTHONUNBUFFERED=1
	# export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1

	python3 -m sglang.launch_server \
		--disaggregation-mode prefill \
		--host 0.0.0.0 \
		--decode-log-interval 1 \
		--max-running-requests 5632 \
		--context-length 2176 \
		--disable-radix-cache \
		--disable-shared-experts-fusion \
		--watchdog-timeout 1000000 \
		--disable-chunked-prefix-cache \
		--attention-backend trtllm_mla \
		--kv-cache-dtype fp8_e4m3 \
		--enable-single-batch-overlap \
		--tp-size 4 \
		--dp-size 4 \
		--ep-size 4 \
		--enable-dp-attention \
		--moe-dense-tp-size 1 \
		--enable-dp-lm-head \
		--chunked-prefill-size 65536 \
		--model-path nvidia/DeepSeek-R1-0528-FP4 \
		--trust-remote-code \
		--disable-cuda-graph \
		--port 30000 \
		--mem-fraction-static 0.84 \
		--max-total-tokens 131072 \
		--max-prefill-tokens 16384 \
		--load-balance-method round_robin \
		--quantization modelopt_fp4 \
		--moe-runner-backend flashinfer_cutlass \
		--disaggregation-bootstrap-port 8991 \
		2>&1 | tee ${WORKSPACE}/../logs/launch_server_prefill_node_rank_${NODE_RANK}.log

```

Decode
```
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

	export SGLANG_TORCH_PROFILER_DIR=${WORKSPACE}/../decode_torch_profiler
	mkdir -p ${SGLANG_TORCH_PROFILER_DIR}	
	export SGLANG_DUMPER_DIR=${WORKSPACE}/../decode_sglang_dump
	mkdir -p ${SGLANG_DUMPER_DIR}
	export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${WORKSPACE}/../decode_sglang_expert_distribution_recorder
	mkdir -p ${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}
	export FLASHINFER_WORKSPACE_BASE=${WORKSPACE}/../decode_flashinfer_workspace_base
	mkdir -p ${FLASHINFER_WORKSPACE_BASE}
	export SGL_DG_CACHE_DIR=${WORKSPACE}/../decode_deepgemm_cache
	mkdir -p ${SGL_DG_CACHE_DIR}

	export SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1
	export SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1
	# export FLASHINFER_WORKSPACE_BASE=/data/numa0/tom/flashinfer_workspace_base
	export SGL_JIT_DEEPGEMM_PRECOMPILE=0
	export MC_TE_METRIC=true
	export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
	export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
	export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
	export SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1
	export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
	# export SGLANG_LOCAL_IP_NIC=eth0
	# export SGLANG_DUMPER_DIR=/data/numa0/tom/temp
	# export SGLANG_TORCH_PROFILER_DIR=/data/numa0/tom/temp_sglang_server2local
	# export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/data/numa0/tom/temp_sglang_server2local
	# export GLOO_SOCKET_IFNAME=eth0
	# export NCCL_SOCKET_IFNAME=eth0
	# export NCCL_MNNVL_ENABLE=1
	# export NCCL_CUMEM_ENABLE=1
	export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
	export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
	export PYTHONUNBUFFERED=1
	export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
	export SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1
	export SGLANG_FP4_GEMM_BACKEND=cutlass

	python3 -m sglang.launch_server \
		--dist-timeout 3600 \
		--disaggregation-mode decode \
		--host 0.0.0.0 \
		--decode-log-interval 1 \
		--max-running-requests 49152 \
		--context-length 2176 \
		--disable-radix-cache \
		--disable-shared-experts-fusion \
		--watchdog-timeout 1000000 \
		--disable-chunked-prefix-cache \
		--attention-backend trtllm_mla \
		--kv-cache-dtype fp8_e4m3 \
		--dist-init-addr $DECODE_HEAD_NODE_IP:5757 \
		--nnodes 12 \
		--node-rank $NODE_RANK \
		--enable-single-batch-overlap \
		--model-path nvidia/DeepSeek-R1-0528-FP4 \
		--trust-remote-code \
		--tp-size 48 \
		--dp-size 48 \
		--ep-size 48 \
		--enable-dp-attention \
		--chunked-prefill-size 786432 \
		--mem-fraction-static 0.83 \
		--moe-a2a-backend deepep \
		--deepep-mode low_latency \
		--ep-dispatch-algorithm static \
		--cuda-graph-bs 1024 \
		--num-reserved-decode-tokens 112 \
		--ep-num-redundant-experts 32 \
		--moe-dense-tp-size 1 \
		--enable-dp-lm-head \
		--prefill-round-robin-balance \
		--max-total-tokens 3122380 \
		--quantization modelopt_fp4 \
		--moe-runner-backend flashinfer_cutedsl \
		2>&1 | tee ${WORKSPACE}/../logs/launch_server_decode_node_rank_${NODE_RANK}.log | grep --line-buffered 'DP0'
```
