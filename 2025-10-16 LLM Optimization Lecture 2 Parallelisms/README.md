# Steps to use tensor parallelism for sglang

```
neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 (118G)
Qwen/Qwen3-Next-80B-A3B-Thinking (80 x 2 = 160G)

export LOG_DIR="./logs"

nohup python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-20b \
  --disable-radix-cache \
  --trust-remote-code \
  --enable-dp-attention \
  --mem-fraction-static 0.8 \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
	--context-length 2176 \
	--max-total-tokens 2200 \
  > ${LOG_DIR:-.}/sglang-no-parallel.log 2>&1 &
tail -f ${LOG_DIR:-.}/sglang-no-parallel.log

pkill -f sglang.launch_server

ps aux | grep sglang.launch_server

squeue -u $USER
srun --jobid=1016159 --overlap --container-name=sglang --pty bash


python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 256 --random-range-ratio 1 --num-prompts 100


Output token throughput (tok/s):         1468.41   

# with tp=4
nohup python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-20b \
  --disable-radix-cache \
  --trust-remote-code \
  --enable-dp-attention \
  --mem-fraction-static 0.8 \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
	--context-length 2176 \
	--max-total-tokens 2200 \
	--tp=4 \
  > ${LOG_DIR:-.}/sglang-tp-4.log 2>&1 &
tail -f ${LOG_DIR:-.}/sglang-tp-4.log

pkill -f sglang.launch_server


Total token throughput (tok/s):          1858.28   


# with ep=4
nohup python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-20b \
  --disable-radix-cache \
  --trust-remote-code \
  --enable-dp-attention \
  --mem-fraction-static 0.8 \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
	--context-length 2176 \
	--max-total-tokens 2200 \
	--tp=4 \
	--ep=4 \
  > ${LOG_DIR:-.}/sglang-ep-4.log 2>&1 &
tail -f ${LOG_DIR:-.}/sglang-ep-4.log

pkill -f sglang.launch_server

Total token throughput (tok/s):          1837.45   


# with dp4 (cannot do dp only) so try dp and tp
nohup python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-20b \
  --disable-radix-cache \
  --trust-remote-code \
  --enable-dp-attention \
  --mem-fraction-static 0.8 \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
	--context-length 2176 \
	--max-total-tokens 2200 \
	--dp=4 \
  > ${LOG_DIR:-.}/sglang-dp-4.log 2>&1 &
tail -f ${LOG_DIR:-.}/sglang-dp-4.log

pkill -f sglang.launch_server

--tp 8 --enable-dp-attention



# with dp + tp (cannot do dp only) so try dp and tp
nohup python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-20b \
  --disable-radix-cache \
  --trust-remote-code \
  --enable-dp-attention \
  --mem-fraction-static 0.8 \
  --max-running-requests 512 \
  --chunked-prefill-size 4096 \
	--context-length 2176 \
	--max-total-tokens 2200 \
	--tp=4 \
	--enable-dp-attention \
  > ${LOG_DIR:-.}/sglang-dp-4.log 2>&1 &
tail -f ${LOG_DIR:-.}/sglang-dp-4.log

pkill -f sglang.launch_server


Total token throughput (tok/s):          1891.02   

```
