# Tutorial to Use LMCache within Nvidia Dynamo

![thumbnai](./thumbnail.png)

### Quick start
```
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

docker compose -f deploy/docker-compose.yml --profile metrics up -d

# Build the container
./container/build.sh --framework vllm
./container/run.sh --framework vllm -it --mount-workspace

# Install nvshem
pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com \
    nvidia-nvshmem-cu12 nvshmem4py-cu12

# serve 
python -m dynamo.frontend --http-port=8000 &

# run worker with LMCache enabled
ENABLE_LMCACHE=1 \
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=20 \
  python -m dynamo.vllm --model Qwen/Qwen3-0.6B

# Test a prompt
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "What is the capital of US"
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
  
# Run the script
python cpu-offloading.py
python cpu-offloading.py --enable-lmcache
```

