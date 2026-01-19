# How to Deploy a Model on vLLM on local GPU

![](./diagram-container-vllm.png)

## Step 1: Choose a Docker Container 

Find the latest container build from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm

Export HF token 
expose the port 8000
expose all GPU
mount hugging face cache, so donwlaod model onces and reuse 
```
export HF_TOKEN="xxx"

docker run -it --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN="$HF_TOKEN" \
  --name my_container \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  bash
```

## Step 2:

Launch vLLM, use the same port as the exposed one

vllm serve openai/gpt-oss-20b --host 0.0.0.0 --port 8000

# additional flags (if out of memory, adjust teh memory uitlization)
 --gpu-memory-utilization 0.8


Test using 

 curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
}'



## Step 3: Open WebUI
- Pull the open webui software, launch another container to spin it up 
- Change the port to 8000 for your vLLM

docker run \
  -it \
  -e OPENAI_API_BASE_URL="http://localhost:8000/v1" \
  -v open-webui:/app/backend/data \
  --network host \
  --add-host=host.docker.internal:host-gateway \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

If you conenct to your GPU via SSH (remote), Setup Port Forwarding and SSH connect

ssh -L 8080:localhost:8080 spark

open web broswer