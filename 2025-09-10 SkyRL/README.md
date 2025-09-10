# SkyRL: how to install and train your LLM with GRPO on GSM8k

### Part 1: Reserve GPU and Install the Library
Basic Docker command (single GPU)
```
docker run -it --runtime=nvidia --gpus all --shm-size=8g --name skyrl-train erictang000/skyrl-train-ray-2.48.0-py3.12-cu12.8 /bin/bash
```

Advanced Docker command (multiple GPU's, remote VLLM inference)
```
docker run -it --runtime=nvidia --gpus all --shm-size=8g --name skyrl-train -p 8265:8265 -p 10001:10001 -p 8000:8000 -p 9000:9000 -p 8001:8001 erictang000/skyrl-train-ray-2.48.0-py3.12-cu12.8 /bin/bash
```


### Part 2:

### Part 3:
