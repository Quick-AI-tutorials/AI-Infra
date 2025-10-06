# Build your own Transformer with FlashInfer

![Flash Attention Thumbnail](./Thumbnail%20Flash%20Attention.png)

Find a Pytorch Docker Image. E.g. Vast.ai: https://hub.docker.com/r/vastai/pytorch/.

Rent a A100 GPU.

### Install FlashInfer
```
git clone https://github.com/faradawn/assignment2-systems.git
cd assignment2-systems

conda create -n flashinfer python=3.10 -y
conda activate flashinfer

pip install flashinfer-python

pip install einx jaxtyping

python -m cs336_systems.benchmarking_script --max_seq_len 512 --batch_size 64
```
