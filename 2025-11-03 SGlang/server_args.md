# SGlang Server Args

1 prefill, 12 decode setup

Notes

### --max-total-tokens: 

all tokens across all concurrent requests

### --mem-fraction-static

Allow how much memory for KV cache and model weights. If GPU memory is 40G, and you set 0.8. Then, you can only use 32G of memory to load model. 

If loading model hits OOM, increase this value.

### --cuda-graph-bs

Should you use this in prefill or decode? Don't use in prefill, b/c variable input size.

### --chunked-prefill-size

Smaller chunk avoids OOM, but slower. 

Prefill 64K. Decode 768K.

### --max-running-requests

### --max-prefill-tokens 



