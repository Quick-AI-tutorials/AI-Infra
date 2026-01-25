# Roofline analysis

Why: how to decide batch size for your GPU? 

References
- [Blackwell GPU Spec](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet?ncid=no-ncid). FP32, 80 TFLOPS. HBM3E, 8 TB/s
- Google ML Book: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/roofline/)
- Try RoCE on [NVIDIA NIXL](https://github.com/ai-dynamo/nixl)