### Goal

1. Load CIFAR-10
2. build a small CNN
3. train with augmentations
4. add validation, mixed precision (AMP), and checkpointing
5. Sanity check by overfitting a tiny subset

**🧠 1. Learning Rate Scheduler: Cosine Decay with Warmup**

- Warmup: Start with a low learning rate for a few steps → helps stabilize training at the beginning.
- Cosine decay: Gradually reduce the learning rate in a smooth cosine curve → better convergence compared to constant or step decay.
  📦 PyTorch tools:
- torch.optim.lr_scheduler.CosineAnnealingLR handles the cosine decay.
- Manual warmup can be applied before handing control to the cosine scheduler.
  ✅ Benefit: Helps model train smoother and reach higher accuracy, especially on small models.

**🪄 2. Better Augmentation**

- ColorJitter: Randomly changes brightness, contrast, saturation, hue.
- RandomErasing: Randomly erases a rectangle region in the image.
- AutoAugment: A learned set of augmentations from literature.
  ✅ Benefit: Improves generalization by making training data more diverse.

**🧽 3. Regularization**

- Dropout: Randomly drops neurons during training to prevent overfitting.
  → nn.Dropout(p=0.3) before the linear layer.
- Label smoothing: Slightly softens the target distribution to make the model less confident and more robust.
  ✅ Benefit: Better generalization, reduced overfitting.

**⚡ 4. Speed Boost with torch.compile**

- `torch.compile` (PyTorch 2+) optimizes the computation graph under the hood.
  Typically gives 1.2x–2x speedup without changing model code.
  ✅ Benefit: Faster training & inference.

**📤 5. Export to ONNX**
ONNX is an open format to export models for inference in other frameworks (like TensorRT, OpenVINO, or ONNX Runtime).
`torch.onnx.export` lets you save a trained PyTorch model into tinycnn.onnx.
✅ Benefit: Easy deployment to other platforms.
