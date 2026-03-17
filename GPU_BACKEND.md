# Jules GPU Computing Backend (wgpu Integration)

## Architecture Overview

### Current State
- GPU tensors cause panic in `TensorStorage::Gpu` branch
- Tree-walking interpreter doesn't leverage GPU at all

### Solution: wgpu Compute Shaders
- Cross-platform computation (DirectX, Vulkan, Metal)
- Rust-native API with excellent error handling
- Perfect for tensor operations

## Implementation Plan

### Phase 1: GPU Memory Management
```rust
// In interp.rs
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: HashMap<u64, wgpu::Buffer>,  // GPU buffer cache
}

pub struct GpuTensor {
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
    elem_type: ElemType,
}
```

### Phase 2: Common Kernels
- Matrix multiplication (matmul)
- Element-wise ops (+, -, *, /)
- Reductions (sum, mean, max)
- Activations (ReLU, Sigmoid, Softmax)
- Convolution (2D)

### Phase 3: Integration Points
1. When `Tensor::data == TensorStorage::Gpu`, dispatch to GPU
2. Return results to CPU for interpretation (initially)
3. Keep GPU buffer resident for next operation

## Key Benefits
- **10-100x speedup** for tensor operations
- **Parallelizes ML training** - critical for practical use
- **Scales to large models** - no single-thread bottleneck
