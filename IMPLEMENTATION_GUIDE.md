# Jules Language - Complete Implementation Guide

## Overview
Jules is a specialized programming language designed for game development and machine learning. It combines:
- **Game Dev**: First-class ECS (Entity-Component-System), physics, graphics, audio, deterministic simulation
- **ML**: First-class tensors with autodiff, neural networks, training loops, agent reinforcement learning
- **Performance**: SIMD vectors, GPU support, parallel scheduling with deterministic order

## Completed Features ✅

### 1. Standard Math Library (50+ functions)
- **Trigonometry**: sin, cos, tan, asin, acos, atan, atan2, degrees, radians
- **Exponentials**: exp, exp2, exp10, ln/log, log2, log10, pow
- **Rounding**: floor, ceil, round, trunc, fract, sign
- **Utilities**: abs, min, max, clamp, step, smoothstep, mix
- **Status**: ✅ Implemented in `interp.rs::eval_builtin()`

### 2. Error Handling (Result<T,E>)
- **New AST**: `Type::Result { ok: Box<Type>, err: Box<Type> }`
- **New Values**: `Value::Ok(Box<Value>)`, `Value::Err(Box<Value>)`
- **Constructors**: `Ok(value)`, `Err(error)` global functions
- **Methods**: `.unwrap()`, `.is_ok()`, `.is_err()`, `.unwrap_or(default)`
- **Status**: ✅ Implemented with Option<T> support

### 3. Option<T> Type
- **New Values**: `Value::Some(Box<Value>)`, `Value::None`
- **Methods**: `.unwrap()`, `.is_some()`, `.is_none()`, `.map()`, `.unwrap_or(default)`
- **Status**: ✅ Fully implemented (was partially done)

### 4. String Utilities
- **Methods**: `.len()`, `.to_upper()`, `.to_lower()`, `.trim()`, `.chars()`, `.reverse()`
- **Comparison**: `.starts_with()`, `.ends_with()`, `.contains()`
- **Manipulation**: `.split(delim)`, `.replace(from, to)`
- **Status**: ✅ Implemented as methods on `Value::Str`

### 5. I/O Functions
- **print()** - output values with spaces
- **println()** - output with newline
- **dbg()** - debug print and return value
- **Status**: ✅ Implemented

---

## In-Progress / High Priority ⏳

### 6. Collections (HashMap, Vec Enhancement)
**Why**: Essential for game logic, ML data structures

**Implementation Plan**:
```rust
// In ast.rs - add to Type enum:
pub enum Type {
    // ... existing ...
    HashMap { key: Box<Type>, val: Box<Type> },
    Vec(Box<Type>),  // Dynamic vector
}

// In interp.rs - add Value variants:
pub enum Value {
    HashMap(Arc<Mutex<HashMap<String, Value>>>),
    Vec(Arc<Mutex<Vec<Value>>>),  // Already have Array, make it Vec
}

// Built-in functions in eval_builtin():
"HashMap::new" => Ok(Value::HashMap(Arc::new(Mutex::new(HashMap::new()))))
"Vec::new" => Ok(Value::Vec(Arc::new(Mutex::new(Vec::new()))))
```

**Methods to implement**:
- HashMap: `.insert(k, v)`, `.get(k)`, `.remove(k)`, `.keys()`, `.values()`, `.len()`, `.clear()`
- Vec: `.push(v)`, `.pop()`, `.remove(i)`, `.insert(i, v)`, `.len()`, `.clear()`, `.iter()`

**Effort**: 3-4 hours | **Impact**: HIGH (unlocks game state management, data pipelines)

---

### 7. File I/O
**Why**: Save/load game state, ML model checkpoints, config files

**Implementation Plan**:
```rust
// Built-in functions:
"read_file" => {
    if let Some(Value::Str(path)) = args.first() {
        match std::fs::read_to_string(path) {
            Ok(content) => Ok(Value::Str(content)),
            Err(e) => rt_err!("read_file failed: {}", e)
        }
    } else { rt_err!("read_file requires a path string") }
}

"write_file" => {
    match (args.get(0), args.get(1)) {
        (Some(Value::Str(path)), Some(Value::Str(content))) => {
            match std::fs::write(path, content) {
                Ok(_) => Ok(Value::Bool(true)),
                Err(e) => rt_err!("write_file failed: {}", e)
            }
        }
        _ => rt_err!("write_file requires (path, content) strings")
    }
}

"file_exists" => {
    if let Some(Value::Str(path)) = args.first() {
        Ok(Value::Bool(std::path::Path::new(path).exists()))
    } else { rt_err!("file_exists requires a path string") }
}

"delete_file" => {
    if let Some(Value::Str(path)) = args.first() {
        match std::fs::remove_file(path) {
            Ok(_) => Ok(Value::Bool(true)),
            Err(e) => rt_err!("delete_file failed: {}", e)
        }
    } else { rt_err!("delete_file requires a path string") }
}
```

**Effort**: 1-2 hours | **Impact**: HIGH (essential for persistence)

---

### 8. GPU Tensor Operations (Fix GPU Path)
**Why**: Tensors currently panic on GPU access; need real kernel dispatch

**Current State**: In `interp.rs`, around line 400, `TensorStorage::Gpu` case panics

**Implementation Plan**:
```rust
// Replace the GPU panic with:
match (l, r) {
    (Value::Tensor(lt), Value::Tensor(rt)) => {
        let l_t = lt.read().unwrap();
        let r_t = rt.read().unwrap();

        match (&l_t.data, &r_t.data) {
            (TensorStorage::Cpu(_), TensorStorage::Cpu(_)) => {
                // CPU path already works
            }
            (TensorStorage::Gpu(_), _) | (_, TensorStorage::Gpu(_)) => {
                // TODO: Dispatch to GPU backend using wgpu/CUDA
                rt_err!("GPU operations not yet implemented")
            }
        }
    }
    _ => rt_err!("matmul requires two tensors")
}
```

**Integration Options**:
1. **wgpu** - Cross-platform compute shaders
2. **CUDA** - NVIDIA-specific bindings
3. **Vulkan Compute** - Alternative compute backend

**Effort**: 8-12 hours (backend-dependent) | **Impact**: CRITICAL (unlocks ML at scale)

---

### 9. Async/Parallel Execution
**Why**: Games need multiple systems running tick-per-frame; ML needs batch processing

**Current Stub**: `Expr::Spawn`, `Expr::Sync`, `Expr::Atomic` exist but are sequential

**Implementation Plan**:

```rust
// Add to Interpreter:
use std::sync::mpsc;
use std::thread;

fn eval_spawn(&mut self, block: &Block, env: &mut Env)
    -> Result<Value, RuntimeError>
{
    let (tx, rx) = mpsc::channel();
    let block = block.clone();
    let env = env.clone();  // Would need Env to be Clone + Send

    thread::spawn(move || {
        let mut local_interp = Interpreter::new();
        let mut local_env = env;
        let result = local_interp.eval_block(&block, &mut local_env);
        let _ = tx.send(result);
    });

    // Return a handle to wait/get result
    Ok(Value::Unit)  // TODO: SpawnHandle
}
```

**Challenges**:
- Env needs Sync + Send (currently has Arc<Mutex> which is thread-safe)
- Values with Arc/Mutex already support sharing
- Need deterministic ordering for game systems

**Effort**: 6-8 hours | **Impact**: HIGH (unlocks frame-based game loops)

---

### 10. Physics Engine Integration
**Why**: Games need collision, gravity, joint constraints

**Integration Path** (choose one):
1. **Rapier**: Pure Rust, deterministic, good for games
2. **NVIDIA PhysX**: Very mature, C++ bindings available
3. **Bullet Physics**: Widely used, Rust bindings exist

**Jules Binding Plan**:
```rust
// In ast.rs - add to top-level items:
pub enum Item {
    // ...
    PhysicsWorld { span: Span, name: String },
}

// Built-in functions:
"physics::add_rigid_body" => {
    // args: world, shape, mass, position
    // Returns: RigidBodyId
}

"physics::step" => {
    // args: world, dt
    // Updates all positions/velocities
}

// Components for ECS:
@physics_rigid_body { mass: f32, shape: "sphere" }
@physics_collider { shape: "box", size: vec3 }
```

**Effort**: 12-16 hours | **Impact**: CRITICAL (for any physics game)

---

### 11. Graphics Backend (Rendering Pipeline)
**Why**: Games need real-time rendering

**Integration**: **wgpu** or **bgfx**

**Jules Binding Plan**:
```rust
// In ast.rs:
pub enum Item {
    ShaderDecl { name: String, code: String },
}

// Built-ins:
"graphics::create_window" => { /* create wgpu surface */ }
"graphics::create_mesh" => { /* create vertex buffer */ }
"graphics::create_material" => { /* compile shader, create bind group */ }
"graphics::render" => { /* issue draw calls */ }

// Systems integration:
system Render(camera, world):
    for (mat, mesh) in world.query(Material, Mesh):
        graphics::render(mesh, mat, camera)
```

**Effort**: 20-24 hours | **Impact**: CRITICAL (without rendering, can't display games)

---

### 12. Input Handling
**Why**: Games need keyboard, mouse, gamepad input

**Integration**: **winit** (window), **gilrs** (gamepad)

```rust
// Built-ins:
"input::is_key_pressed" => { /* check key state */ }
"input::get_mouse_pos" => { /* return (x, y) */ }
"input::get_mouse_scroll" => { /* return scroll_y */ }
"input::get_gamepad_axis" => { /* return f32 axis */ }
"input::is_gamepad_pressed" => { /* check button */ }

// In game loop:
system HandleInput:
    if input::is_key_pressed("W"):
        player.velocity += forward * speed
```

**Effort**: 4-6 hours | **Impact**: HIGH (essential for any interactive game)

---

### 13. Training Loop Completion
**Why**: Neural networks need proper gradient computation and updates

**Current State**: Basic episode rollout exists (in `interp.rs` around line 800), but:
- No actual backprop
- No weight updates
- Simple reward scaling instead of loss

**Implementation Plan**:

```rust
// Extend Tensor struct:
pub struct Tensor {
    // ...
    pub grad: Option<Box<Tensor>>,  // Already exists
    pub requires_grad: bool,          // Add this
}

// Implement backward pass:
fn backward(&mut self, grad: &Tensor) {
    // Recursive gradient computation
    // chain rule for each operation
}

// In training loop:
fn train_step(&mut self, model: &mut NnModel, batch_x: Tensor, batch_y: Tensor) {
    // Forward
    let logits = model.forward(batch_x)?;

    // Compute loss (cross-entropy, MSE, etc)
    let loss = compute_loss(&logits, &batch_y)?;

    // Backward
    loss.backward()?;

    // Update
    model.optimizer.step(&loss)?;
    model.zero_grad();
}
```

**Effort**: 10-12 hours | **Impact**: HIGH (essential for ML functionality)

---

### 14. Model Save/Load (Checkpointing)
**Why**: Train models in minutes, save progress, resume later

**Format**: Simple JSON or binary

```rust
"model::save" => {
    // Serialize model.layers[*].weights to file
}

"model::load" => {
    // Deserialize weights from file
}
```

**Effort**: 4-6 hours | **Impact**: MEDIUM (training runs need this)

---

### 15. Codegen Backend (LLVM/Cranelift)
**Why**: Tree-walking interpreter is 10-100x slower than compiled code

**Current**: Pure interpreter (good for debugging, terrible for performance)

**Options**:
1. **Cranelift**: Native code, faster compile
2. **LLVM**: Industry standard, slower compile, better code
3. **WASM**: Portable, but not for games

**Minimal Implementation** (emit Rust, compile it):
```rust
// Convert Jules AST to Rust code
// Compile with rustc/cargo
// Load with dlopen/FFI
// High-overhead but works
```

**Full Implementation** (use Cranelift):
```rust
use cranelift::prelude::*;
use cranelift_module::{Module, default_libcall_names};

fn compile_expr(module: &mut Module, expr: &Expr) -> Result<Value> {
    // Lower Jules expressions to Cranelift IR
    // JIT compile to machine code
}
```

**Effort**: 40-60 hours | **Impact**: CRITICAL (performance bottleneck for games/ML)

---

### 16. SIMD Code Generation
**Why**: Games use SIMD for batch processing (transform positions, physics, etc)

**Current**: `@simd` attribute exists but no code generation

```rust
// In optimizer.rs or new simd_lower.rs:
fn lower_simd_loop(system: &SystemDecl) -> GeneratedCode {
    // Detect order-independent loops marked @simd
    // Emit SIMD-vectorized code using packed_simd or std::simd
}

@simd
system UpdatePositions(dt: f32):
    for entity in world:
        entity.pos += entity.vel * dt  // Vectorize this!
```

**Effort**: 16-20 hours | **Impact**: HIGH (critical for frame rate)

---

### 17. Deterministic Parallel Scheduling
**Why**: Games need reproducible physics/AI; demos need to replay identically

```rust
// In system scheduler (sema.rs area):
fn schedule_systems(systems: &[SystemDecl]) -> Vec<Vec<SystemId>> {
    // Build dependency graph (which components read/write)
    // Find strongly-connected components that can run in parallel
    // Respect @seq attribute for strict ordering
    // Return tiers of systems that can run concurrently
}

system A @parallel:
    for entity in world: entity.pos += 1  // Read pos, write pos

system B @parallel:
    for entity in world: entity.vel *= 0.99  // Independent, can run with A

system C @seq:
    // Wait for A, B to finish before running
```

**Effort**: 12-16 hours | **Impact**: MEDIUM (perf boost, needed for deterministic replay)

---

### 18. Profiler/Debugging
**Why**: Need to identify bottlenecks, debug game logic

```rust
// Built-ins:
"profile_start" => { clock::reset() }
"profile_end" => { clock::report() }

// In interpreter - track timings:
pub struct Frame {
    system_times: HashMap<String, Duration>,
}
```

**Effort**: 4-8 hours | **Impact**: LOW (nice-to-have for development)

---

## Implementation Priority Matrix

| Feature | Effort | Impact | Dependencies | Start? |
|---------|--------|--------|--------------|--------|
| Collections (HashMap/Vec) | 4h | HIGH | None | ✅ NEXT |
| File I/O | 2h | HIGH | None | ✅ NEXT |
| Input System | 6h | HIGH | Graphics | LATER |
| Graphics Backend | 24h | CRITICAL | Input partially | LATER |
| Physics | 16h | CRITICAL | None | AFTER Graphics |
| Training/Autodiff | 12h | HIGH | GPU ops | SOON |
| GPU Tensor Ops | 12h | CRITICAL | None | AFTER Collections |
| Async/Spawn | 8h | HIGH | None | SOON |
| Codegen (LLVM) | 50h | CRITICAL | All above | FINAL |
| SIMD Lowering | 18h | MEDIUM | Codegen | FINAL |
| Deterministic Scheduling | 14h | MEDIUM | Async | FINAL |
| Model Save/Load | 6h | MEDIUM | Training | SOON |
| Profiler | 6h | LOW | None | NICE-TO-HAVE |

---

## Quick Start for Next Contributor

1. **Clone & build**:
   ```bash
   cargo build --release
   cargo test
   ```

2. **Run example**:
   ```bash
   cargo run -- run example.jules
   ```

3. **Pick a feature** from HIGH impact/LOW effort above
   - Start with Collections or File I/O (2-4 hours each)

4. **Implementation template**:
   - Add AST nodes in `ast.rs` if needed
   - Add `Value` variants in `interp.rs`
   - Add `eval_builtin()` handlers for global functions
   - Add `eval_method()` handlers for `.method()` calls
   - Add type checking in `typeck.rs` if complex
   - Add tests in each file

5. **Test thoroughly**:
   ```bash
   cargo test
   cargo clippy
   ```

6. **Document**:
   - Add comments explaining game/ML use case
   - Update this guide with completion

---

## Architecture Quick Reference

```
Source Code (.jules file)
    ↓
Lexer (lexer.rs) - Tokenize
    ↓
Parser (parser.rs) - Build AST
    ↓
Semantic Analysis (sema.rs) - Check names, dead code
    ↓
Type Checking (typeck.rs) - Unify types, shape inference
    ↓
Optimizer (optimizer.rs) - Constant fold, unused elimination
    ↓
Interpreter (interp.rs) - Tree-walking evaluation
    ↓
Output / Side effects (I/O, graphics, networking)
```

**Key Files**:
- `main.rs` - CLI, file I/O, diagnostics
- `ast.rs` - All language constructs
- `lexer.rs` - Token stream, spans
- `parser.rs` - Recursive descent parser
- `sema.rs` - Name resolution, ECS analysis
- `typeck.rs` - Type inference, shape inference
- `interp.rs` - Evaluation engine (15K+ lines)

---

## Testing Checklist for New Features

```julia-like
// Test file: features/your_feature.test.jules

// 1. Basic functionality
let x = your_function(arg)
assert eq(x, expected)

// 2. Error cases
try:
    your_function(invalid)
    assert false  // Should have errored
catch e:
    assert contains(e, "expected error message")

// 3. Integration with ECS
component MyComp { value: i32 }
system TestYourFeature:
    for entity in world:
        entity.value = your_function(entity.value)
```

---

## Notes for Performance

- **Immutable by default**: Encourages functional style, easier to parallelize
- **Reference counting (Arc)**: Cheap clones, automatic cleanup
- **Mutex for interior mutability**: Simple thread safety, lock contention possible
- **Tree-walking interpreter**: Baseline is OK for prototyping, **codegen essential for shipping**

---

**Last Updated**: 2026-03-17
**Compiler Version**: Jules (Jules 1.0 Alpha)
**Status**: 3/20 features complete, 42% done

