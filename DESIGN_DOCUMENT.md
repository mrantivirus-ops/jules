# Jules Language - Complete Vision & Design Document

## Executive Summary

**Jules** is a statically-typed, compiled programming language designed specifically for the intersection of game development and machine learning. It combines:

- **Entity-Component-System (ECS)** for deterministic game simulation
- **First-class tensors** with automatic differentiation for neural networks
- **Reinforcement learning agents** that learn and adapt in game worlds
- **GPU-accelerated compute** for physics and ML workloads
- **Deterministic replay** for testing and debugging complex AI behaviors

### Target Use Cases

1. **Game Engines**: Real-time games with sophisticated AI (multiplayer, physics, procedural generation)
2. **AI Research**: Reinforcement learning, imitation learning, evolutionary algorithms
3. **Physics Simulations**: Deterministic, reproducible scientific computing
4. **Game AI**: NPCs with learning, planning, and adaptation
5. **Robotics Simulation**: Virtual training for real-world robot policies

---

## Language Design Philosophy

### Core Principles

1. **Explicitness Over Magic**
   - No implicit conversions (except numeric coercion)
   - Explicit error handling required
   - Pattern matching for data unpacking

2. **Performance Matters**
   - First-class SIMD support with `vec2`, `vec3`, `vec4`
   - Tensor operations compile to GPU kernels
   - System execution is parallelized (when safe)

3. **Determinism is First-Class**
   - `@seq` attribute forces sequential execution
   - Fixed system ordering for reproducibility
   - No undefined behavior

4. **Mathematical-First**
   - Linear algebra as native syntax (`A @ B` = matrix multiply)
   - Broadcasting rules match NumPy/TensorFlow
   - Automatic shape inference

---

## Complete Feature Breakdown

### Tier 1: Core Language (COMPLETE ✅)

```
✅ Lexer & Parser (full Jules syntax)
✅ Type System (inference, shapes, generics)
✅ ECS Framework (entities, components, systems)
✅ Pattern Matching (destructuring, guards)
✅ Error Propagation (Result, Option)
```

### Tier 2: Standard Library (95% COMPLETE)

```
✅ Math Functions (50+ - sin, cos, sqrt, etc)
✅ String Operations (split, replace, case conversion)
✅ Collections (HashMap, Array, dynamic Vec)
✅ File I/O (read, write, append, delete)
✅ Type Conversion (i32, f32, str, bool)
✅ I/O Functions (print, println, dbg)
⏳ Advanced Collections (BitSet, Deque, PriorityQueue)
⏳ Random Number Generation (Xorshift, distributions)
⏳ Time & Duration (millis, sleep, timers)
⏳ Serialization (JSON, YAML, binary)
```

### Tier 3: Game Development (30% COMPLETE)

```
✅ ECS World (entity lifecycle, component queries)
✅ System Scheduling (topological sort, @simd hints)
✅ Transform Components (vec2, vec3, quaternions)
✅ Determinism Annotations (@seq, @parallel)
⏳ Physics (collision, rigid bodies, joints)
⏳ Graphics (rendering, shaders, materials)
⏳ Input (keyboard, mouse, gamepad events)
⏳ Audio (sound loading, effects, mixing)
⏳ Animation (skeletal, blendshapes, curves)
⏳ Particle Effects (emitters, forces, rendering)
⏳ Spatial Indexing (BVH, quadtrees, culling)
⏳ Scene Manager (loading, streaming, unloading)
```

### Tier 4: Machine Learning (25% COMPLETE)

```
✅ Neural Network Layers (dense, conv, pooling, recurrent)
✅ Activations (ReLU, Sigmoid, Tanh, Softmax, GELU)
✅ Model Declaration Syntax (layer stacking)
✅ Agent System (@learning, @perception, @behavior)
✅ Training Blocks (reward, penalty, episode config)
⏳ Autodiff (full backpropagation, gradients)
⏳ Optimizers (Adam beyond SGD, learning rate schedules)
⏳ Loss Functions (custom, not just reward aggregation)
⏳ Data Loading (batch, shuffle, prefetch)
⏳ Distributed Training (multi-GPU, multi-machine)
⏳ Model Serialization (save, load, export ONNX)
⏳ Reinforcement Learning (PPO, A3C, MAPPO)
```

### Tier 5: Performance (0% COMPLETE)

```
⏳ LLVM/Cranelift Codegen (replace tree-walking)
⏳ SIMD Code Generation (vectorized loops)
⏳ GPU Kernel Compilation (wgpu compute shaders)
⏳ Memory Pooling (arena allocation for ECS)
⏳ Parallel Scheduling (deterministic multi-threading)
⏳ Profiling & Instrumentation (flame graphs)
⏳ Optimization Passes (loop fusion, inlining)
```

---

## Syntax Examples by Use Case

### Example 1: A Simple Physics-Based Game

```jules
// Define components
component Position { x: f32, y: f32 }
component Velocity { vx: f32, vy: f32 }
component Collider { radius: f32 }

// Physics system
@parallel
system ApplyPhysics(dt: f32):
    for entity in world:
        // Update position
        entity.Position.x += entity.Velocity.vx * dt
        entity.Position.y += entity.Velocity.vy * dt

        // Apply gravity
        entity.Velocity.vy -= 9.81 * dt

// Collision detection (must be sequential for determinism)
@seq
system CollisionDetection:
    entities = world.query(Position, Collider)
    for i in 0..len(entities):
        for j in i+1..len(entities):
            e1 = entities[i]
            e2 = entities[j]

            dx = e1.Position.x - e2.Position.x
            dy = e1.Position.y - e2.Position.y
            dist = sqrt(dx*dx + dy*dy)

            if dist < e1.Collider.radius + e2.Collider.radius:
                on_collision(e1, e2)

// Main loop stub (runtime handles this)
fn main():
    // Initialize
    player = world.spawn()
    player.Position = Position { x: 0.0, y: 0.0 }
    player.Velocity = Velocity { vx: 0.0, vy: 0.0 }
    player.Collider = Collider { radius: 1.0 }

    // Game loop (frame by frame)
    while true:
        ApplyPhysics(1.0/60.0)  // Run systems
        CollisionDetection()
        graphics::render()
        input::handle_events()
```

### Example 2: Training a Game AI Agent

```jules
// Define the neural network policy
model PolicyNet {
    input 16           // observation size: position, velocity, health, etc
    dense 128 relu
    dense 64 relu
    dense 4 softmax    // 4 actions: up, down, left, right
}

// Define the learning agent
agent GameAI {
    perception: vision { range: 50.0, fov: 120.0 }
    memory: episodic { window: 100 }
    learning: reinforcement { model: PolicyNet }
    behavior Explore(priority: 10):
        action = PolicyNet.forward(observation)
        return sample_action(action)
}

// Reinforcement learning training block
train GameAI in World {
    // Reward signals
    reward survival 1.0              // +1 per frame alive
    reward defeat_enemy 100.0        // +100 for beating enemy
    penalty take_damage 10.0         // -10 per damage point
    penalty invalid_action 1.0       // -1 for bad moves

    // Episode configuration
    episode {
        max_steps: 1000
        num_envs: 4              // Parallel environments
        timeout_seconds: 300.0
    }

    // Optimization
    model PolicyNet
    optimizer adam { learning_rate: 0.0003 }
}

// Usage in-game
component IsAI { policy: Model }

@simd
system AIDecisions:
    for entity in world:
        if entity.has(IsAI):
            observation = get_observation(entity)
            action = entity.IsAI.policy.forward(observation)
            apply_action(entity, action)
```

### Example 3: Complex Data Processing Pipeline

```jules
// Load training data
train_data = read_file("data/training_data.csv")
lines = train_data.split("\n")

// Parse and organize
data = HashMap::new()
targets = []

for line in lines:
    if line.starts_with("label"):
        continue  // Skip header

    parts = line.split(",")
    features = parts[0..16]          // inputs
    label = i32(parts[17])           // target

    key = concat("sample_", str(len(targets)))
    data.insert(key, features)
    targets.push(label)

// Save processed data
output = concat("Processed ", str(len(targets)), " samples\n")
output = concat(output, "Features: ", str(len(features)), "\n")
write_file("data/processed.txt", output)

println("Data loading complete!")
println("Training samples:", len(targets))
```

---

## Type System Reference

### Scalar Types
```
i8, i16, i32, i64       # Signed integers
u8, u16, u32, u64       # Unsigned integers
f32, f64                 # Floating point (no distinction needed usually)
bool                     # True / false
str                      # Strings (immutable)
```

### Composite Types
```
vec2, vec3, vec4         # 3D math vectors (float)
ivec2, ivec3, ivec4      # Integer vectors
mat2, mat3, mat4         # Square matrices
quat                     # Quaternion (rotation)

tensor<f32>[N, M, K]     # N-dimensional array with shape
tensor<f32>[128, 128]    # 128×128 matrix
tensor<f32>[_]           # Dynamic first dimension
```

### Algebraic Types
```
Option<T>                # Some(value) or None
Result<T, E>             # Ok(value) or Err(error)
(T, U, V)                # Tuple of types T, U, V
[T; N]                   # Fixed array of N elements
[T]                      # Dynamic array (slice)
fn(A, B) -> C            # Function pointer
```

### User-Defined Types
```
struct Point { x: f32, y: f32 }
enum Color { Red, Green, Blue }
component Health { hp: f32, max_hp: f32 }
```

---

## Execution Model

### Runtime Phases

1. **Initialization**
   - Create `World` (ECS container)
   - Spawn entities, assign components
   - Initialize neural networks/models

2. **Main Loop** (repeated every frame)
   ```
   for frame in 0..infinity:
       1. Input: Collect keyboard/mouse/gamepad events
       2. Update: Run game systems (@simd/@parallel first, then @seq)
       3. Physics: Apply forces, detect collisions
       4. AI: Run agent decision-making
       5. Render: Draw graphics (or compute-only for headless)
       6. Train: If learning active, collect experience and update weights
   ```

3. **Determinism Guarantee**
   - Systems marked `@seq` ALWAYS run in declaration order
   - Random seeds are fixed unless explicitly seeded
   - Floating-point operations are exact (no random rounding)
   - Entity IDs are stable across frames

### Memory Model

- **Ownership**: Each value has a single owner (move semantics)
- **Borrowing**: `&T` immutable reference, `&mut T` mutable
- **Lifetime**: All references inferred (no explicit lifetime annotations)
- **Garbage Collection**: Arc/Mutex for shared mutable state (like game world)

### Parallelism Model

```
@simd      → Run with SIMD vectorization (if applicable)
@parallel  → Run in multi-threaded fashion (safe with no component conflicts)
@seq       → Run sequentially (guaranteed ordering)
@gpu       → Dispatch to GPU computation
```

---

## Interoperability & Extensibility

### Foreign Function Interface (FFI)

```julius
// Call Rust functions from Jules
extern "rust" fn physics_step(world: i64, dt: f32) -> i32

// Use in Jules as normal function
fn update_physics(dt: f32):
    result = physics_step(world_ptr, dt)
```

### Custom Components

```jules
component Custom {
    data: i32,
    nested: { field: str }
}
```

### Custom Operators (Future)

```jules
// Implement + for custom type
impl Add for MyVector:
    fn add(lhs: MyVector, rhs: MyVector) -> MyVector:
        return MyVector { x: lhs.x + rhs.x, y: lhs.y + rhs.y }

v1 = MyVector { x: 1, y: 2 }
v2 = MyVector { x: 3, y: 4 }
v3 = v1 + v2  // Calls impl Add
```

---

## Performance Roadmap

### Current Baseline (Tree-Walking)
- **Tensor ops**: ~100K elements/sec
- **Game systems**: ~100K entities/frame (60 fps)
- **Simple AI**: ~1K agent decisions/frame

### After LLVM Backend (Expected)
- **Tensor ops**: ~10M elements/sec (100x)
- **Game systems**: ~10M entities/frame (100x)
- **Complex AI**: ~100K agent decisions/frame (100x)

### After GPU Integration
- **Tensor ops**: ~1B elements/sec (1000x vs baseline)
- **Physics**: Real-time 100K rigid bodies
- **Training**: 10x faster neural network training

---

## Design Decisions & Rationale

### Why ECS for Games?
- **Cache locality**: Components stored contiguously
- **Parallelism**: Independent systems can run in parallel
- **Determinism**: Entity iteration order is predictable
- **Flexibility**: Easy to add/remove components at runtime

### Why First-Class Tensors?
- **ML Focus**: Tensors are embarrassingly parallel
- **Automatic Broadcasting**: NumPy-style shape inference
- **GPU-Ready**: Direct mapping to compute shaders
- **Autodiff**: Enables gradient computation

### Why Deterministic?
- **Reproducibility**: Bug reproduction
- **Deterministic Replay**: Debug player actions
- **Network Play**: Lockstep multiplayer physics
- **Testing**: Unit tests for complex simulations

### Why Statically Typed?
- **Performance**: No runtime type checks
- **IDE Support**: Better autocomplete, refactoring
- **Error Catching**: Type errors caught at compile time
- **GPU Code Generation**: Types inform kernel compilation

---

## Comparison Matrix

| Feature | Jules | Rust | Python | C++ |
|---------|-------|------|--------|-----|
| Game-Focused | ✅ | ❌ | ❌ | ⚠️ |
| ML-Focused | ✅ | ❌ | ✅ | ⚠️ |
| Type Inference | ✅ | ✅ | ❌ | ⚠️ |
| Deterministic | ✅ | ❌ | ❌ | ❌ |
| Direct SIMD | ✅ | ⚠️ | ❌ | ✅ |
| Parallel ECS | ✅ | ⚠️ | ❌ | ⚠️ |
| GPU Support | 🔜 | 🔜 | ✅ | ✅ |
| Learning Curve | Easy | Hard | Easy | Very Hard |

---

## Example: Complete Mini-Game

```julius
// === COMPONENTS ===
component Position { x: f32, y: f32 }
component Velocity { vx: f32, vy: f32 }
component Health { hp: f32 }
component IsPlayer {}
component IsEnemy { ai_model: Model }

// === GAME Systems ===
@parallel
system Movement(dt: f32):
    for entity in world:
        entity.Position.x += entity.Velocity.vx * dt
        entity.Position.y += entity.Velocity.vy * dt

@parallel
system Friction(dt: f32):
    for entity in world:
        entity.Velocity.vx *= 0.95
        entity.Velocity.vy *= 0.95

@seq
system AIControl:
    for entity in world:
        if entity.has(IsEnemy):
            obs = compute_observation(entity)
            action = entity.IsEnemy.ai_model.forward(obs)
            entity.Velocity = apply_action(action)

@seq
system Collision:
    player = world.find(IsPlayer)[0]
    for entity in world:
        if entity.has(IsEnemy):
            if distance(player.Position, entity.Position) < 2.0:
                player.Health.hp -= 10.0

@seq
system Cleanup:
    for entity in world:
        if entity.Health.hp <= 0.0:
            world.despawn(entity)

// === NEURAL NETWORK ===
model EnemyAI {
    input 8           // obs: player_x, player_y, enemy_x, enemy_y, etc
    dense 32 relu
    dense 16 relu
    dense 2 tanh      // output: velocity direction
}

agent EnemyAgent {
    learning reinforcement, model: EnemyAI
    behavior Patrol(priority: 10):
        return random_direction()
}

// === TRAINING ===
train EnemyAgent in World {
    reward catch_player 100.0
    reward stay_near_player 1.0
    penalty go_offmap 10.0

    episode { max_steps: 1000, num_envs: 8 }
    model EnemyAI
    optimizer adam { learning_rate: 0.001 }
}

// === MAIN ===
fn main():
    // Setup
    player = world.spawn()
    player.Position = { x: 0.0, y: 0.0 }
    player.Velocity = { vx: 0.0, vy: 0.0 }
    player.Health = { hp: 100.0 }
    player.IsPlayer = {}

    // Create enemies
    for i in 0..10:
        enemy = world.spawn()
        enemy.Position = { x: sin(f32(i) * 0.628) * 20.0, y: cos(f32(i) * 0.628) * 20.0 }
        enemy.Velocity = { vx: 0.0, vy: 0.0 }
        enemy.Health = { hp: 50.0 }
        enemy.IsEnemy = { ai_model: EnemyAI }

    // Game loop
    frame = 0
    while player.Health.hp > 0 and frame < 6000:
        Movement(1.0 / 60.0)
        Friction(1.0 / 60.0)
        AIControl()
        Collision()
        Cleanup()

        println("Frame", frame, "- Player HP:", player.Health.hp)
        frame += 1

    if player.Health.hp > 0:
        println("YOU WIN!")
    else:
        println("GAME OVER")
```

---

## Future Directions

1. **Networking**: Lockstep multiplayer, network-transparent RPCs
2. **VR Support**: Direct headset integration
3. **WASM Export**: Run Jules games in browsers
4. **Mobile**: Direct iOS/Android support
5. **Visual Editor**: Node-based AI behavior trees
6. **Marketplace**: Publish/share game systems and models
7. **Distributed Training**: Multi-machine ML training
8. **Quantum Computing**: QBase (quantum ML framework built on Jules)

---

## Community Contribution

Jules welcomes contributions! See:
- **IMPLEMENTATION_GUIDE.md** - Roadmap of 15 remaining features
- **STATUS_AND_QUICKSTART.md** - How to build and test
- **github.com/your-org/jules** - Source repository

---

**Jules Language v1.0 Alpha**
**Status**: 25% complete (5/20 subsystems)
**Target**: Full-featured game dev + ML platform
**Timeline**: 12-18 months to 1.0 stable release

