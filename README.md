# Jules

Jules is a game-dev + ML-oriented language/runtime with:
- ECS + system runtime
- ML tensor engine (CPU + GPU hooks)
- C ABI for embedding in external engines

## Engine / Host Integrations

### C++
Use `bindings/cpp/jules.hpp` and link against the generated `libjules`.

### C# (Unity/Godot C#)
Use `bindings/csharp/Jules.cs` (`DllImport("jules")`) and ship the native library next to the game binary.

### Python
Use `bindings/python/jules.py` (ctypes wrapper) or build with `--features python` for PyO3 bindings.

## C ABI entry points

- `jules_run_file_ffi(const char* path)`
- `jules_check_code_ffi(const char* source)`
- tensor lifecycle + shape/data APIs
- physics sample APIs

## Experimental low-byte inference

Jules now includes an INT8 inference path for linear layers:
- `Tensor::quantize_linear_int8()` (offline weight quantization)
- `Tensor::linear_int8()` (inference with 1-byte weights + per-channel scales)
- `Int8LinearWeights::effective_bytes_per_param()` to estimate memory cost

This targets roughly **~1 byte/parameter** plus small scale overhead, for
running (inference) workloads.

For larger INT8 linear projections, Jules dequantizes once and dispatches the
projection through SGEMM for higher throughput.

## Faster CPU matmul path

`Tensor::matmul()` uses a cache-aware blocked kernel for larger matrices
(with existing multi-threading), while keeping the low-overhead unrolled kernel
for smaller shapes.

The CPU fallback in `gpu_backend.rs` now mirrors this strategy for `matmul`
as well (transpose + blocked kernel + threaded row chunks), following the same
core ideas used by high-performance JIT stacks: contiguous access, tiling, and
parallel work partitioning.

For large GEMMs, Jules now routes to an in-tree blocked **Jules kernel**
(`jules_kernel`) instead of relying on an external SGEMM API call.

## Native accelerated linear algebra

Jules includes a native fused linear API for common inference paths:

- `Tensor::linear_native(weights, bias, activation)`
  - Computes `x @ weights`
  - Optionally adds bias
  - Optionally applies activation (`relu`, `tanh`, `sigmoid`)
  - Performs the bias/activation in a single post-GEMM pass

`linear_jax(...)` remains as a compatibility alias for older scripts/docs.

`Tensor::matmul()` also uses a native Jules-kernel dispatch heuristic
for larger or long-and-skinny GEMMs.

## Native transformer primitives

Jules now includes in-tree transformer building blocks:

- `Tensor::layer_norm_last_dim(gamma, beta, eps)`  
  layer normalization over the last dimension.
- `Tensor::rms_norm_last_dim(weight, eps)`  
  RMS normalization over the last dimension.
- `Tensor::scaled_dot_product_attention(q, k, v, causal)`  
  native scaled dot-product attention for rank-3 tensors.

## Vision

Jules is being built to be a top-tier language for both **ML** and **game
development**: high-performance numerics + practical runtime gameplay systems.

## Rich `@ai(...)` agent decorator

You can configure architecture + learning knobs directly from the AI decorator:

```jules
@ai(network="128->64->32", lr=0.0003, input=128, output=32)
agent Bot {
  // ...
}
```

Supported keys in `@ai(...)`:
- `network` (or `arch`, `architecture`) — architecture string
- `lr` (or `learning_rate`) — learning rate
- `input` — expected input width
- `output` — expected output width

## Automated syntax fix flow

Use:

```bash
jules fix path/to/file.jules
```

`jules fix` applies safe automatic syntax fixes from parser diagnostics
(e.g. missing `;`, missing `)`, `]`, `}`, and common `fun`/`func` → `fn`
keyword typo recovery).

## Run a sample game script

```bash
cargo run --offline -- run small_game.jules
```

## Run game + NN together (Jules script)

```bash
cargo run --offline -- run game_nn_demo.jules
```

`game_nn_demo.jules` combines game map simulation (`game::run_loop`) with a tiny
native NN policy builtin (`ml::nn_binary`) and prints both game and ML outputs.

## Code-only maps, sprites, and models

Jules runtime graphics helpers now support creating assets and a tile/grid map from code:

```jules
let mesh = graphics::create_mesh("quad", 1.0);
let mat = graphics::create_material(1.0, 1.0, 1.0, 1.0);
let sprite = graphics::create_sprite("grass", 1.0, 1.0);
let model = graphics::create_model("tree", mesh);
let grass_obj = graphics::create_object("sprite", sprite, mat);
let tree_obj = graphics::create_object("model", model, mat);

let map = graphics::create_grid_map(16, 16);
graphics::set_grid_cell(map, 0, 0, grass_obj);
graphics::set_grid_cell(map, 1, 0, tree_obj);
let drawn = graphics::render_grid_map(map); // number of visible cells

// giant/sparse worlds: allocate only touched chunks
let giant = graphics::create_chunked_grid_map(100000, 100000, 64);
graphics::set_chunked_grid_cell(giant, 90000, 90000, tree_obj);
let giant_drawn = graphics::render_chunked_grid_map(giant);
```

For higher-level game scripting, use:
- `game::make_object(kind, name, size, r, g, b, a)` → object id
- `game::build_map(width, height, cells_array)` → map id
- `game::run_loop(map_id, ticks, dt)` → total rendered tiles across ticks

Game-tooling helpers now also include:
- `graphics::create_animation(frames_array, fps)` → animation id
- `graphics::animation_frame(animation_id, elapsed_seconds)` → sprite id
- `audio::create_clip(name, length_seconds)` → clip id
- `audio::play_clip(clip_id, volume, looped)` → voice id
- `audio::stop_clip(voice_id)` / `audio::active_voices()`

Low-level map APIs:
- `graphics::create_grid_map(width, height)` / `graphics::set_grid_cell(...)` / `graphics::render_grid_map(...)`
- `graphics::create_chunked_grid_map(width, height, chunk_size)` / `graphics::set_chunked_grid_cell(...)` / `graphics::render_chunked_grid_map(...)`

## ML memory planning helpers

To help low-end deployment planning and 1-byte/param targeting:
- `ml::estimate_inference_memory(param_count, bytes_per_param)` → bytes
- `ml::estimate_one_byte_model_mb(param_count)` → MiB at 1 byte/param

## Large Jules benchmark/examples page

See `BENCHMARKS_EXAMPLES.md` for a **low-end-PC-friendly** benchmark cookbook
with **36 Jules examples** (core language, graphics assets, dense maps, chunked
maps, and benchmark harness patterns), plus a step-by-step syntax
troubleshooting guide (what is wrong and exactly how to fix it).
