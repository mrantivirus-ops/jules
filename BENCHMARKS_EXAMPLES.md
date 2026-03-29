# Jules Low-End PC Cookbook (Benchmarks + 36 Examples)

This page is designed for **low-end PCs** first:
- tiny maps by default
- short loops
- modest tick counts
- copy/paste examples that can be scaled up later

> For timing, use shell timing: `/usr/bin/time -p cargo run --offline -- run your_file.jules`

---

## Low-end baseline profile

Use these limits unless you deliberately stress test:
- Grid maps: `16x16` to `48x48`
- Chunked maps: keep writes under `500`
- `game::run_loop`: `30..120` ticks
- Math loops: `10_000..100_000` iterations

---

## 36 Jules examples

### A) Core language (1–12)

**1. Variables**
```jules
fn main() { let hp = 100; let mana = 40; println(hp, mana); }
```

**2. Mutable value**
```jules
fn main() { let mut score = 0; score = score + 5; println(score); }
```

**3. For range**
```jules
fn main() { let mut s = 0; for i in 0..10 { s = s + i; } println(s); }
```

**4. If branch**
```jules
fn main() { let x = 7; if x > 5 { println("big"); } else { println("small"); } }
```

**5. Function call**
```jules
fn add(a, b) { a + b } fn main() { println(add(2, 3)); }
```

**6. Float math**
```jules
fn main() { let x = 0.5; println(sin(x), cos(x), tanh(x)); }
```

**7. Integer micro-loop**
```jules
fn main() { let mut c = 0; for i in 0..20000 { c = c + ((i * 3) % 17); } println(c); }
```

**8. Mixed arithmetic**
```jules
fn main() { let a = 12; let b = 5; println(a + b, a - b, a * b, a / b); }
```

**9. Clamp-like pattern**
```jules
fn main() { let x = -3; if x < 0 { println(0); } else { println(x); } }
```

**10. Nested loops**
```jules
fn main() { let mut t = 0; for y in 0..8 { for x in 0..8 { t = t + x + y; } } println(t); }
```

**11. Simple benchmark checksum**
```jules
fn main() { let mut z = 0; for i in 0..50000 { z = z + (i % 97); } println("chk=", z); }
```

**12. Print labels**
```jules
fn main() { println("low-end profile active"); println("done"); }
```

---

### B) Graphics assets (13–18)

**13. Create material**
```jules
fn main() { let m = graphics::create_material(0.2, 0.8, 0.2, 1.0); println(m); }
```

**14. Create mesh**
```jules
fn main() { let mesh = graphics::create_mesh("quad", 1.0); println(mesh); }
```

**15. Create sprite**
```jules
fn main() { let s = graphics::create_sprite("grass", 1.0, 1.0); println(s); }
```

**16. Create model**
```jules
fn main() {
  let mesh = graphics::create_mesh("quad", 1.0);
  let model = graphics::create_model("tree", mesh);
  println(model);
}
```

**17. Sprite object**
```jules
fn main() {
  let mat = graphics::create_material(0.8, 0.8, 0.8, 1.0);
  let spr = graphics::create_sprite("tile", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  println(obj);
}
```

**18. Model object**
```jules
fn main() {
  let mesh = graphics::create_mesh("quad", 1.0);
  let model = graphics::create_model("rock", mesh);
  let mat = graphics::create_material(0.5, 0.5, 0.6, 1.0);
  let obj = graphics::create_object("model", model, mat);
  println(obj);
}
```

---

### C) Grid maps (19–26)

**19. Create empty map**
```jules
fn main() { let map = graphics::create_grid_map(16, 16); println(map); }
```

**20. Set one cell**
```jules
fn main() {
  let mat = graphics::create_material(1.0, 1.0, 1.0, 1.0);
  let spr = graphics::create_sprite("a", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_grid_map(16, 16);
  graphics::set_grid_cell(map, 0, 0, obj);
  println(graphics::render_grid_map(map));
}
```

**21. Fill first row**
```jules
fn main() {
  let mat = graphics::create_material(0.7, 0.7, 0.7, 1.0);
  let spr = graphics::create_sprite("line", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_grid_map(32, 16);
  for x in 0..32 { graphics::set_grid_cell(map, x, 0, obj); }
  println(graphics::render_grid_map(map));
}
```

**22. Checker pattern**
```jules
fn main() {
  let mat = graphics::create_material(0.3, 0.9, 0.3, 1.0);
  let spr = graphics::create_sprite("g", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_grid_map(24, 24);
  for i in 0..576 {
    let x = i % 24; let y = i / 24;
    if (x + y) % 2 == 0 { graphics::set_grid_cell(map, x, y, obj); }
  }
  println(graphics::render_grid_map(map));
}
```

**23. Tiny run loop**
```jules
fn main() {
  let mat = graphics::create_material(0.6, 0.6, 1.0, 1.0);
  let spr = graphics::create_sprite("p", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_grid_map(16, 16);
  for i in 0..64 { graphics::set_grid_cell(map, i % 16, i / 16, obj); }
  println(game::run_loop(map, 30, 0.016));
}
```

**24. Medium low-end render**
```jules
fn main() {
  let m = graphics::create_material(0.4, 0.8, 0.9, 1.0);
  let s = graphics::create_sprite("w", 1.0, 1.0);
  let o = graphics::create_object("sprite", s, m);
  let map = graphics::create_grid_map(40, 40);
  for i in 0..800 { graphics::set_grid_cell(map, i % 40, i / 40, o); }
  println(game::run_loop(map, 60, 0.016));
}
```

**25. Sprite + model overlay**
```jules
fn main() {
  let mesh = graphics::create_mesh("quad", 1.0);
  let mat_a = graphics::create_material(0.2, 0.8, 0.2, 1.0);
  let mat_b = graphics::create_material(0.8, 0.3, 0.2, 1.0);
  let spr = graphics::create_sprite("g", 1.0, 1.0);
  let mdl = graphics::create_model("tree", mesh);
  let a = graphics::create_object("sprite", spr, mat_a);
  let b = graphics::create_object("model", mdl, mat_b);
  let map = graphics::create_grid_map(24, 24);
  for i in 0..576 { graphics::set_grid_cell(map, i % 24, i / 24, a); }
  for i in 0..60 { graphics::set_grid_cell(map, (i * 3) % 24, (i * 5) % 24, b); }
  println(graphics::render_grid_map(map));
}
```

**26. Quick dense benchmark**
```jules
fn main() {
  let mat = graphics::create_material(0.9, 0.9, 0.9, 1.0);
  let spr = graphics::create_sprite("bench", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_grid_map(48, 48);
  for i in 0..2304 { graphics::set_grid_cell(map, i % 48, i / 48, obj); }
  println("rendered=", game::run_loop(map, 90, 0.016));
}
```

---

### D) Chunked giant maps (27–32, low-end-safe)

**27. Create chunked map**
```jules
fn main() { let m = graphics::create_chunked_grid_map(50000, 50000, 64); println(m); }
```

**28. Sparse 100 writes**
```jules
fn main() {
  let mat = graphics::create_material(0.2, 0.2, 1.0, 1.0);
  let spr = graphics::create_sprite("dot", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_chunked_grid_map(100000, 100000, 64);
  for i in 0..100 { graphics::set_chunked_grid_cell(map, i * 97, i * 53, obj); }
  println(graphics::render_chunked_grid_map(map));
}
```

**29. Sparse 300 writes**
```jules
fn main() {
  let mat = graphics::create_material(0.7, 0.5, 0.2, 1.0);
  let spr = graphics::create_sprite("pt", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_chunked_grid_map(120000, 120000, 64);
  for i in 0..300 { graphics::set_chunked_grid_cell(map, (i * 17) % 90000, (i * 29) % 90000, obj); }
  println(graphics::render_chunked_grid_map(map));
}
```

**30. Hot chunk (overwrite)**
```jules
fn main() {
  let mat = graphics::create_material(1.0, 0.4, 0.1, 1.0);
  let spr = graphics::create_sprite("hot", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_chunked_grid_map(100000, 100000, 32);
  for i in 0..1000 { graphics::set_chunked_grid_cell(map, 50000 + (i % 32), 50000 + ((i * 3) % 32), obj); }
  println(graphics::render_chunked_grid_map(map));
}
```

**31. Two distant clusters**
```jules
fn main() {
  let mat = graphics::create_material(0.5, 1.0, 0.5, 1.0);
  let spr = graphics::create_sprite("c", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_chunked_grid_map(150000, 150000, 64);
  for i in 0..120 { graphics::set_chunked_grid_cell(map, 1000 + i, 1000 + i, obj); }
  for i in 0..120 { graphics::set_chunked_grid_cell(map, 140000 + i, 140000 + i, obj); }
  println(graphics::render_chunked_grid_map(map));
}
```

**32. Low-end giant benchmark**
```jules
fn main() {
  let mat = graphics::create_material(0.8, 0.8, 0.2, 1.0);
  let spr = graphics::create_sprite("g", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
  let map = graphics::create_chunked_grid_map(200000, 200000, 64);
  for i in 0..500 { graphics::set_chunked_grid_cell(map, (i * 211) % 160000, (i * 127) % 160000, obj); }
  println("visible=", graphics::render_chunked_grid_map(map));
}
```

---

### E) Benchmark harness patterns (33–36)

**33. Minimal benchmark wrapper**
```jules
fn bench() { let mut s = 0; for i in 0..50000 { s = s + (i % 31); } println(s); }
fn main() { bench(); }
```

**34. Compare two loop sizes**
```jules
fn run(n) { let mut s = 0; for i in 0..n { s = s + (i % 7); } println(s); }
fn main() { run(20000); run(80000); }
```

**35. Fixed workload + label**
```jules
fn main() {
  println("workload=A");
  let mut s = 0;
  for i in 0..60000 { s = s + ((i * 5) % 29); }
  println("result=", s);
}
```

**36. End-to-end low-end demo**
```jules
fn main() {
  let m = graphics::create_material(0.3, 0.6, 1.0, 1.0);
  let s = graphics::create_sprite("end", 1.0, 1.0);
  let o = graphics::create_object("sprite", s, m);
  let map = graphics::create_grid_map(20, 20);
  for i in 0..200 { graphics::set_grid_cell(map, i % 20, i / 20, o); }
  println("frames=", game::run_loop(map, 45, 0.016));
}
```

---

## Quick shell commands

```bash
# Validate compiler/build health
cargo check --offline

# Run any example file you saved
/usr/bin/time -p cargo run --offline -- run my_example.jules
```

If you want, I can generate these 36 snippets as actual `.jules` files in an `examples/` folder next.

### Bonus: Native fused linear example (Rust API)

```rust
let x = Tensor::he(vec![32, 128]);
let w = Tensor::xavier(vec![128, 64]);
let b = Tensor::zeros(vec![64]);
let y = x.linear_native(&w, Some(&b), Some("relu"))?;
```

---

## Step-by-step syntax troubleshooting (what is wrong + how to fix)

This section is intentionally explicit for beginners.

### 1) Missing semicolon after `let`

**Wrong**
```jules
fn main() {
  let hp = 100
  println(hp);
}
```

**What is wrong**
1. `let hp = 100` is a statement.
2. Statements must end with `;`.
3. Parser reaches `println` while still expecting the previous statement to end.

**Fix**
```jules
fn main() {
  let hp = 100;
  println(hp);
}
```

---

### 2) Missing `)` in a function call

**Wrong**
```jules
fn main() {
  println("hello";
}
```

**What is wrong**
1. Call starts with `(`.
2. Every opening `(` must be closed with `)`.
3. `;` appears before the call is closed.

**Fix**
```jules
fn main() {
  println("hello");
}
```

---

### 3) Missing `}` at end of block

**Wrong**
```jules
fn main() {
  let x = 1;
  if x > 0 {
    println("ok");
}
```

**What is wrong**
1. `if x > 0 {` opens one block.
2. `fn main() {` opens another block.
3. Only one `}` appears, so one block remains unclosed.

**Fix**
```jules
fn main() {
  let x = 1;
  if x > 0 {
    println("ok");
  }
}
```

---

### 4) Using assignment `=` instead of comparison `==`

**Wrong**
```jules
fn main() {
  let x = 3;
  if x = 3 { println("three"); }
}
```

**What is wrong**
1. `if` needs a boolean condition.
2. `x = 3` is assignment, not comparison.
3. Use `==` when you mean “is equal to”.

**Fix**
```jules
fn main() {
  let x = 3;
  if x == 3 { println("three"); }
}
```

---

### 5) Calling function before giving required arguments

**Wrong**
```jules
fn add(a, b) { a + b }
fn main() { println(add(1)); }
```

**What is wrong**
1. `add` is declared with two parameters.
2. Call site gives only one argument.
3. Arity mismatch triggers an error.

**Fix**
```jules
fn add(a, b) { a + b }
fn main() { println(add(1, 2)); }
```

---

### 6) Typo in namespace function name

**Wrong**
```jules
fn main() {
  let map = graphics::create_gridmap(16, 16);
}
```

**What is wrong**
1. Function name is misspelled.
2. Runtime builtins use exact names.
3. Unknown function error is expected.

**Fix**
```jules
fn main() {
  let map = graphics::create_grid_map(16, 16);
}
```

---

### 7) Forgetting `mut` for reassignment

**Wrong**
```jules
fn main() {
  let score = 0;
  score = score + 1;
}
```

**What is wrong**
1. `let score = 0;` creates an immutable binding.
2. Reassignment requires mutable binding.

**Fix**
```jules
fn main() {
  let mut score = 0;
  score = score + 1;
}
```

---

### 8) Unbalanced range loop braces

**Wrong**
```jules
fn main() {
  for i in 0..10
    println(i);
  }
}
```

**What is wrong**
1. `for ...` expects a block with `{ ... }`.
2. Current code has extra `}` and missing `{` after loop header.

**Fix**
```jules
fn main() {
  for i in 0..10 {
    println(i);
  }
}
```

---

### 9) Wrong object kind string

**Wrong**
```jules
fn main() {
  let mat = graphics::create_material(1.0, 1.0, 1.0, 1.0);
  let spr = graphics::create_sprite("a", 1.0, 1.0);
  let obj = graphics::create_object("sprites", spr, mat);
}
```

**What is wrong**
1. `graphics::create_object` accepts `"sprite"` or `"model"`.
2. `"sprites"` is invalid.

**Fix**
```jules
fn main() {
  let mat = graphics::create_material(1.0, 1.0, 1.0, 1.0);
  let spr = graphics::create_sprite("a", 1.0, 1.0);
  let obj = graphics::create_object("sprite", spr, mat);
}
```

---

### 10) Wrong argument count for map API

**Wrong**
```jules
fn main() {
  let map = graphics::create_grid_map(32);
}
```

**What is wrong**
1. `create_grid_map` needs both width and height.
2. One argument provided; two required.

**Fix**
```jules
fn main() {
  let map = graphics::create_grid_map(32, 32);
}
```

---

### 11) Coordinate out of bounds

**Wrong**
```jules
fn main() {
  let m = graphics::create_material(1.0, 1.0, 1.0, 1.0);
  let s = graphics::create_sprite("a", 1.0, 1.0);
  let o = graphics::create_object("sprite", s, m);
  let map = graphics::create_grid_map(16, 16);
  graphics::set_grid_cell(map, 99, 0, o);
}
```

**What is wrong**
1. Valid x range is `0..15` for width `16`.
2. `99` is outside map bounds.

**Fix**
```jules
fn main() {
  let m = graphics::create_material(1.0, 1.0, 1.0, 1.0);
  let s = graphics::create_sprite("a", 1.0, 1.0);
  let o = graphics::create_object("sprite", s, m);
  let map = graphics::create_grid_map(16, 16);
  graphics::set_grid_cell(map, 15, 0, o);
}
```

---

### 12) Step-by-step debugging workflow

When a file fails, use this exact order:
1. Run `cargo run --offline -- check your_file.jules`.
2. Fix the **first** parser error (later ones may be cascading).
3. Re-run check.
4. Once syntax is clean, run `cargo run --offline -- run your_file.jules`.
5. If runtime error appears, verify:
   - function names
   - argument count/order
   - object IDs exist before map placement
   - coordinates are in bounds

---

## Game + ML production snippets (new)

These patterns were added to align Jules with common engine/runtime needs
(animation, audio, huge maps) and ML deployment needs (memory budgeting for
low-end hardware).

### 37) Build a 2-frame sprite animation

```jules
fn main() {
  let s0 = graphics::create_sprite("hero_idle_0", 1.0, 1.0);
  let s1 = graphics::create_sprite("hero_idle_1", 1.0, 1.0);
  let anim = graphics::create_animation([s0, s1], 8.0);
  let frame = graphics::animation_frame(anim, 0.22);
}
```

### 38) Basic audio playback lifecycle

```jules
fn main() {
  let clip = audio::create_clip("laser", 0.3);
  let voice = audio::play_clip(clip, 0.8, false);
  let active = audio::active_voices();
  audio::stop_clip(voice);
}
```

### 39) Looping ambience for giant maps

```jules
fn main() {
  let ambience = audio::create_clip("wind_loop", 20.0);
  let _voice = audio::play_clip(ambience, 0.35, true);
}
```

### 40) Map + animation together

```jules
fn main() {
  let mat = graphics::create_material(1.0, 1.0, 1.0, 1.0);
  let s0 = graphics::create_sprite("water_0", 1.0, 1.0);
  let s1 = graphics::create_sprite("water_1", 1.0, 1.0);
  let water_anim = graphics::create_animation([s0, s1], 6.0);
  let frame_sprite = graphics::animation_frame(water_anim, 0.5);
  let water_obj = graphics::create_object("sprite", frame_sprite, mat);
  let map = graphics::create_grid_map(32, 32);
  graphics::set_grid_cell(map, 4, 7, water_obj);
}
```

### 41) Estimate model memory for low-end targets

```jules
fn main() {
  let params = 12000000; // 12M params
  let fp16_bytes = ml::estimate_inference_memory(params, 2.0);
  let int8_mb = ml::estimate_one_byte_model_mb(params);
}
```

### 42) Compare candidate parameter budgets quickly

```jules
fn main() {
  let tiny = ml::estimate_one_byte_model_mb(3000000);
  let base = ml::estimate_one_byte_model_mb(7000000);
  let big  = ml::estimate_one_byte_model_mb(15000000);
}
```

### 43) Native transformer block skeleton (Rust API)

```rust
let x = Tensor::he(vec![2, 128, 256]);          // [batch, seq, hidden]
let gamma = Tensor::ones(vec![256]);
let beta = Tensor::zeros(vec![256]);
let x_norm = x.layer_norm_last_dim(Some(&gamma), Some(&beta), 1e-5)?;

let q = Tensor::he(vec![2, 128, 64]);
let k = Tensor::he(vec![2, 128, 64]);
let v = Tensor::he(vec![2, 128, 64]);
let attn = Tensor::scaled_dot_product_attention(&q, &k, &v, true)?;
```
