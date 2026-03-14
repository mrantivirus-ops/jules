# Jules

Jules is a small language project (lexer + parser + typechecker + interpreter) implemented in Rust.

## ✅ Build

Requires Rust (stable) and Cargo.

```bash
# From the repository root
cargo build
```

## ▶️ Run

By default, the CLI operates on a `.jules` source file.

```bash
# Run a Jules program
cargo run -- run example.jules

# Typecheck only (no execution)
cargo run -- check example.jules

# Run the REPL
cargo run -- repl
```

## 🧪 Testing

```bash
cargo test
```

## 📄 Example

The repository includes `example.jules` as a minimal demo program. You can edit it and re-run:

```bash
cargo run -- run example.jules
```
