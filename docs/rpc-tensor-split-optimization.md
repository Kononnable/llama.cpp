# RPC tensor-split: performance findings, fixes, and proposals

This documents the performance work on the RPC tensor-split path for a 3-backend
Meta split (`ROCm0` attention GPU + `RPC0` remote expert CPU + `CPU` local expert
CPU). It covers one optimization already committed, and two problems found by
inspecting the RPC server log during the compute phase.

## Setup / configuration

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (35B MoE, ~3B active, linear attention
  + Gated Delta Net + Lightning Indexer + DeepSeek V4 HC).
- Split mode: `-sm tensor`, `-ts 1,1e,1e`.
- Devices (3-backend Meta split):
  - `ROCm0` (RX 6600 GPU) - non-expert (attention) data/weights.
  - `RPC0` (127.0.0.1:50052, remote CPU) - expert weights.
  - `CPU` (local Ryzen 7 3800X) - expert weights.
- The experts are split across RPC0 + CPU; attention work is on ROCm0.

---

## 1. Allreduce optimization - implemented and committed

### Problem

Every subgraph boundary ends in an allreduce that puts the reduced value on all 3
devices. For the expert FFN output (`ffn_moe_out`, `ne=[2048x1]`, `RPC0(Y) CPU(Y)
ROCm0(-)`) the generic 3-way butterfly walked through the empty backend (ROCm0),
costing 3 serial RPC hops per expert allreduce:

1. `CPU -> ROCm0` (excess fold)
2. `ROCm0 <-> RPC0` (butterfly pair)
3. `ROCm0 -> CPU` (copyback)

### Fix (committed `5d9f70cdd`)

For the common `2 holders + 1 empty` expert case, replace the butterfly with a
2-hop scheme that puts the same value on the same 3 devices:

1. **Stage 1**: `RPC0 <-> CPU` exchange in parallel - each holder becomes
   `partial_self + partial_other` = full.
2. **Stage 2**: copy the full value to the empty backend (ROCm0), sourced from the
   **local (non-RPC) holder** to avoid a remote round-trip.

This is correct-by-construction (identical values on identical devices, verified
byte-identical output). Guarded to only the exact `2 holders + 1 empty` case;
everything else falls through to the original butterfly.

### Result

| Metric | Butterfly (baseline) | 2-stage reduce | Delta |
|--------|------|------|-------|
| clean ms/token | ~217 | ~195 | **~22 ms** |
| instrumented allreduce/token | 82.96 ms | 69.74 ms | -13.2 ms |
| instrumented total/token | 243.62 ms | 229.21 ms | -14.4 ms |

### Status / relevance

Still relevant as the baseline allreduce path. It is the reference implementation
the further proposals below build on.

---

## 2. Problem A - loading-time bottleneck: expert weight written block-by-block

### Symptom

Model load takes ~421 s. The RPC log during loading (and the first prompt/graph
build) is dominated by tiny `[set_tensor] ... size: 176` calls - ~289k in the
tail of the log, writing the same buffer at incrementing offsets from one source
pointer.

### Root cause

**176 bytes = exactly one Q5_K block** (256 elements/block, 176 bytes/block). The
tensor being uploaded block-by-block is the split expert-down weight
`blk.*.ffn_down_exps.weight` `[512, 2048, 256] Q5_K`, split across RPC0/CPU.
Each backend's slice is written **one quantized block per RPC call** instead of in
a few large transfers. Each call is a separate TCP send + server-side op, so a
~184 MB weight turns into ~1M round-trips.

This is a pure loading-time bottleneck: **zero** `size:176` writes appear in
steady-state generation (the final ~1000 lines of the log contain none).

### Solution (implemented, cherry-picked from PR #26610)

Implement `get/set_tensor_2d` on the RPC buffer (and the `set/get_tensor_2d_async`
backend variants) so a strided `set_2d` call packs all rows into one bulk transfer
instead of falling back to a per-row `set_tensor` loop. When the meta backend's
`set_tensor` calls `ggml_backend_tensor_set_2d` on the RPC slice, the RPC buffer
now packs the whole 2D region into a single `RPC_CMD_SET_TENSOR_2D` message.

### Result

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Load time | 421 s | 67 s | ~6.3x |
| 176-byte set_tensor writes | ~289k | 0 | gone |
| rpc log size | 1.8 GB | 751 KB | -99.96% |

Generation throughput is unaffected (the 176-byte writes never appeared in
steady-state generation); this is a pure loading fix.

---

## 3. Problem B - generation bottleneck: blocking allreduce reads

### Symptom

Generation runs ~195 ms/token. Per token the RPC server receives roughly:
- ~42 **blocking `get_tensor`** round-trips (allreduce reads from RPC0),
- ~236 one-way `set_tensor` sends,
- ~129 `graph_compute` sends (≈81 subgraph computes + ~48 tiny `n_nodes:1` ADD
  graphs from the allreduce).

The ~42 blocking round-trips are the serialized allreduce cost that remains.

### Root cause

`ggml_backend_tensor_copy_async()` only performs an async copy if the destination
backend implements `cpy_tensor_async`. **Neither the CPU backend nor the RPC
backend implements it** (both are `NULL`), so every allreduce copy falls back to:

```
sync(src); sync(dst); blocking copy    // one full round-trip per copy
```

Consequences:
- Copies that should run in parallel (the 2-stage exchange, the attention
  broadcast pair) actually **serialize**.
- The blocking copy into/from RPC0 is a `get_tensor` (round-trip) or `set_tensor`
  send per operation.
- Each allreduce ADD is submitted as a separate 1-node `graph_compute` RPC,
  adding serialization overhead even for a trivial add.

### Solution (proposed)

Implement `cpy_tensor_async` for the RPC backend (and CPU), so cross-backend
copies are queued asynchronously and only waited on at the right point. The RPC
server already processes commands in order on a single connection and
`graph_compute` is already fire-and-forget, so the async-copy machinery can build
on that. It needs:
- a real async copy command + in-flight tracking on the RPC client, and
- making RPC `synchronize` wait for pending async ops (currently a no-op).

This lets the parallel allreduce copies overlap and removes the per-copy
double-sync + blocking read. It is the main remaining lever for generation
throughput on top of the committed 2-stage reduce.

---

## 4. Recommendation

- Keep the committed 2-stage allreduce (safe, correct, ~22 ms/token).
- **Problem B (async copies)** is the generation-throughput lever - continue
  there if generation speed is the goal.
- **Problem A (blocked block-by-block load)** is the largest absolute-time win
  (421 s -> seconds) and is simpler/one-shot - worth doing if load time matters.

## 5. Transport notes (from RPC log inspection)

- Raw TCP `send`/`recv` with `TCP_NODELAY`.
- `set_tensor` (small, below HASH_THRESHOLD) is one-way / fire-and-forget.
- `get_tensor` is a blocking request/response round-trip.
- `graph_compute` is fire-and-forget (client does not wait; server processes
  commands in order, so ordering is preserved).
- `get_alloc_size` and `alloc_buffer` are one-time during graph build, not per
  token (negligible).
