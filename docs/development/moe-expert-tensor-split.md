# MoE expert tensor split with host offload

Design summary. Not implemented yet.

## Goal

Run a MoE model that does not fit in one GPU across two or more backends, possibly on
two machines over RPC. Dense tensors use pipeline (layer) split, expert tensors use
feature split. Expert weights may live in pinned host RAM and are streamed to the
owning GPU during prompt processing.

Constraints:

- Latency is the binding constraint, not bandwidth. Cost is measured in serialized
  RPC round trips and barriers per token.
- Weights never cross the network at inference time.
- Communication per MoE layer is one activation mirror out and one partial back.

## Current state

- Layer split (`LLAMA_SPLIT_MODE_LAYER`, default) assigns each layer to one device
  (`src/llama-model.cpp:1445-1517`). No cross-device traffic during decode.
- Tensor split (`-sm tensor`) wraps all devices in a Meta device and slices each
  tensor by a per-tensor split axis (`src/llama.cpp:167-180`,
  `src/llama-model.cpp:371-850`, `ggml/src/ggml-backend-meta.cpp`).
- The Meta path already supports expert feature split: `ffn_gate_exps` / `ffn_up_exps`
  split axis 1, `ffn_down_exps` split axis 0, down output is `PARTIAL` and reduced.
- RPC has a single serialized dispatcher and no `cpy_tensor_async`
  (`ggml/src/ggml-rpc/ggml-rpc.cpp:1054`), so every cross-device copy falls back to
  synchronize both backends plus a blocking copy (`ggml/src/ggml-backend.cpp:511`).
- RCCL/NCCL allreduce is unavailable when a sub-device is RPC
  (`ggml-cuda.cu:1211`).
- `ROCm_Host` is `cudaMallocHost` pinned memory exposed as a CPU buffer
  (`ggml-cuda.cu:1262-1327`). On discrete GPUs the CUDA/ROCm backend does not accept
  it in `supports_buft` (`:5516`), so a weight in host memory is computed on the CPU
  unless the scheduler offloads the op.
- Existing RAM offload: `-cmoe` places expert tensors in the CPU buffer; the
  scheduler reassigns the `MUL_MAT_ID` op to a GPU when the batch is >= 32
  (`ggml/src/ggml-backend.cpp:969-975`) and copies only the used experts
  (`:1690-1788`). Threshold is `GGML_OP_OFFLOAD_MIN_BATCH`, default 32
  (`ggml-cuda.cu:5528-5539`).
- The scheduler can pin a tensor to a backend with
  `ggml_backend_sched_set_tensor_backend`; llama already uses this to pin the last op
  of a layer to the layer's device (`src/llama-context.cpp:2537`,
  `src/llama-graph.cpp:2720`).

## Decisions

1. Split semantics: option A, feature split (the existing Meta axes). Not
   expert-parallel. Feature split is ratio-exact and never idles a device; expert
   parallel is routing-dependent and can put 0 of k experts on a device.
2. Reduce: explicit shard tensors plus a graph-level `ggml_add`, reduce-to-owner. The
   result is not sent to backends that do not consume it. Do not use the Meta
   butterfly reduce.
3. `ffn_down_exps_s` and `ffn_down_exps_b` are applied once after the combine on the
   owner. `ffn_gate_exps_s` / `ffn_up_exps_s` are mirrored to every expert device and
   applied before swiglu, because swiglu is nonlinear.
4. `-tse` is a separate array from `-ts`. `-ts 1,0` puts all dense layers on backend 0;
   `-tse 1,3` puts 25 percent of each expert's feature width on backend 0 and 75
   percent on backend 1. A zero entry means the device holds no expert features.
5. Dense membership is controlled by `-ts`, expert membership by `-tse`.
6. Reduce owner is the backend that owns the layer (from `-ts`). `-ts 1,1,0` is valid.
7. Representation: fixed array of `GGML_BACKEND_META_MAX_DEVICES` (16) per sharded
   expert tensor in `llama_layer`, zero-initialized, plus one shard count per layer.
   Wasted space is under 1 KB per layer.
8. Shard creation uses `create_tensor_shard`.
9. Decode (batch < 32) runs on the CPU of the machine that owns the shard's memory, as
   with `-cmoe` today. The per-layer network round trip is accepted because it is
   small compared to the expert activation time.
10. `--fit` is a non-goal.
11. First targets: `qwen35moe` and `qwen4exp` only.
12. Prefill RAM to VRAM copy reuses the existing `op_offload` path; no copy is forced.

## Split semantics

Per expert weight tensor:

| tensor | shape | split axis |
|---|---|---|
| `ffn_gate_exps` / `ffn_up_exps` | `[n_embd, n_ff, n_expert]` | 1 |
| `ffn_gate_up_exps` | `[n_embd, 2*n_ff, n_expert]` | 1, segments `{n_ff, 2}` |
| `ffn_down_exps` | `[n_ff, n_embd, n_expert]` | 1 (output `n_embd`) |

Siblings:

- `ffn_gate_exps_s` / `ffn_up_exps_s` (`{n_expert}`): mirrored to each expert device,
  applied before swiglu.
- `ffn_down_exps_s` (`{n_expert}`): applied post-combine on the owner.
- `ffn_down_exps_b` (`{n_embd, n_expert}`): applied post-combine on the owner.
- `_exps_in_s`: unused by the MoE graph; ignore.

Shard boundaries must be multiples of `lcm(blck_size, 128)` for the FFN expert case.
Rows stay whole, so the quant block structure is preserved.

## Configuration

New public parameter:

```c
struct llama_model_params {
    ...
    const float * tensor_split_experts; // per-device share of expert feature width, size llama_max_devices()
};
```

Default is `nullptr`. New flag `-tse` / `--tensor-split-experts`, parsed like `-ts`,
stored in `common_params` and copied into the model params.

Allowed only with `LLAMA_SPLIT_MODE_LAYER` and the allow-listed archs; otherwise throw.
Require explicit `-ngl` since `--fit` is a non-goal.

## Placement and offload

- Each shard is a normal weight tensor placed through the owning device's buft list.
  Overrides (`-cmoe`, `-ot`) are matched against the parent name, since shard names
  are synthetic.
- Local shard in host RAM: op runs on CPU at batch < 32; at batch >= 32 the existing
  offload branch reassigns the op to the local GPU and copies only the used experts.
  The used-expert copy works per shard unchanged, because `input->ne[2]` is still
  `n_expert` and `input->nb[2]` is the shard's per-expert stride.
- Remote shard: lives in the RPC device's buffer (server VRAM, or server RAM when the
  server has no GPU). The op runs on the RPC backend; no weight crosses the network at
  inference. RPC exposes no host buft and no `offload_op`
  (`ggml-rpc.cpp:2166`, `:2193`, `:2213-2235`), so there is no server-side host to
  VRAM streaming expressible from the client.
- Owner affinity: the existing offload loop picks the first GPU
  (`ggml-backend.cpp:971`). This is correct with one local GPU. With multiple local
  GPUs the target must be matched to the shard's host buft device.

## Load path

New loader structs:

```cpp
struct llama_tensor_shard_segment { int64_t extent; uint32_t repeat; };

struct llama_tensor_shard {
    const llama_tensor_weight * w;
    int                                       axis;
    int64_t                                   blck_size;
    std::vector<llama_tensor_shard_segment>   segments;      // parent {extent, repeat}
    std::vector<int64_t>                      shard_extent;  // this shard's extent per segment
};

std::unordered_map<const ggml_tensor *, llama_tensor_shard> shard_map;
```

```cpp
ggml_tensor * create_tensor_shard(
    const llama_hparams &          hparams,
    const buft_list_t *            buft_list,
    const LLM_TN_IMPL &            tn,        // parent name, drives metadata and overrides
    std::initializer_list<int64_t> ne,        // full parent shape
    int                            axis,
    int64_t                        low,
    int64_t                        high,
    const std::vector<llama_tensor_shard_segment> & segments,
    int                            shard_idx);
```

Steps:

1. Look up the parent metadata with `require_tensor_meta(tn.str())`.
2. Set `ne[axis]` to the shard extent; keep the type.
3. Choose the buft with the existing override logic.
4. Create the tensor in the ctx for that buft, with a synthetic unique name.
5. Record the `llama_tensor_shard` descriptor.
6. Count the parent once in `n_created`, not once per shard, because
   `done_getting_tensors` checks `n_created == n_tensors`
   (`llama-model-loader.cpp:1367-1375`). Do not add shard bytes to `size_data`.

Loading in `load_all_data`: look up `shard_map` by tensor before `get_weight()`, and
copy the slice with `ggml_backend_tensor_set` (axis 1, contiguous run per expert) or
`ggml_backend_tensor_set_2d` (axis 0, strided). Stream per expert or per segment rather
than staging a whole parent. The merged `gate_up` case iterates its two segments and
places `[gate_shard; up_shard]` sequentially. The first cut loads shards synchronously
and skips the async-upload and lazy fast paths.

## Graph path

Shape:

```
routing (once): cur -> selected_experts, weights
for each shard d:
    up_d   = mul_mat_id(up_exps_shards[d],   cur, selected_experts) * up_exps_s_shards[d]
    gate_d = mul_mat_id(gate_exps_shards[d], cur, selected_experts) * gate_exps_s_shards[d]
    act_d  = swiglu(gate_d, up_d)                 // [n_ff_d, n_expert_used, n_tokens]
act = concat_d act_d                              // gather, [n_ff, ...]
for each shard d:
    out_d  = mul_mat_id(down_exps_shards[d], act, selected_experts)  // [n_embd_d, ...]
experts = concat_d out_d                          // [n_embd, n_expert_used, n_tokens]
experts = experts * weights
moe_out = sum over n_expert_used views
```

Why the down split moved to `n_embd`: splitting `ffn_down_exps` on `n_ff` splits the
quantized dimension, so the granularity is one quant block. For `n_ff_exp = 512` and
`QK_K = 256` that is only two blocks, which collapses every ratio to 1:1. Splitting on
`n_embd` (and on `n_ff` for gate/up) touches no quant block, so the ratio is arbitrary.
The price is that every device needs the full activation, hence the `concat` gather, and
the output is sharded along `n_embd` and concatenated on the way out.

Factorization to avoid duplicating the shared function:

- Extract the routing prologue (`src/llama-graph.cpp:2019-2165`) into
  `build_moe_routing(...)` returning `{cur, selected_experts, weights}`.
- Extract the FFN core (`:2167-2318`) into `build_moe_experts_one(...)` with a name
  suffix for `cb`.
- Extract the per-expert scale multiply from `build_lora_mm_id` into
  `apply_expert_scale(t, w_s, selected_experts)`.
- `build_moe_ffn` becomes routing + one core call + post.
- Add `build_moe_ffn_sharded(...)` for the shard path.

Post-combine then pins the reduce output to the layer's device, mirroring
`src/llama-context.cpp:2533-2543`:

```cpp
for (auto & backend : backends) {
    if (ggml_backend_get_device(backend.get()) == model.dev_layer(il)) {
        ggml_backend_sched_set_tensor_backend(sched, acc, backend.get());
    }
}
```

## RPC and latency

Per MoE layer: mirror the activation to the other expert devices, run the per-shard
matmuls, one remote-to-owner partial transfer, local `ggml_add`. Two serialized network
waits per layer, mandatory because the activation is needed by every shard and the
reduce must come back to the owner. No butterfly. Weights are only transferred at model
load.

## Scope and call sites

Arch allow-list: `LLM_ARCH_QWEN35MOE`, `LLM_ARCH_QWEN4EXP`. Both share
`create_tensor_gate_up_exps` and the same MoE FFN shape (`LLM_FFN_SILU`, `norm_w=true`,
`SOFTMAX`, no expert biases).

- Load: `qwen35moe.cpp:96`, `:123`; `qwen4exp.cpp:250` and the
  `create_tensor_gate_up_exps` path.
- Graph: `qwen35moe.cpp:499`, `:679`; `qwen4exp.cpp:978`.

Boundary logic: factor only the expert FFN cases out of
`llama_meta_device_get_split_state` (gate/up `{{ne[axis], 1}}`, merged `gate_up`
`{{n_ff_exp, 2}}`, down axis 0, granularity `lcm(blck_size, 128)`).

## Verification checklist

- Scheduler places the `ggml_add` on the owner and adds no extra copy. Check with
  `GGML_SCHED_DEBUG=1`.
- Offload path copies the correct expert bytes for a host shard; check `nb[2]` and
  `ne[2]` assumptions on real tensors.
- `n_created`, `size_data`, `done_getting_tensors` stay consistent across shards.
- `-cmoe` / `-ot` overrides still select the intended buft for shards.
- Merged `gate_up` shard load places `[gate_shard; up_shard]` correctly.
- No duplicate routing in the graph (one softmax/topk per layer).
- Server still uses `GRAPH_RECOMPUTE` with N matmuls instead of one.

## Non-goals and known gaps

- `--fit` integration.
- LoRA on expert weights (`build_lora_mm_id` looks up adapters by weight pointer and
  will miss shards; assert or ignore for the PoC).
- `llama-model-saver` reconstruction of sharded experts; saving a sharded model throws.
- Expert-parallel (split by expert dimension).
- Server-side host to VRAM streaming for RPC devices.
- Multi-local-GPU owner affinity in the offload target.

## Backlog

- Parameter to place experts on the dense side for the first N layers, similar to `-ncmoe`.
  Proposed `--n-dense-moe-experts N` or reuse the override mechanism: for the first N layers
  keep the expert weights on the device that owns the layer's dense weights (single owner,
  no split) instead of splitting them across the expert devices. This lets a small part of
  the model stay fully in device memory while the rest keeps the split and host placement.

## Implementation status

Done (PoC, qwen35moe and qwen4exp):

- `llama_model_params.tensor_split_experts` and `-tse` / `--tensor-split-experts`.
- `llama_layer` shard arrays and `llama_model_loader::create_tensor_shard` / `load_shard_data`.
- `create_tensor_exps` and wiring in `create_tensor_gate_up_exps`; qwen35moe and qwen4exp call it
  for `ffn_down_exps` and gate/up.
- `build_moe_ffn_sharded`: computes the routing once, runs gate/up per shard, gathers the
  activation with `ggml_concat`, runs down per shard on `n_embd`, concatenates the output
  slices, and the `ffn_moe_out_sharded` callback pins the result to the layer device. Only
  supports `SOFTMAX` gating, `LLM_FFN_SILU`, and no biases for now.
- Validation: `-tse` requires `LLAMA_SPLIT_MODE_LAYER`, the allow-listed archs, at least two
  nonzero shares, and no expert scales.
- Host RAM placement: an expert shard goes to the device host buffer (`ROCm_Host`) when its
  device also holds dense weights and exposes a host buffer type. Remote RPC shards stay on
  the server device. At batch `< 32` a host shard runs on the CPU; at batch `>= 32` the
  existing `op_offload` path streams the used experts to the owning GPU.
- The model saver refuses to save a model loaded with `-tse` (the shard names would not
  reproduce the parent tensor).

Verified on `Qwen3.6-35B-A3B-UD-IQ1_M.gguf` with `ROCm0` plus a local `ggml-rpc-server` as
`RPC0`: `-ts 1,0 -tse 1,3` and `-ts 1,1 -tse 1,3` produce byte-identical output to the
non-split baseline for both decode and a 1700-token prefill.

Known limitations of the PoC:

- The split is on the non-quantized axes, so the ratio is arbitrary. Verified with
  `-tse 1,10`: `n_ff` splits 46/466 and `n_embd` splits 186/1862, and the output is
  byte-identical to the non-split baseline.
- Routing is computed once per layer in the sharded path.
- The merged `ffn_gate_up_exps` shard load path is implemented but untested on this machine.
- The activation gather and the output concatenation add serialized work per layer; their
  placement relies on the scheduler inserting the cross-device copies.
- The host offload path (RAM to VRAM at batch >= 32) uses the existing scheduler path and is
  not exercised on a real second machine here.
