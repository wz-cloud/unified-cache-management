# Qwen3.5 KVCacheConfig passed to connector

This is the `KVCacheConfig` shape vLLM builds from the supplied HuggingFace
config, following the current local code in this workspace.

Important runtime-dependent fields:

- `num_blocks` is not determined by the HF config. It is computed as
  `available_memory // bytes_per_block`, then possibly overridden by
  `--num-gpu-blocks-override`.
- `kv_cache_layout` is resolved at runtime from backend support,
  `VLLM_KV_CACHE_LAYOUT`, and connector preference.
- Formulas below use `tp = tensor_parallel_size` and assume no speculative
  decoding (`num_speculative_tokens = 0`). With speculative decoding, the GDN
  conv state length becomes `3 + num_speculative_tokens`.

## Resolved cache knobs for this model

With the default vLLM flow for Qwen3.5:

- `cache_dtype = auto` -> attention KV dtype is `torch.bfloat16`
- `mamba_ssm_cache_dtype = auto` is updated from HF
  `text_config.mamba_ssm_dtype = "float32"`
- prefix caching is enabled by default, so `mamba_cache_mode = "align"`
- Qwen3.5 rejects `mamba_cache_mode = "all"`
- initial `block_size = 16`, but hybrid block/page alignment raises it to:

```text
attention_per_token_bytes = (num_key_value_heads / tp) * (head_dim + head_dim) * 2
                          = (4 / tp) * (256 + 256) * 2
                          = 4096 / tp

gdn_raw_page_bytes = conv_state_bytes + recurrent_state_bytes
conv_state_shape   = (3, 10240 / tp) when VLLM_SSM_CONV_STATE_LAYOUT is default "SD"
conv_state_dtype   = torch.bfloat16
conv_state_bytes   = 3 * (10240 / tp) * 2 = 61440 / tp

recurrent_state_shape = (48 / tp, 128, 128)
recurrent_state_dtype = torch.float32
recurrent_state_bytes = (48 / tp) * 128 * 128 * 4 = 3145728 / tp

gdn_raw_page_bytes = 3207168 / tp

aligned block_size = 16 * ceil(gdn_raw_page_bytes / (16 * attention_per_token_bytes))
                   = 16 * ceil(3207168 / 65536)
                   = 784

mamba_block_size = 784
mamba_page_size_padded = attention_page_size = 784 * (4096 / tp) = 3211264 / tp
```

So the full-attention and GDN/linear-attention page sizes are equal after
padding:

```text
FullAttentionSpec.page_size_bytes = 3211264 / tp
MambaSpec.real_page_size_bytes    = 3207168 / tp
MambaSpec.page_size_bytes         = 3211264 / tp
```

For `tp = 1`, each layer page is `3,211,264` bytes and each GDN page contains
`4,096` bytes of padding.

## Layer classification from `layer_types`

Full attention layers are every fourth layer:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

All other 48 layers are `linear_attention` and become
`QwenGatedDeltaNetAttention`, whose `get_kv_cache_spec()` returns `MambaSpec`.

For `Qwen3_5ForConditionalGeneration`, the layer module names passed through
the KV cache spec path use this prefix:

```text
model.language_model.model.layers.<layer_idx>.linear_attn
model.language_model.model.layers.<layer_idx>.self_attn
```

## `kv_cache_groups`

Because there are 48 GDN layers and 16 full-attention layers, vLLM's hybrid
grouping chooses `group_size = 16`. The linear layers are split by stride into
three groups, and the full-attention layers form one group.

```yaml
kv_cache_groups:
  - group_id: 0
    kv_cache_spec:
      type: MambaSpec
      block_size: 784
      page_size_bytes: 3211264 / tp
      real_page_size_bytes: 3207168 / tp
      page_size_padded: 3211264 / tp
      mamba_type: GDN_ATTN
      mamba_cache_mode: align
      shapes:
        - [3, 10240 / tp]        # conv state, default SD layout
        - [48 / tp, 128, 128]    # recurrent state
      dtypes:
        - torch.bfloat16
        - torch.float32
      num_speculative_blocks: 0
      num_prefill_checkpoint_blocks: 0
      num_heads: 1
      tokens_per_state: -1
      tp_replicated: false
    layer_names:
      - model.language_model.model.layers.0.linear_attn
      - model.language_model.model.layers.4.linear_attn
      - model.language_model.model.layers.8.linear_attn
      - model.language_model.model.layers.12.linear_attn
      - model.language_model.model.layers.16.linear_attn
      - model.language_model.model.layers.20.linear_attn
      - model.language_model.model.layers.24.linear_attn
      - model.language_model.model.layers.28.linear_attn
      - model.language_model.model.layers.32.linear_attn
      - model.language_model.model.layers.36.linear_attn
      - model.language_model.model.layers.40.linear_attn
      - model.language_model.model.layers.44.linear_attn
      - model.language_model.model.layers.48.linear_attn
      - model.language_model.model.layers.52.linear_attn
      - model.language_model.model.layers.56.linear_attn
      - model.language_model.model.layers.60.linear_attn

  - group_id: 1
    kv_cache_spec: same as group 0
    layer_names:
      - model.language_model.model.layers.1.linear_attn
      - model.language_model.model.layers.5.linear_attn
      - model.language_model.model.layers.9.linear_attn
      - model.language_model.model.layers.13.linear_attn
      - model.language_model.model.layers.17.linear_attn
      - model.language_model.model.layers.21.linear_attn
      - model.language_model.model.layers.25.linear_attn
      - model.language_model.model.layers.29.linear_attn
      - model.language_model.model.layers.33.linear_attn
      - model.language_model.model.layers.37.linear_attn
      - model.language_model.model.layers.41.linear_attn
      - model.language_model.model.layers.45.linear_attn
      - model.language_model.model.layers.49.linear_attn
      - model.language_model.model.layers.53.linear_attn
      - model.language_model.model.layers.57.linear_attn
      - model.language_model.model.layers.61.linear_attn

  - group_id: 2
    kv_cache_spec: same as group 0
    layer_names:
      - model.language_model.model.layers.2.linear_attn
      - model.language_model.model.layers.6.linear_attn
      - model.language_model.model.layers.10.linear_attn
      - model.language_model.model.layers.14.linear_attn
      - model.language_model.model.layers.18.linear_attn
      - model.language_model.model.layers.22.linear_attn
      - model.language_model.model.layers.26.linear_attn
      - model.language_model.model.layers.30.linear_attn
      - model.language_model.model.layers.34.linear_attn
      - model.language_model.model.layers.38.linear_attn
      - model.language_model.model.layers.42.linear_attn
      - model.language_model.model.layers.46.linear_attn
      - model.language_model.model.layers.50.linear_attn
      - model.language_model.model.layers.54.linear_attn
      - model.language_model.model.layers.58.linear_attn
      - model.language_model.model.layers.62.linear_attn

  - group_id: 3
    kv_cache_spec:
      type: FullAttentionSpec
      block_size: 784
      page_size_bytes: 3211264 / tp
      num_kv_heads: max(1, 4 // tp)
      head_size: 256
      head_size_v: 256
      dtype: torch.bfloat16
      kv_quant_mode: NONE
      page_size_padded: null
      num_head_slots: null
      state_content_bytes: null
      tokens_per_state: 1
      sliding_window: null
      attention_chunk_size: null
      non_causal: false
    layer_names:
      - model.language_model.model.layers.3.self_attn
      - model.language_model.model.layers.7.self_attn
      - model.language_model.model.layers.11.self_attn
      - model.language_model.model.layers.15.self_attn
      - model.language_model.model.layers.19.self_attn
      - model.language_model.model.layers.23.self_attn
      - model.language_model.model.layers.27.self_attn
      - model.language_model.model.layers.31.self_attn
      - model.language_model.model.layers.35.self_attn
      - model.language_model.model.layers.39.self_attn
      - model.language_model.model.layers.43.self_attn
      - model.language_model.model.layers.47.self_attn
      - model.language_model.model.layers.51.self_attn
      - model.language_model.model.layers.55.self_attn
      - model.language_model.model.layers.59.self_attn
      - model.language_model.model.layers.63.self_attn
```

Each `KVCacheGroupSpec` also has:

```yaml
is_eagle_group: false
enable_kv_transfer: true
```

## Top-level `KVCacheConfig`

```yaml
KVCacheConfig:
  num_blocks: <runtime: available_memory // bytes_per_block, or --num-gpu-blocks-override>
  kv_cache_layout: <runtime resolved layout name>
  prefix_cache_retention_interval: <cache_config.prefix_cache_retention_interval>
  kv_cache_groups: <the four groups above>
  kv_cache_tensors: <derived from resolved layout and num_blocks>
```

Since all four groups have 16 layers with the same padded page size:

```text
bytes_per_block = 16 * (3211264 / tp)
                = 51380224 / tp
```

For `tp = 1`, the block pool consumes `51,380,224` bytes per block. Therefore:

```text
num_blocks = available_memory // 51380224
```

## `kv_cache_tensors`

The exact strides depend on the resolved `kv_cache_layout`. vLLM calls
`compute_layout_strides()` for each group/spec and creates one
`KVCacheTensor` per group here, because each group contains one concrete spec.

For the common layer-compact layouts (`LBHNC` / `LBNHC`) the tensor entries are:

```yaml
kv_cache_tensors:
  - layers: <group 0 layer_names>
    size: bytes_per_block * num_blocks
    layer_stride: (3211264 / tp) * num_blocks
    block_stride: 3211264 / tp
    offset: 0

  - layers: <group 1 layer_names>
    size: bytes_per_block * num_blocks
    layer_stride: (3211264 / tp) * num_blocks
    block_stride: 3211264 / tp
    offset: 0

  - layers: <group 2 layer_names>
    size: bytes_per_block * num_blocks
    layer_stride: (3211264 / tp) * num_blocks
    block_stride: 3211264 / tp
    offset: 0

  - layers: <group 3 layer_names>
    size: bytes_per_block * num_blocks
    layer_stride: (3211264 / tp) * num_blocks
    block_stride: 3211264 / tp
    offset: 0
```

The groups intentionally alias the same backing allocation from offset 0; a
physical block ID is owned by one group at a time.

For block-outermost layouts such as `BLHNC` / `BLNHC`, `block_stride` becomes
`bytes_per_block`, and `layer_stride` / `offset` are the corresponding
layout-derived byte strides.

## Connector-visible summary

The HLA connector will see:

```text
len(kv_cache_config.kv_cache_groups) = 4
group 0: MambaSpec, block_size=784, 16 linear_attn layers
group 1: MambaSpec, block_size=784, 16 linear_attn layers
group 2: MambaSpec, block_size=784, 16 linear_attn layers
group 3: FullAttentionSpec, block_size=784, 16 self_attn layers
transfer_group_ids = (0, 1, 2, 3)
has_mamba_layers = true
needs_kv_cache_zeroing = true
```
