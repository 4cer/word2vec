# Checkpoint Format: `.wght`

All integers are little-endian. The format is self-describing: the index
sections are always read sequentially and fully into memory before any tensor
I/O begins. Random access into individual tensors is then driven by the
in-memory index using the absolute file offsets stored in the tensor mappings.

## Layout

```
File header
───────────────────────────────────────────────
  4B   magic           b'WGHT'  (ASCII)
  4I   version         currently 2
  4I   layer_count     number of per-layer index records that follow
  4I   tensor_count    number of tensor mapping records in the file
  4I   mapping_count   number of in/out edge records in the file
  8B   hash_address    uint64 absolute file offset of the SHA-256 hash
                       (equivalently: total file size minus 32)

Layer index (repeated layer_count times)
───────────────────────────────────────────────
  4I   layer_index     position in model.layers (0-based)
  4I   layer_type      LayerType.value (see com/enums.py)
  4I   precision       LayerPrecision.value (see com/enums.py)
  4B   purpose         bitmask; bit positions correspond to
                       LayerPurpose.value (see com/enums.py)

Layer in/out mappings (repeated mapping_count times)
───────────────────────────────────────────────
  4I   layer_index     layer which outputs a tensor
  4I   layer_index     layer which accepts the tensor

Tensor mappings (repeated tensor_count times)
───────────────────────────────────────────────
  4I   layer_index     layer the tensor belongs to
  4I   tensor_index    position within the layer's tensor list
                       e.g. Linear weights = 0, biases = 1
  4I   ndim            number of shape dimensions
  ndim × 4I            shape dimensions (C-contiguous order)
  8B   offset          uint64 absolute file offset of the tensor's raw data

Tensor entries (repeated tensor_count times)
───────────────────────────────────────────────
  raw weight data, dtype and size determined by the owning layer's
  precision field and the shape recorded in the tensor mapping
  (ndim[0] × ndim[1] × … elements)

SHA-256 hash
───────────────────────────────────────────────
  32B  hash            SHA-256(file bytes [0 : hash_address])
```

### Notes

- Activation layers (`SoftMax`, `ReLU`, `Sigmoid`) have no tensors and
  therefore contribute no tensor mapping records. They are still present in the
  layer index so that `layer_index` values remain contiguous and can be used
  for direct lookup.

- Purpose filtering is applied at write time: a layer is serialised only when
  the intersection of its purpose mask and the active filter mask is non-empty.
  `layer_count` and `tensor_count` in the header reflect only the layers
  actually written. The default filter is `INFERENCE` only; every layer carries
  `INFERENCE` by default unless explicitly overridden (e.g. training-only
  dropout layers).

- At load time the same purpose filter is applied: only layers whose purpose
  mask intersects the filter are deserialised. The loader reads the entire index
  into memory first, applies the filter, then seeks to and reads only the
  surviving tensors.

- The shape recorded in the tensor mapping is the authoritative shape. The
  loader uses it together with the layer's `precision` field to determine the
  exact byte count to read: `product(shape) × sizeof(precision)`.

## Enum reference

All enum tables reflect `com/enums.py` as of format version 2. New members are
always appended to the end of each enum; existing ordinals are stable and will
not be reassigned.

### LayerType

Serialized as its integer `.value`.

| Serialized value | Name             |
|------------------|------------------|
| 0                | LINEAR           |
| 1                | RELU             |
| 2                | SOFTMAX          |
| 3                | SIGMOID          |
| 4                | AVERAGINGLINEAR  |

### LayerPrecision

`LayerPrecision` members hold numpy dtype objects as their `.value`, which
carry no natural integer ordinal. The serialized value is therefore the
**0-based declaration order** in the enum. This is stable under the
append-only guarantee: commented-out reserved slots (`BF16`, `INT4`, `Q1BT`)
still consume an ordinal and must be accounted for if ever activated.

| Serialized value | Name  | numpy dtype  | Bytes/element | Notes                              |
|------------------|-------|--------------|---------------|------------------------------------|
| 0                | FP64  | `np.float64` | 8             | Double precision; scientific use   |
| 1                | FP32  | `np.float32` | 4             | Default training precision         |
| ~~2~~            | BF16  | —            | 2             | Not supported (numpy lacks BF16)   |
| 3                | FP16  | `np.float16` | 2             | Half precision; numerically unstable for training |
| 4                | INT8  | `np.int8`    | 1             | Quantized arithmetic               |
| ~~5~~            | INT4  | —            | 0.5           | Not supported (numpy lacks INT4)   |
| ~~6~~            | Q1BT  | —            | —             | Not supported                      |

Unsupported precisions are reserved ordinals only. A loader encountering one
should raise an error rather than attempt to read tensor data.

### LayerPurpose

Serialized as a **bitmask** in the `purpose` field of the layer index record:
bit *N* is set when the `LayerPurpose` member with `.value` *N* applies to
that layer. A layer may carry multiple purposes simultaneously.

| `.value` | Bitmask      | Name      | Description                                      |
|----------|--------------|-----------|--------------------------------------------------|
| 0        | `0x00000001` | INFERENCE | Required for forward pass at inference time. All layers carry this by default unless explicitly overridden. |
| 1        | `0x00000002` | TRAINING  | Required for training and/or fine-tuning.        |
| 2        | `0x00000004` | DEBUGGING | Required for debugging and profiling tasks.      |

Example: a layer tagged `INFERENCE | TRAINING` has purpose mask `0x00000003`.

The 4-byte field supports up to 32 distinct purposes.

### LayerFilter presets

`LayerFilter` presets are convenience aliases for common purpose-mask sets
used at (de)serialization call sites. They are **not stored in the file**.
A layer is included when the intersection of its purpose mask and the active
filter set is non-empty. The default filter when none is specified is
`INFERENCE_ONLY`.

| Name            | Included purposes              | Typical use                                       |
|-----------------|--------------------------------|---------------------------------------------------|
| INFERENCE_ONLY  | INFERENCE                      | Deployment; discard all training state            |
| DEBUGGING       | INFERENCE, DEBUGGING           | Profiling runs; excludes training-only layers     |
| FULL_CHECKPOINT | INFERENCE, TRAINING, DEBUGGING | Training continuation or fine-tuning              |

## Concrete example: CBOW with V = 500, N = 25

Parameters:
- Vocabulary size `V = 500`
- Hidden (embedding) size `N = V // 20 = 25`
- Precision `FP32` (serialized value = `1`) for all layers
- No in/out mapping edges (`mapping_count = 0`)

The model has three layers in registration order. In CBOW the embedding
weights in `AVERAGINGLINEAR` are the primary inference artefact; the
projection back to vocabulary space (`LINEAR` + `SOFTMAX`) is only needed
to compute loss during training. `AVERAGINGLINEAR` is also tagged `TRAINING`
because it receives gradient updates, though this is somewhat arbitrary — the
tag mainly ensures it is included in a `FULL_CHECKPOINT` alongside the other
training layers.

| Index | Type            | Tensors | Shape     | Purpose               | Purpose mask |
|-------|-----------------|---------|-----------|-----------------------|--------------|
| 0     | AVERAGINGLINEAR | 1       | (500, 25) | INFERENCE \| TRAINING | `0x00000003` |
| 1     | LINEAR          | 1       | (25, 500) | TRAINING              | `0x00000002` |
| 2     | SOFTMAX         | 0       | —         | TRAINING              | `0x00000002` |

Under the default `INFERENCE_ONLY` filter only layer 0 is serialized.
A `FULL_CHECKPOINT` serializes all three.

File size breakdown depends on the filter applied at serialization time.

**`INFERENCE_ONLY` (default) — layer 0 only:**

| Section               | Size                                         |
|-----------------------|----------------------------------------------|
| File header           | 4 + 4 + 4 + 4 + 4 + 8 = **28 B**            |
| Layer index (×1)      | 1 × (4+4+4+4) = **16 B**                    |
| In/out mappings       | 0 × 8 = **0 B**                             |
| Tensor mappings (×1)  | 1 × (4+4+4 + 2×4 + 8) = **28 B**           |
| Tensor data           | 50 000 = **50 000 B**                        |
| SHA-256 hash          | **32 B**                                     |
| **Total**             | **50 104 B** (≈ 48.9 KB)                    |

**`FULL_CHECKPOINT` — all three layers:**

| Section               | Size                                         |
|-----------------------|----------------------------------------------|
| File header           | 4 + 4 + 4 + 4 + 4 + 8 = **28 B**            |
| Layer index (×3)      | 3 × (4+4+4+4) = **48 B**                    |
| In/out mappings       | 0 × 8 = **0 B**                             |
| Tensor mappings (×2)  | 2 × (4+4+4 + 2×4 + 8) = **56 B**           |
| Tensor data           | 50 000 + 50 000 = **100 000 B**              |
| SHA-256 hash          | **32 B**                                     |
| **Total**             | **100 164 B** (≈ 97.8 KB)                   |

The hex layout below shows the `FULL_CHECKPOINT` case.

Hex layout (all integers little-endian):

```
Offset    Bytes                         Field
─────────────────────────────────────────────────────────────────────────────
── File header ──────────────────────────────────────────────────────────────
0x0000    57 47 48 54                   magic           = 'WGHT'
0x0004    02 00 00 00                   version         = 2
0x0008    03 00 00 00                   layer_count     = 3
0x000C    02 00 00 00                   tensor_count    = 2
0x0010    00 00 00 00                   mapping_count   = 0
0x0014    24 87 01 00 00 00 00 00       hash_address    = 0x00018724 (100132)

── Layer index ──────────────────────────────────────────────────────────────
0x001C    00 00 00 00                   layer_index     = 0
0x0020    04 00 00 00                   layer_type      = 4  (AVERAGINGLINEAR)
0x0024    01 00 00 00                   precision       = 1  (FP32)
0x0028    03 00 00 00                   purpose         = 0x00000003 (INFERENCE | TRAINING)

0x002C    01 00 00 00                   layer_index     = 1
0x0030    00 00 00 00                   layer_type      = 0  (LINEAR)
0x0034    01 00 00 00                   precision       = 1  (FP32)
0x0038    02 00 00 00                   purpose         = 0x00000002 (TRAINING)

0x003C    02 00 00 00                   layer_index     = 2
0x0040    02 00 00 00                   layer_type      = 2  (SOFTMAX)
0x0044    01 00 00 00                   precision       = 1  (FP32)
0x0048    02 00 00 00                   purpose         = 0x00000002 (TRAINING)

── Tensor mappings ──────────────────────────────────────────────────────────
0x004C    00 00 00 00                   layer_index     = 0  (AVERAGINGLINEAR)
0x0050    00 00 00 00                   tensor_index    = 0  (weights)
0x0054    02 00 00 00                   ndim            = 2
0x0058    F4 01 00 00                   shape[0]        = 500  (0x01F4)
0x005C    19 00 00 00                   shape[1]        = 25   (0x0019)
0x0060    84 00 00 00 00 00 00 00       offset          = 0x00000084 (132)

0x0068    01 00 00 00                   layer_index     = 1  (LINEAR)
0x006C    00 00 00 00                   tensor_index    = 0  (weights)
0x0070    02 00 00 00                   ndim            = 2
0x0074    19 00 00 00                   shape[0]        = 25   (0x0019)
0x0078    F4 01 00 00                   shape[1]        = 500  (0x01F4)
0x007C    D4 C3 00 00 00 00 00 00       offset          = 0x0000C3D4 (50132)

── Tensor data ──────────────────────────────────────────────────────────────
0x0084    … 50 000 bytes …              AVERAGINGLINEAR weights  (500×25 float32)
0xC3D4    … 50 000 bytes …              LINEAR weights           (25×500 float32)

── SHA-256 hash ─────────────────────────────────────────────────────────────
0x18724   … 32 bytes …                 SHA-256(file[0x0000 : 0x18724])
```