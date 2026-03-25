# Checkpoint Format: `.wght`

All integers are little-endian. The format is self-describing: every layer slot
is always present, so the position index is always aligned even for activation
layers that carry no weights.

## Layout

```
File header
───────────────────────────────────────────────
  4B   magic        b'WGHT'  (ASCII)
  4I   version      currently 1
  4I   layer_count  number of per-layer records that follow

Per-layer record  (repeated layer_count times)
───────────────────────────────────────────────
  4I   layer_index  position in model.layers (0-based)
  4I   layer_type   LayerType.value (see com/layer.py)
  4I   ndim         number of shape dimensions
                    0 for activation layers (no weights follow)

  if ndim > 0:
    ndim × 4I   shape dimensions (C-contiguous order)
    ndim>0      raw float32 weight data  (ndim[0] × ndim[1] × … floats)
```

Activation layers (`SoftMax`, `ReLU`, `Sigmoid`) write only the three-field
header with `ndim = 0` and no weight bytes. Their slot is still counted, so
`layer_index` values remain contiguous and can be used for direct lookup.

## LayerType values

| Value | Name             |
|-------|------------------|
| 0     | LINEAR           |
| 1     | RELU             |
| 2     | SOFTMAX          |
| 3     | SIGMOID          |
| 4     | AVERAGINGLINEAR  |

## Concrete example: CBOW with V = 500, N = 25, C = 4

Parameters:
- Vocabulary size `V = 500`
- Hidden (embedding) size `N = V // 20 = 25`  ← rule of thumb used in `test.py`
- Context words per sample `C = 2 × window_size = 4`

The model has three layers in registration order:

| Index | Type            | Weight shape   | Weight bytes          |
|-------|-----------------|----------------|-----------------------|
| 0     | AVERAGINGLINEAR | (500, 25)      | 500 × 25 × 4 = 50 000 |
| 1     | LINEAR          | (25, 500)      | 25 × 500 × 4 = 50 000 |
| 2     | SOFTMAX         | -              | 0                     |

Total weight bytes: 100 000. File size: 12 (header) + 3 × 12 (layer headers)
+ 2 × ndim × 4 (shape fields) + 100 000 (floats) = **100 060 bytes** (≈ 98 KB).

Hex layout of the first ~60 bytes:

```
Offset  Bytes                        Field
──────────────────────────────────────────────────────────────────────
0x00    57 47 48 54                  magic  'WGHT'
0x04    01 00 00 00                  version  = 1
0x08    03 00 00 00                  layer_count  = 3

── Layer 0: AVERAGINGLINEAR ──────────────────────────────────────────
0x0C    00 00 00 00                  layer_index  = 0
0x10    04 00 00 00                  layer_type   = 4  (AVERAGINGLINEAR)
0x14    02 00 00 00                  ndim         = 2
0x18    F4 01 00 00                  shape[0]     = 500  (0x1F4)
0x1C    19 00 00 00                  shape[1]     = 25   (0x19)
0x20    … 50 000 bytes of float32 weights …

── Layer 1: LINEAR ───────────────────────────────────────────────────
        00 00 00 01                  layer_index  = 1
        01 00 00 00                  layer_type   = 0  (LINEAR)
        02 00 00 00                  ndim         = 2
        19 00 00 00                  shape[0]     = 25
        F4 01 00 00                  shape[1]     = 500
        … 50 000 bytes of float32 weights …

── Layer 2: SOFTMAX ──────────────────────────────────────────────────
        02 00 00 00                  layer_index  = 2
        02 00 00 00                  layer_type   = 2  (SOFTMAX)
        00 00 00 00                  ndim         = 0  (no weights)
```