# QuantizedGather: dequantize only the rows you read

## What it does

A language model starts by looking up one row of its embedding table per token. In a uint8 model, that table is stored as uint8 and turned back into float32 ("dequantized") before the lookup. Inference Engine dequantizes the whole table to read a few rows: for a large vocabulary that is more than a gigabyte of temporary float32, on every run. `QuantizedGather.Fuse` changes the model so that it picks the rows first and dequantizes only those. The result is the same.

## Similar approaches

- **llama.cpp's `get_rows`** reads rows straight from a quantized tensor and dequantizes only them.
- **bitsandbytes** and other quantization libraries keep embedding lookups on the quantized data for the same reason.

## How it works with Inference Engine

1. **Find the pattern.** A `DequantizeUint8` layer whose only user is a `Gather` on axis 0. That is how `ModelQuantizer.QuantizeWeights` leaves an embedding table.
2. **Replace it.** Both layers become one custom layer that gathers the rows of the uint8 data, then computes `(x - zeroPoint) * scale` for those rows only.
3. **Automatic.** `ModelFile` does this when it opens a file.

## Limits

Each row must have a multiple of 4 values, because Inference Engine packs 4 uint8 values into each 32-bit element.
