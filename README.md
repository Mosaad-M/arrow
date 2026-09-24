# arrow — Pure-Mojo Apache Arrow IPC

A pure-Mojo implementation of the Apache Arrow IPC stream and file (Feather v2)
formats.  No C dependencies, no FFI — just Mojo.

## Features

- **IPC framing** — encode/decode Arrow IPC messages (continuation marker,
  metadata, body, end-of-stream)
- **Arrow types** — Null, Int8/16/32/64, Float32/64, Bool, Utf8, Binary
- **Schema messages** — encode/decode Arrow Schema IPC messages (field names,
  types, nullability, endianness)
- **RecordBatch messages** — encode/decode RecordBatch IPC messages (FieldNode
  + Buffer struct vectors via FlatBuffers)
- **Typed column encoding** — `ArrowArray` with validity bitmaps, offset
  buffers, and value buffers; full encode/decode roundtrip
- **Feather v2 / Arrow IPC file format** — `encode_arrow_file` /
  `decode_arrow_file` with proper magic bytes and FlatBuffers Footer
- **CSV → Feather** — `csv_to_feather` reads a CSV, infers types (Int64,
  Float64, Utf8), and writes a standards-compliant `.feather` file
- **Feather → CSV** — `feather_to_csv` reconstructs a CSV from any
  `.feather` file produced by this library (or by pyarrow)

## Quick start

```bash
pixi install                        # install dependencies (flatbuffers 1.0.2)
pixi run test-arrow                 # 59 Arrow IPC tests
pixi run test-csv-arrow             # 7 CSV converter tests
```

End-to-end smoke test:

```bash
echo "id,name,score
1,alice,9.5
2,bob,8.0" > /tmp/test.csv

pixi run csv-to-feather /tmp/test.csv /tmp/test.feather

python3 -c "import pyarrow.feather as f; print(f.read_table('/tmp/test.feather'))"
```

**Real pyarrow interop: verified working end-to-end.** A real
`pyarrow.feather.read_table()` call against a file written by this repo's
`encode_arrow_file` now succeeds and returns correct data — confirmed
directly, not just via this repo's own self-roundtrip tests. Getting here
took three separate, real bugs, each only visible once the previous one
was fixed:

1. **Magic bytes** (fixed): `encode_arrow_file`'s header/trailer magic
   are now spec-correct (8-byte padded header, 6-byte unpadded trailer —
   older versions wrote 8 bytes for both, which real Arrow readers reject
   outright with `ArrowInvalid: Not an Arrow file`).
2. **Footer FlatBuffer verification** (fixed, in the `flatbuffers`
   dependency, `>=1.1.1`): the writer and reader agreed with each other on
   an inverted `soffset` sign convention, which every self-roundtrip test
   in both repos passed while real Arrow C++'s stricter FlatBuffers
   verifier rejected it (`OSError: Verification of flatbuffer-encoded
   Footer failed.`). Fixed at the source; see that repo's `v1.1.1` release
   notes.
3. **RecordBatch body/meta length in the Footer's Block entries** (fixed):
   these were computed from an internal helper's return value that
   already pointed past the body, producing `bodyLength: 0` regardless of
   the batch's real size. This repo's own `decode_arrow_file` never
   noticed, since it re-derives batch boundaries by parsing each message
   directly rather than trusting these fields — but real Arrow does trust
   and cross-check them (`ArrowInvalid: Mismatching body length for IPC
   message`).

Each fix has a regression test that checks the actual external invariant
that was violated, not just this repo's own self-consistency — that
distinction is exactly why these three bugs went unnoticed for as long as
they did.

**Still not fixed**: the `pixi run csv-to-feather` task above errors with
`mojo: error: module does not define a 'main' function` — `csv_arrow.mojo`
has no `main()`, so the quickstart as written doesn't run as-is. Call
`csv_to_feather(csv_path, feather_path)` from a small Mojo script instead,
or see `test_csv_arrow.mojo`'s `test_csv_to_feather_file` for a working
call site. Unrelated to the interop fixes above, not fixed here.

### Arrow IPC (schema + record batches)

```mojo
from arrow import (
    ArrowSchema, ArrowField, ArrowType,
    ArrowArray, RecordBatch,
    encode_arrow_file, decode_arrow_file,
)

# Build a schema
var fields = List[ArrowField]()
fields.append(ArrowField("id",    ArrowType.int_(64, True), False))
fields.append(ArrowField("score", ArrowType.float_(2),      True))
var schema = ArrowSchema(fields)

# Build a batch
var id_bytes = List[UInt8]()
# ... populate 8-byte LE values for each id ...
var id_col = ArrowArray(ArrowType.int_(64, True), 2, 0,
                        List[UInt8](), List[UInt8](), id_bytes)
var cols = List[ArrowArray]()
cols.append(id_col.copy())
var batch = RecordBatch(Int64(2), cols)
var batches = List[RecordBatch]()
batches.append(batch.copy())

# Write Feather v2 file
var file_bytes = encode_arrow_file(schema, batches)

# Read it back
var result   = decode_arrow_file(file_bytes)
var schema2  = result[0].copy()
var batches2 = result[1].copy()
```

### CSV → Feather

```mojo
from csv_arrow import csv_to_feather, feather_to_csv

csv_to_feather("data.csv",    "data.feather")
feather_to_csv("data.feather", "out.csv")
```

## Project structure

| File | Contents |
|------|----------|
| `arrow.mojo` | Arrow IPC framing, type encoding, Schema, RecordBatch, Feather v2 |
| `csv_arrow.mojo` | CSV reader, type inference, `csv_to_feather`, `feather_to_csv` |
| `test_arrow.mojo` | 59 unit + adversarial tests for `arrow.mojo` |
| `test_csv_arrow.mojo` | 7 tests for `csv_arrow.mojo` |
| `bench_arrow.mojo` | Micro-benchmarks |

## Dependencies

- [flatbuffers](https://github.com/Mosaad-M/flatbuffers) `>=1.0.2` —
  pure-Mojo FlatBuffers encoder/decoder (managed by
  [mojo-pkg](https://github.com/Mosaad-M/mojo-pkg))

## License

MIT
