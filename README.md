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

## Usage

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
