from std.time import perf_counter_ns
from flatbuffers import FlatBufferBuilder, FlatBuffersReader, write_i64_le, write_i32_le
from arrow import (
    ipc_pad8,
    encode_ipc_message,
    decode_ipc_message,
    encode_arrow_type,
    decode_arrow_type,
    ArrowType,
    ArrowField,
    ArrowSchema,
    encode_schema_message,
    decode_schema_message,
    ArrowArray,
    RecordBatch,
    encode_record_batch,
    decode_record_batch,
)


def bench(name: String, ns: Int, iters: Int):
    var per_op = ns / iters
    print("  " + name + ": " + String(per_op) + " ns/op")


def bench_ipc_pad8() raises:
    var iters = 1_000_000
    var t0 = perf_counter_ns()
    var acc = 0
    for i in range(iters):
        acc += ipc_pad8(i * 13)
    var elapsed = perf_counter_ns() - t0
    _ = acc
    bench("bench_ipc_pad8", elapsed, iters)


def bench_encode_ipc_message_1kb() raises:
    var iters = 10_000
    var meta = List[UInt8](capacity=32)
    for i in range(32):
        meta.append(UInt8(i))
    var body = List[UInt8](capacity=1024)
    for i in range(1024):
        body.append(UInt8(i % 256))

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var msg = encode_ipc_message(meta, body)
        _ = len(msg)
    var elapsed = perf_counter_ns() - t0
    bench("bench_encode_ipc_message_1kb", elapsed, iters)


def bench_decode_ipc_message_1kb() raises:
    # `metadata` must be a real FlatBuffers Message table (decode_ipc_message
    # reads bodyLength from slot 3), not arbitrary filler bytes — see
    # test_decode_schema_wrong_header_type in test_arrow.mojo for the same
    # hand-built-message pattern.
    var body = List[UInt8](capacity=1024)
    for i in range(1024):
        body.append(UInt8(i % 256))
    var b = FlatBufferBuilder(128)
    b.start_table()
    b.add_field_i16(0, Int16(4))
    b.add_field_u8(1, UInt8(2))
    b.add_field_i64(3, Int64(len(body)))
    var msg_off = b.end_table()
    var meta = b.finish(msg_off)
    var msg = encode_ipc_message(meta, body)

    var iters = 10_000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var result = decode_ipc_message(msg, 0)
        _ = len(result[0])
    var elapsed = perf_counter_ns() - t0
    bench("bench_decode_ipc_message_1kb", elapsed, iters)


def bench_type_encode_decode_int32() raises:
    var iters = 50_000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var b = FlatBufferBuilder(64)
        var t = ArrowType.int_(32, True)
        var result = encode_arrow_type(b, t)
        var disc = result[0]
        var off = result[1]
        var buf = b.finish(off)
        var r = FlatBuffersReader(buf)
        var decoded = decode_arrow_type(r, disc, r.root())
        _ = decoded.int_meta.bit_width
    var elapsed = perf_counter_ns() - t0
    bench("bench_type_encode_decode_int32", elapsed, iters)


def bench_type_encode_decode_utf8() raises:
    var iters = 50_000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var b = FlatBufferBuilder(64)
        var t = ArrowType.utf8()
        var result = encode_arrow_type(b, t)
        var disc = result[0]
        var off = result[1]
        var buf = b.finish(off)
        var r = FlatBuffersReader(buf)
        var decoded = decode_arrow_type(r, disc, r.root())
        _ = decoded.tag
    var elapsed = perf_counter_ns() - t0
    bench("bench_type_encode_decode_utf8", elapsed, iters)


def bench_encode_ipc_empty() raises:
    var iters = 100_000
    var meta = List[UInt8](capacity=8)
    for i in range(8):
        meta.append(UInt8(i))
    var body = List[UInt8]()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        var msg = encode_ipc_message(meta, body)
        _ = len(msg)
    var elapsed = perf_counter_ns() - t0
    bench("bench_encode_ipc_empty_body", elapsed, iters)


def bench_encode_schema_1field() raises:
    var iters = 10_000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var fields = List[ArrowField]()
        fields.append(ArrowField("id", ArrowType.int_(64, False), False))
        var schema = ArrowSchema(fields, Int16(0))
        var buf = encode_schema_message(schema)
        _ = len(buf)
    var elapsed = perf_counter_ns() - t0
    bench("bench_encode_schema_1field", elapsed, iters)


def bench_encode_schema_10fields() raises:
    var iters = 5_000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var fields = List[ArrowField]()
        fields.append(ArrowField("id", ArrowType.int_(64, False), False))
        fields.append(ArrowField("name", ArrowType.utf8(), True))
        fields.append(ArrowField("score", ArrowType.float_(2), False))
        fields.append(ArrowField("active", ArrowType.bool_(), False))
        fields.append(ArrowField("created", ArrowType.int_(64, True), False))
        fields.append(ArrowField("tag", ArrowType.binary(), True))
        fields.append(ArrowField("kind", ArrowType.int_(8, False), False))
        fields.append(ArrowField("ratio", ArrowType.float_(1), False))
        fields.append(ArrowField("label", ArrowType.utf8(), True))
        fields.append(ArrowField("flag", ArrowType.bool_(), False))
        var schema = ArrowSchema(fields, Int16(0))
        var buf = encode_schema_message(schema)
        _ = len(buf)
    var elapsed = perf_counter_ns() - t0
    bench("bench_encode_schema_10fields", elapsed, iters)


def bench_decode_schema_10fields() raises:
    var fields = List[ArrowField]()
    fields.append(ArrowField("id", ArrowType.int_(64, False), False))
    fields.append(ArrowField("name", ArrowType.utf8(), True))
    fields.append(ArrowField("score", ArrowType.float_(2), False))
    fields.append(ArrowField("active", ArrowType.bool_(), False))
    fields.append(ArrowField("created", ArrowType.int_(64, True), False))
    fields.append(ArrowField("tag", ArrowType.binary(), True))
    fields.append(ArrowField("kind", ArrowType.int_(8, False), False))
    fields.append(ArrowField("ratio", ArrowType.float_(1), False))
    fields.append(ArrowField("label", ArrowType.utf8(), True))
    fields.append(ArrowField("flag", ArrowType.bool_(), False))
    var schema = ArrowSchema(fields, Int16(0))
    var buf = encode_schema_message(schema)

    var iters = 5_000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var result = decode_schema_message(buf, 0)
        _ = len(result[0].fields)
    var elapsed = perf_counter_ns() - t0
    bench("bench_decode_schema_10fields", elapsed, iters)


# ============================================================================
# Realistic-scale RecordBatch encode/decode — exercises encode_array/
# decode_array/encode_record_batch's column-body copy path at a volume
# large enough for a byte-by-byte copy loop to actually show up (the
# benchmarks above are all small, fixed-size messages).
# ============================================================================


def _build_bench_batch(n_rows: Int) raises -> Tuple[ArrowSchema, RecordBatch]:
    var fields = List[ArrowField]()
    fields.append(ArrowField("id", ArrowType.int_(64, True), False))
    fields.append(ArrowField("count", ArrowType.int_(64, True), False))
    fields.append(ArrowField("label", ArrowType.utf8(), False))
    var schema = ArrowSchema(fields, Int16(0))

    var id_values = List[UInt8](capacity=n_rows * 8)
    for _ in range(n_rows * 8):
        id_values.append(UInt8(0))
    for i in range(n_rows):
        write_i64_le(id_values, i * 8, Int64(i))
    var id_col = ArrowArray(ArrowType.int_(64, True), n_rows, 0, List[UInt8](), List[UInt8](), id_values)

    var count_values = List[UInt8](capacity=n_rows * 8)
    for _ in range(n_rows * 8):
        count_values.append(UInt8(0))
    for i in range(n_rows):
        write_i64_le(count_values, i * 8, Int64(i * 2))
    var count_col = ArrowArray(ArrowType.int_(64, True), n_rows, 0, List[UInt8](), List[UInt8](), count_values)

    # Utf8 column: short fixed-width-ish strings, e.g. "row-0", "row-1", ...
    var label_offsets = List[UInt8](capacity=(n_rows + 1) * 4)
    for _ in range((n_rows + 1) * 4):
        label_offsets.append(UInt8(0))
    var label_values = List[UInt8]()
    var cur = 0
    write_i32_le(label_offsets, 0, Int32(0))
    for i in range(n_rows):
        var s = String("row-") + String(i)
        var sb = s.as_bytes()
        for j in range(len(sb)):
            label_values.append(sb[j])
        cur += len(sb)
        write_i32_le(label_offsets, (i + 1) * 4, Int32(cur))
    var label_col = ArrowArray(ArrowType.utf8(), n_rows, 0, List[UInt8](), label_offsets, label_values)

    var cols = List[ArrowArray]()
    cols.append(id_col^)
    cols.append(count_col^)
    cols.append(label_col^)
    var batch = RecordBatch(Int64(n_rows), cols)
    return Tuple[ArrowSchema, RecordBatch](schema^, batch^)


def bench_encode_record_batch_100k_rows() raises:
    var n_rows = 100_000
    var built = _build_bench_batch(n_rows)
    var schema = built[0].copy()
    var batch = built[1].copy()

    var iters = 10
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var msg = encode_record_batch(schema, batch.columns)
        _ = len(msg)
    var elapsed = perf_counter_ns() - t0
    bench("bench_encode_record_batch_100k_rows (3 cols)", elapsed, iters)


def bench_decode_record_batch_100k_rows() raises:
    var n_rows = 100_000
    var built = _build_bench_batch(n_rows)
    var schema = built[0].copy()
    var batch = built[1].copy()
    var msg = encode_record_batch(schema, batch.columns)

    var iters = 10
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var result = decode_record_batch(msg, 0, schema)
        _ = len(result[0])
    var elapsed = perf_counter_ns() - t0
    bench("bench_decode_record_batch_100k_rows (3 cols)", elapsed, iters)


def main() raises:
    print("=== arrow benchmarks ===")
    bench_ipc_pad8()
    bench_encode_ipc_message_1kb()
    bench_decode_ipc_message_1kb()
    bench_type_encode_decode_int32()
    bench_type_encode_decode_utf8()
    bench_encode_ipc_empty()
    bench_encode_schema_1field()
    bench_encode_schema_10fields()
    bench_decode_schema_10fields()
    bench_encode_record_batch_100k_rows()
    bench_decode_record_batch_100k_rows()
    print("done")
