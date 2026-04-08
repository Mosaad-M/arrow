from std.time import perf_counter_ns
from flatbuffers import FlatBufferBuilder, FlatBuffersReader
from arrow import (
    ipc_pad8,
    encode_ipc_message,
    decode_ipc_message,
    encode_arrow_type,
    decode_arrow_type,
    ArrowType,
)


fn bench(name: String, ns: UInt, iters: Int):
    var per_op = ns / UInt(iters)
    print("  " + name + ": " + String(per_op) + " ns/op")


fn bench_ipc_pad8() raises:
    var iters = 1_000_000
    var t0 = perf_counter_ns()
    var acc = 0
    for i in range(iters):
        acc += ipc_pad8(i * 13)
    var elapsed = perf_counter_ns() - t0
    _ = acc
    bench("bench_ipc_pad8", elapsed, iters)


fn bench_encode_ipc_message_1kb() raises:
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


fn bench_decode_ipc_message_1kb() raises:
    var meta = List[UInt8](capacity=32)
    for i in range(32):
        meta.append(UInt8(i))
    var body = List[UInt8](capacity=1024)
    for i in range(1024):
        body.append(UInt8(i % 256))
    var msg = encode_ipc_message(meta, body)

    var iters = 10_000
    var t0 = perf_counter_ns()
    for _ in range(iters):
        var result = decode_ipc_message(msg, 0)
        _ = len(result[0])
    var elapsed = perf_counter_ns() - t0
    bench("bench_decode_ipc_message_1kb", elapsed, iters)


fn bench_type_encode_decode_int32() raises:
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


fn bench_type_encode_decode_utf8() raises:
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


fn bench_encode_ipc_empty() raises:
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


fn main() raises:
    print("=== arrow benchmarks ===")
    bench_ipc_pad8()
    bench_encode_ipc_message_1kb()
    bench_decode_ipc_message_1kb()
    bench_type_encode_decode_int32()
    bench_type_encode_decode_utf8()
    bench_encode_ipc_empty()
    print("done")
