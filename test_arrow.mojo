from arrow import (
    ipc_pad8,
    encode_ipc_message,
    encode_eos,
)
from flatbuffers import read_i32_le, read_u32_le, write_u32_le, write_i32_le


# ============================================================================
# Test helpers
# ============================================================================


fn assert_true(cond: Bool, msg: String = "") raises:
    if not cond:
        raise Error(msg if len(msg) > 0 else "expected True")


fn assert_eq_int(actual: Int, expected: Int, msg: String = "") raises:
    if actual != expected:
        var m = "expected " + String(expected) + " got " + String(actual)
        raise Error(msg + ": " + m if len(msg) > 0 else m)


fn assert_eq_u8(actual: UInt8, expected: UInt8, msg: String = "") raises:
    if actual != expected:
        var m = "expected " + String(expected) + " got " + String(actual)
        raise Error(msg + ": " + m if len(msg) > 0 else m)


fn assert_eq_u32(actual: UInt32, expected: UInt32, msg: String = "") raises:
    if actual != expected:
        var m = "expected " + String(expected) + " got " + String(actual)
        raise Error(msg + ": " + m if len(msg) > 0 else m)


fn assert_eq_i32(actual: Int32, expected: Int32, msg: String = "") raises:
    if actual != expected:
        var m = "expected " + String(expected) + " got " + String(actual)
        raise Error(msg + ": " + m if len(msg) > 0 else m)


fn assert_f64_near(a: Float64, b: Float64, eps: Float64 = 1e-9) raises:
    var diff = a - b
    if diff < 0.0:
        diff = -diff
    if diff > eps:
        raise Error("float64 mismatch: " + String(a) + " vs " + String(b))


# ============================================================================
# Phase 1 — IPC message framing
# ============================================================================


fn test_ipc_pad8_already_aligned() raises:
    assert_eq_int(ipc_pad8(0), 0, "pad8(0)")
    assert_eq_int(ipc_pad8(8), 8, "pad8(8)")
    assert_eq_int(ipc_pad8(16), 16, "pad8(16)")
    assert_eq_int(ipc_pad8(256), 256, "pad8(256)")


fn test_ipc_pad8_needs_padding() raises:
    assert_eq_int(ipc_pad8(1), 8, "pad8(1)")
    assert_eq_int(ipc_pad8(7), 8, "pad8(7)")
    assert_eq_int(ipc_pad8(9), 16, "pad8(9)")
    assert_eq_int(ipc_pad8(15), 16, "pad8(15)")
    assert_eq_int(ipc_pad8(17), 24, "pad8(17)")


fn test_encode_ipc_message_continuation() raises:
    var meta = List[UInt8]()
    meta.append(UInt8(1))
    meta.append(UInt8(2))
    var body = List[UInt8]()
    var msg = encode_ipc_message(meta, body)
    assert_eq_u8(msg[0], UInt8(0xFF), "continuation byte 0")
    assert_eq_u8(msg[1], UInt8(0xFF), "continuation byte 1")
    assert_eq_u8(msg[2], UInt8(0xFF), "continuation byte 2")
    assert_eq_u8(msg[3], UInt8(0xFF), "continuation byte 3")


fn test_encode_ipc_message_metadata_length() raises:
    var meta = List[UInt8]()
    for _ in range(13):
        meta.append(UInt8(0xAB))
    var body = List[UInt8]()
    var msg = encode_ipc_message(meta, body)
    var mlen = read_i32_le(msg, 4)
    assert_eq_i32(mlen, Int32(13), "metadata_length field")


fn test_encode_ipc_message_metadata_content() raises:
    var meta = List[UInt8]()
    meta.append(UInt8(0x11))
    meta.append(UInt8(0x22))
    meta.append(UInt8(0x33))
    var body = List[UInt8]()
    var msg = encode_ipc_message(meta, body)
    assert_eq_u8(msg[8], UInt8(0x11), "meta[0]")
    assert_eq_u8(msg[9], UInt8(0x22), "meta[1]")
    assert_eq_u8(msg[10], UInt8(0x33), "meta[2]")


fn test_encode_ipc_message_body_aligned() raises:
    # 4 (continuation) + 4 (length) + 5 (metadata) = 13 bytes → pad to 16
    # body follows at byte 16
    var meta = List[UInt8]()
    for _ in range(5):
        meta.append(UInt8(0xCC))
    var body = List[UInt8]()
    for i in range(3):
        body.append(UInt8(i + 1))
    var msg = encode_ipc_message(meta, body)
    var header_plus_meta = 4 + 4 + len(meta)
    var padded_header = ipc_pad8(header_plus_meta)
    assert_eq_int(padded_header % 8, 0, "header+meta padded to 8")
    # body starts at padded_header
    assert_eq_u8(msg[padded_header], UInt8(1), "body[0]")
    assert_eq_u8(msg[padded_header + 1], UInt8(2), "body[1]")
    assert_eq_u8(msg[padded_header + 2], UInt8(3), "body[2]")
    # total length is 8-byte aligned
    assert_eq_int(len(msg) % 8, 0, "total length % 8")


fn test_encode_ipc_message_empty_body() raises:
    var meta = List[UInt8]()
    meta.append(UInt8(0x55))
    var body = List[UInt8]()
    var msg = encode_ipc_message(meta, body)
    # Must still be 8-byte aligned and have the continuation marker
    assert_eq_u8(msg[0], UInt8(0xFF), "continuation")
    assert_eq_int(len(msg) % 8, 0, "empty body: total % 8")


fn test_encode_eos_exact_bytes() raises:
    var eos = encode_eos()
    assert_eq_int(len(eos), 8, "eos length")
    assert_eq_u8(eos[0], UInt8(0xFF), "eos[0]")
    assert_eq_u8(eos[1], UInt8(0xFF), "eos[1]")
    assert_eq_u8(eos[2], UInt8(0xFF), "eos[2]")
    assert_eq_u8(eos[3], UInt8(0xFF), "eos[3]")
    assert_eq_u8(eos[4], UInt8(0x00), "eos[4]")
    assert_eq_u8(eos[5], UInt8(0x00), "eos[5]")
    assert_eq_u8(eos[6], UInt8(0x00), "eos[6]")
    assert_eq_u8(eos[7], UInt8(0x00), "eos[7]")


# ============================================================================
# Test runner
# ============================================================================


fn run_test(name: String, mut passed: Int, mut failed: Int, test_fn: fn () raises -> None):
    try:
        test_fn()
        print("  PASS: " + name)
        passed += 1
    except e:
        print("  FAIL: " + name + " — " + String(e))
        failed += 1


fn main() raises:
    print("=== arrow tests ===")

    var passed = 0
    var failed = 0

    # Phase 1 — IPC message framing
    run_test("test_ipc_pad8_already_aligned", passed, failed, test_ipc_pad8_already_aligned)
    run_test("test_ipc_pad8_needs_padding", passed, failed, test_ipc_pad8_needs_padding)
    run_test("test_encode_ipc_message_continuation", passed, failed, test_encode_ipc_message_continuation)
    run_test("test_encode_ipc_message_metadata_length", passed, failed, test_encode_ipc_message_metadata_length)
    run_test("test_encode_ipc_message_metadata_content", passed, failed, test_encode_ipc_message_metadata_content)
    run_test("test_encode_ipc_message_body_aligned", passed, failed, test_encode_ipc_message_body_aligned)
    run_test("test_encode_ipc_message_empty_body", passed, failed, test_encode_ipc_message_empty_body)
    run_test("test_encode_eos_exact_bytes", passed, failed, test_encode_eos_exact_bytes)

    print("\n" + String(passed) + "/" + String(passed + failed) + " passed")
    if failed > 0:
        raise Error(String(failed) + " test(s) failed")
