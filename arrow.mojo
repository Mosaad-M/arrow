from flatbuffers import (
    read_u8, read_u16_le, read_u32_le, read_i32_le, read_i64_le,
    read_u64_le, read_f32_le, read_f64_le,
    write_u8, write_u16_le, write_u32_le, write_i32_le, write_i64_le,
    write_u64_le, write_f32_le, write_f64_le,
    FlatBufferBuilder, FlatBuffersReader,
)


# ============================================================================
# Phase 1 — IPC message framing
# ============================================================================

fn _ipc_continuation() -> UInt32:
    return UInt32(0xFFFFFFFF)


fn ipc_pad8(size: Int) -> Int:
    """Return the smallest multiple of 8 >= size."""
    return (size + 7) & ~7


fn encode_ipc_message(metadata: List[UInt8], body: List[UInt8]) raises -> List[UInt8]:
    """
    Encode one IPC message:
      [0xFFFFFFFF: u32 LE]         continuation marker
      [metadata_len: i32 LE]       byte count of metadata (no padding included)
      [metadata bytes]
      [zero padding to 8-byte boundary]
      [body bytes]
      [zero padding to 8-byte boundary]
    """
    var meta_len = len(metadata)
    var header_size = 4 + 4 + meta_len          # continuation + length + metadata
    var padded_header = ipc_pad8(header_size)
    var meta_pad = padded_header - header_size

    var body_len = len(body)
    var padded_body = ipc_pad8(body_len)
    var body_pad = padded_body - body_len

    var total = padded_header + padded_body
    var out = List[UInt8](capacity=total)

    # Continuation marker
    out.append(UInt8(0xFF))
    out.append(UInt8(0xFF))
    out.append(UInt8(0xFF))
    out.append(UInt8(0xFF))

    # Metadata length (i32 LE)
    out.append(UInt8(meta_len & 0xFF))
    out.append(UInt8((meta_len >> 8) & 0xFF))
    out.append(UInt8((meta_len >> 16) & 0xFF))
    out.append(UInt8((meta_len >> 24) & 0xFF))

    # Metadata bytes
    for i in range(meta_len):
        out.append(metadata[i])

    # Padding after metadata
    for _ in range(meta_pad):
        out.append(UInt8(0))

    # Body bytes
    for i in range(body_len):
        out.append(body[i])

    # Padding after body
    for _ in range(body_pad):
        out.append(UInt8(0))

    return out^


fn decode_ipc_message(buf: List[UInt8], pos: Int) raises -> Tuple[List[UInt8], List[UInt8], Int]:
    """
    Parse one IPC message from buf at pos.
    Returns (metadata_bytes, body_bytes, next_pos).
    Raises on truncation or bad continuation marker.
    body_bytes is extracted using the bodyLength field from the caller's
    FlatBuffers Message table (Phase 4+). In Phase 1, returns everything
    after the padded header up to the next 8-byte boundary as body.
    """
    if pos + 8 > len(buf):
        raise Error("arrow: truncated IPC message at pos " + String(pos))

    var cont = read_u32_le(buf, pos)
    if cont != _ipc_continuation():
        raise Error("arrow: bad continuation marker: " + String(cont))

    var meta_len = Int(read_i32_le(buf, pos + 4))
    if meta_len < 0:
        raise Error("arrow: negative metadata length")

    var header_end = pos + 4 + 4 + meta_len
    if header_end > len(buf):
        raise Error("arrow: metadata truncated")

    var padded_header_end = ipc_pad8(4 + 4 + meta_len) + pos

    var metadata = List[UInt8](capacity=meta_len)
    for i in range(meta_len):
        metadata.append(buf[pos + 8 + i])

    # Return everything from padded_header_end to end of buf as body
    # (Phase 4 will use bodyLength from the FlatBuffers Message table instead)
    var body = List[UInt8]()
    var next_pos = padded_header_end
    return Tuple[List[UInt8], List[UInt8], Int](metadata, body, next_pos)


fn encode_eos() -> List[UInt8]:
    """Returns the 8-byte IPC end-of-stream marker."""
    var out = List[UInt8](capacity=8)
    out.append(UInt8(0xFF))
    out.append(UInt8(0xFF))
    out.append(UInt8(0xFF))
    out.append(UInt8(0xFF))
    out.append(UInt8(0x00))
    out.append(UInt8(0x00))
    out.append(UInt8(0x00))
    out.append(UInt8(0x00))
    return out^
