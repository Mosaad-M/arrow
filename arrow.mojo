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


# ============================================================================
# Phase 2 — Arrow type encoding/decoding
# ============================================================================

fn TYPE_NULL() -> UInt8:
    return UInt8(1)

fn TYPE_INT() -> UInt8:
    return UInt8(2)

fn TYPE_FLOAT() -> UInt8:
    return UInt8(3)

fn TYPE_BINARY() -> UInt8:
    return UInt8(4)

fn TYPE_UTF8() -> UInt8:
    return UInt8(5)

fn TYPE_BOOL() -> UInt8:
    return UInt8(6)


struct ArrowInt(Copyable, Movable):
    """Metadata for Int Arrow type."""
    var bit_width: Int32
    var is_signed: Bool

    fn __init__(out self, bit_width: Int32, is_signed: Bool):
        self.bit_width = bit_width
        self.is_signed = is_signed

    fn __copyinit__(out self, copy: Self):
        self.bit_width = copy.bit_width
        self.is_signed = copy.is_signed

    fn __moveinit__(out self, deinit take: Self):
        self.bit_width = take.bit_width
        self.is_signed = take.is_signed

    fn copy(self) -> Self:
        return Self(self.bit_width, self.is_signed)


struct ArrowFloat(Copyable, Movable):
    """Metadata for FloatingPoint Arrow type."""
    var precision: UInt16   # 1=Single, 2=Double

    fn __init__(out self, precision: UInt16):
        self.precision = precision

    fn __copyinit__(out self, copy: Self):
        self.precision = copy.precision

    fn __moveinit__(out self, deinit take: Self):
        self.precision = take.precision

    fn copy(self) -> Self:
        return Self(self.precision)


struct ArrowType(Copyable, Movable):
    """A tagged Arrow column type with optional Int or Float metadata."""
    var tag: UInt8
    var int_meta: ArrowInt
    var float_meta: ArrowFloat

    fn __init__(out self, tag: UInt8, int_bit_width: Int32, int_is_signed: Bool, float_precision: UInt16):
        self.tag = tag
        self.int_meta = ArrowInt(int_bit_width, int_is_signed)
        self.float_meta = ArrowFloat(float_precision)

    fn __copyinit__(out self, copy: Self):
        self.tag = copy.tag
        self.int_meta = copy.int_meta.copy()
        self.float_meta = copy.float_meta.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.tag = take.tag
        self.int_meta = take.int_meta.copy()
        self.float_meta = take.float_meta.copy()

    @staticmethod
    fn null() -> ArrowType:
        return ArrowType(TYPE_NULL(), 0, False, 0)

    @staticmethod
    fn int_(bit_width: Int32, is_signed: Bool) -> ArrowType:
        return ArrowType(TYPE_INT(), bit_width, is_signed, 0)

    @staticmethod
    fn float_(precision: UInt16) -> ArrowType:
        return ArrowType(TYPE_FLOAT(), 0, False, precision)

    @staticmethod
    fn binary() -> ArrowType:
        return ArrowType(TYPE_BINARY(), 0, False, 0)

    @staticmethod
    fn utf8() -> ArrowType:
        return ArrowType(TYPE_UTF8(), 0, False, 0)

    @staticmethod
    fn bool_() -> ArrowType:
        return ArrowType(TYPE_BOOL(), 0, False, 0)


fn encode_arrow_type(mut b: FlatBufferBuilder, t: ArrowType) raises -> Tuple[UInt8, UInt32]:
    """
    Builds the type table in b and returns (discriminant, table_offset).
    Build order: type table must be built before the Field table that references it.
    """
    var disc = t.tag
    if disc == TYPE_INT():
        b.start_table()
        b.add_field_i32(0, t.int_meta.bit_width)
        b.add_field_bool(1, t.int_meta.is_signed)
        var off = b.end_table()
        return Tuple[UInt8, UInt32](disc, off)
    elif disc == TYPE_FLOAT():
        b.start_table()
        b.add_field_u16(0, t.float_meta.precision)
        var off = b.end_table()
        return Tuple[UInt8, UInt32](disc, off)
    else:
        # Null, Binary, Utf8, Bool — empty table
        b.start_table()
        var off = b.end_table()
        return Tuple[UInt8, UInt32](disc, off)


fn decode_arrow_type(r: FlatBuffersReader, discriminant: UInt8, type_tp: UInt32) raises -> ArrowType:
    """Reads the type table at type_tp and returns an ArrowType."""
    if discriminant == TYPE_NULL():
        return ArrowType.null()
    elif discriminant == TYPE_INT():
        var bw = r.read_i32(type_tp, 0)
        var signed = r.read_bool(type_tp, 1)
        return ArrowType.int_(bw, signed)
    elif discriminant == TYPE_FLOAT():
        var prec = r.read_u16(type_tp, 0)
        return ArrowType.float_(prec)
    elif discriminant == TYPE_BINARY():
        return ArrowType.binary()
    elif discriminant == TYPE_UTF8():
        return ArrowType.utf8()
    elif discriminant == TYPE_BOOL():
        return ArrowType.bool_()
    else:
        raise Error("arrow: unknown type discriminant: " + String(discriminant))


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
