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


fn ipc_pad8(size: Int) raises -> Int:
    """Return the smallest multiple of 8 >= size. Raises on negative or huge input."""
    if size < 0:
        raise Error("arrow: ipc_pad8: negative size")
    # Guard: size + 7 must not overflow (max safe = 2^62 - 1 on 64-bit)
    if size > 0x3FFF_FFFF_FFFF_FFFF:
        raise Error("arrow: ipc_pad8: size too large")
    return (size + 7) & ~7


# 1 GB hard cap per IPC message — matches flatbuffers._grow() guard philosophy
fn _max_ipc_msg() -> Int:
    return 1 << 30


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
    # S2: enforce 1 GB cap to prevent overflow-induced OOM
    if len(metadata) > _max_ipc_msg() or len(body) > _max_ipc_msg():
        raise Error("arrow: encode_ipc_message: message too large (> 1 GB)")

    var meta_len = len(metadata)
    var header_size = 4 + 4 + meta_len          # continuation + length + metadata
    var padded_header = ipc_pad8(header_size)
    var meta_pad = padded_header - header_size

    var body_len = len(body)
    var padded_body = ipc_pad8(body_len)
    var body_pad = padded_body - body_len

    var total = padded_header + padded_body
    var out = List[UInt8](capacity=total)

    # P2: write continuation marker and metadata length using LE write helpers
    # pre-append 8 zero bytes then write at known positions
    for _ in range(8):
        out.append(UInt8(0))
    write_u32_le(out, 0, UInt32(0xFFFFFFFF))
    write_i32_le(out, 4, Int32(meta_len))

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
    FlatBuffers Message table (Phase 4+). Phase 1 returns empty body.
    """
    # S4: validate pos before any arithmetic
    if pos < 0:
        raise Error("arrow: decode_ipc_message: negative pos")
    if pos + 8 > len(buf):
        raise Error("arrow: truncated IPC message at pos " + String(pos))

    var cont = read_u32_le(buf, pos)
    if cont != _ipc_continuation():
        raise Error("arrow: bad continuation marker")

    var meta_len_i32 = read_i32_le(buf, pos + 4)
    if meta_len_i32 < 0:
        raise Error("arrow: negative metadata length")

    var meta_len = Int(meta_len_i32)

    # S3: overflow-safe bounds check — subtract instead of add to avoid wrapping
    if meta_len > len(buf) - pos - 8:
        raise Error("arrow: metadata truncated")

    var padded_header_end = ipc_pad8(8 + meta_len) + pos
    if padded_header_end > len(buf):
        raise Error("arrow: padded metadata exceeds buffer")

    var metadata = List[UInt8](capacity=meta_len)
    for i in range(meta_len):
        metadata.append(buf[pos + 8 + i])

    # Phase 4 will populate body using bodyLength from the FlatBuffers Message table
    var body = List[UInt8]()
    return Tuple[List[UInt8], List[UInt8], Int](metadata^, body^, padded_header_end)


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
    # S5: validate tag before encoding to prevent silent type confusion
    var disc = t.tag
    if disc < TYPE_NULL() or disc > TYPE_BOOL():
        raise Error("arrow: encode_arrow_type: unknown type tag: " + String(disc))

    # P1: locals eliminate repeated function calls in comparisons
    var T_INT   = UInt8(2)
    var T_FLOAT = UInt8(3)

    if disc == T_INT:
        b.start_table()
        b.add_field_i32(0, t.int_meta.bit_width)
        b.add_field_bool(1, t.int_meta.is_signed)
        var off = b.end_table()
        return Tuple[UInt8, UInt32](disc, off)
    elif disc == T_FLOAT:
        b.start_table()
        b.add_field_u16(0, t.float_meta.precision)
        var off = b.end_table()
        return Tuple[UInt8, UInt32](disc, off)
    else:
        # Null, Binary, Utf8, Bool — empty table (vtable dedup shares one vtable)
        b.start_table()
        var off = b.end_table()
        return Tuple[UInt8, UInt32](disc, off)


fn decode_arrow_type(r: FlatBuffersReader, discriminant: UInt8, type_tp: UInt32) raises -> ArrowType:
    """Reads the type table at type_tp and returns an ArrowType."""
    # P1: locals eliminate repeated function calls
    var T_NULL  = UInt8(1)
    var T_INT   = UInt8(2)
    var T_FLOAT = UInt8(3)
    var T_BIN   = UInt8(4)
    var T_UTF8  = UInt8(5)
    var T_BOOL  = UInt8(6)

    if discriminant == T_NULL:
        return ArrowType.null()
    elif discriminant == T_INT:
        var bw = r.read_i32(type_tp, 0)
        var signed = r.read_bool(type_tp, 1)
        return ArrowType.int_(bw, signed)
    elif discriminant == T_FLOAT:
        var prec = r.read_u16(type_tp, 0)
        return ArrowType.float_(prec)
    elif discriminant == T_BIN:
        return ArrowType.binary()
    elif discriminant == T_UTF8:
        return ArrowType.utf8()
    elif discriminant == T_BOOL:
        return ArrowType.bool_()
    else:
        raise Error("arrow: unknown type discriminant: " + String(discriminant))


# ============================================================================
# Phase 3 — Schema message encoding/decoding
# ============================================================================

struct ArrowField(Copyable, Movable):
    """One column descriptor: name, type, nullable."""
    var name: String
    var type: ArrowType
    var nullable: Bool

    fn __init__(out self, name: String, type: ArrowType, nullable: Bool):
        self.name = name
        self.type = type.copy()
        self.nullable = nullable

    fn __copyinit__(out self, copy: Self):
        self.name = copy.name
        self.type = copy.type.copy()
        self.nullable = copy.nullable

    fn __moveinit__(out self, deinit take: Self):
        self.name = take.name^
        self.type = take.type^
        self.nullable = take.nullable

    fn copy(self) -> Self:
        return Self(self.name, self.type.copy(), self.nullable)


struct ArrowSchema(Copyable, Movable):
    """Schema: ordered list of fields and endianness."""
    var fields: List[ArrowField]
    var endianness: Int16   # 0 = little-endian, 1 = big-endian

    fn __init__(out self, fields: List[ArrowField], endianness: Int16 = Int16(0)):
        self.fields = List[ArrowField]()
        for i in range(len(fields)):
            self.fields.append(fields[i].copy())
        self.endianness = endianness

    fn __copyinit__(out self, copy: Self):
        self.fields = List[ArrowField]()
        for i in range(len(copy.fields)):
            self.fields.append(copy.fields[i].copy())
        self.endianness = copy.endianness

    fn __moveinit__(out self, deinit take: Self):
        self.fields = take.fields^
        self.endianness = take.endianness


fn encode_schema_message(schema: ArrowSchema) raises -> List[UInt8]:
    """
    Encode an ArrowSchema as an IPC Schema message.
    Layout (FlatBuffers, bottom-up):
      For each field: type table → name string → Field table
      Vector of Field offsets → Schema table → Message table
      Wrapped in encode_ipc_message (no body for Schema).
    """
    # S6: cap field count to prevent runaway allocation
    if len(schema.fields) > 65536:
        raise Error("arrow: encode_schema_message: too many fields (> 65536)")

    # P4: size estimate — each field needs ~100 bytes in the FlatBuffer
    var est_capacity = len(schema.fields) * 100 + 512
    if est_capacity < 1024:
        est_capacity = 1024
    var b = FlatBufferBuilder(est_capacity)

    # Build each Field table bottom-up, collecting offsets
    var field_offs = List[UInt32]()
    for i in range(len(schema.fields)):
        # Explicit copy to ensure safe ownership before FlatBuffer mutations
        var f = schema.fields[i].copy()

        # 1. Build type sub-table (must precede start_table for Field)
        var type_result = encode_arrow_type(b, f.type)
        var type_disc = type_result[0]
        var type_off  = type_result[1]

        # 2. Create name string (must precede start_table for Field)
        var name_off = b.create_string(f.name)

        # 3. Build Field table
        b.start_table()
        b.add_field_offset(0, name_off)
        b.add_field_bool(1, f.nullable)
        b.add_field_u8(2, type_disc)     # union discriminant
        b.add_field_offset(3, type_off)  # union value
        # slot 4 (children) intentionally absent — Phase 3 has no nested types
        var foff = b.end_table()
        field_offs.append(foff)

    # 4. Vector of Field offsets
    var fields_vec_off = b.create_vector_offsets(field_offs)

    # 5. Schema table
    b.start_table()
    b.add_field_i16(0, schema.endianness)
    b.add_field_offset(1, fields_vec_off)
    var schema_off = b.end_table()

    # 6. Message table
    b.start_table()
    b.add_field_i16(0, Int16(4))        # version = MetadataVersion.V5
    b.add_field_u8(1, UInt8(1))         # header_type = Schema
    b.add_field_offset(2, schema_off)   # union value
    b.add_field_i64(3, Int64(0))        # bodyLength = 0
    var msg_off = b.end_table()

    # 7. Finalize FlatBuffer and wrap in IPC framing
    var flatbuf = b.finish(msg_off)
    return encode_ipc_message(flatbuf, List[UInt8]())


fn decode_schema_message(buf: List[UInt8], pos: Int) raises -> Tuple[ArrowSchema, Int]:
    """
    Decode an IPC Schema message from buf at pos.
    Returns (schema, next_pos).
    Raises if header_type != Schema or if the buffer is malformed.
    """
    # S7: validate pos
    if pos < 0:
        raise Error("arrow: decode_schema_message: negative pos")

    var ipc_result = decode_ipc_message(buf, pos)
    var metadata = ipc_result[0].copy()
    var next_pos  = ipc_result[2]

    var r = FlatBuffersReader(metadata)
    var msg_tp = r.root()

    # Validate this is a Schema message (header_type slot 1 must be 1)
    var header_type = r.read_u8(msg_tp, 1)
    if header_type != UInt8(1):
        raise Error("arrow: decode_schema_message: invalid header_type (expected Schema)")

    # Read Schema table via union_table (slot 2 is the union value offset)
    var schema_tp = r.union_table(msg_tp, 2)

    var endianness = r.read_i16(schema_tp, 0)

    # Read fields vector
    var fields_vec = r.read_vector(schema_tp, 1)
    var n_fields = r.vector_len(fields_vec)

    # S8: cap field count on decode to avoid runaway allocation from corrupt data
    if n_fields > UInt32(65536):
        raise Error("arrow: decode_schema_message: field count exceeds limit")

    var fields = List[ArrowField]()
    for i in range(Int(n_fields)):
        var field_tp = r.vec_offset(fields_vec, UInt32(i))
        var name = r.read_string(field_tp, 0)
        var nullable = r.read_bool(field_tp, 1)
        var disc = r.union_type(field_tp, 2)
        var type_tp = r.union_table(field_tp, 3)
        var arrow_type = decode_arrow_type(r, disc, type_tp)
        fields.append(ArrowField(name, arrow_type, nullable))

    return Tuple[ArrowSchema, Int](ArrowSchema(fields, endianness), next_pos)
