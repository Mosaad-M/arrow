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

    # Read bodyLength from the FlatBuffers Message table (slot 3, i64).
    # Schema messages have bodyLength=0; RecordBatch messages have the actual body size.
    var body = List[UInt8]()
    var r_meta = FlatBuffersReader(metadata)
    var msg_tp = r_meta.root()
    var body_len_i64 = r_meta.read_i64(msg_tp, 3)
    if body_len_i64 < Int64(0):
        raise Error("arrow: negative bodyLength in IPC message")
    var body_len = Int(body_len_i64)
    if body_len > 0:
        if body_len > len(buf) - padded_header_end:
            raise Error("arrow: body truncated")
        body = List[UInt8](capacity=body_len)
        for i in range(body_len):
            body.append(buf[padded_header_end + i])
    var padded_body_end = padded_header_end + ipc_pad8(body_len)
    return Tuple[List[UInt8], List[UInt8], Int](metadata^, body^, padded_body_end)


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

    fn copy(self) -> Self:
        return Self(self.tag, self.int_meta.bit_width, self.int_meta.is_signed, self.float_meta.precision)

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


# ============================================================================
# Phase 4 — RecordBatch Message
# ============================================================================


struct FieldNode(Copyable, Movable):
    """Describes one Arrow column's row count and null count in a RecordBatch."""

    var length: Int64
    var null_count: Int64

    fn __init__(out self, length: Int64, null_count: Int64):
        self.length = length
        self.null_count = null_count

    fn __copyinit__(out self, copy: Self):
        self.length = copy.length
        self.null_count = copy.null_count

    fn __moveinit__(out self, deinit take: Self):
        self.length = take.length
        self.null_count = take.null_count

    fn copy(self) -> Self:
        return Self(self.length, self.null_count)


struct BufferDesc(Copyable, Movable):
    """Describes one buffer's byte offset and length within an IPC message body."""

    var offset: Int64
    var length: Int64

    fn __init__(out self, offset: Int64, length: Int64):
        self.offset = offset
        self.length = length

    fn __copyinit__(out self, copy: Self):
        self.offset = copy.offset
        self.length = copy.length

    fn __moveinit__(out self, deinit take: Self):
        self.offset = take.offset
        self.length = take.length

    fn copy(self) -> Self:
        return Self(self.offset, self.length)


fn _field_node_bytes(node: FieldNode) -> List[UInt8]:
    """Serialize a FieldNode as 16 LE bytes: [length:i64][null_count:i64]."""
    var result = List[UInt8](capacity=16)
    for _ in range(16):
        result.append(UInt8(0))
    write_i64_le(result, 0, node.length)
    write_i64_le(result, 8, node.null_count)
    return result^


fn _buffer_desc_bytes(bd: BufferDesc) -> List[UInt8]:
    """Serialize a BufferDesc as 16 LE bytes: [offset:i64][length:i64]."""
    var result = List[UInt8](capacity=16)
    for _ in range(16):
        result.append(UInt8(0))
    write_i64_le(result, 0, bd.offset)
    write_i64_le(result, 8, bd.length)
    return result^


fn _field_node_from_bytes(data: List[UInt8]) raises -> FieldNode:
    """Deserialize a FieldNode from 16 LE bytes."""
    return FieldNode(read_i64_le(data, 0), read_i64_le(data, 8))


fn _buffer_desc_from_bytes(data: List[UInt8]) raises -> BufferDesc:
    """Deserialize a BufferDesc from 16 LE bytes."""
    return BufferDesc(read_i64_le(data, 0), read_i64_le(data, 8))


fn encode_record_batch_message(
    length: Int64,
    nodes: List[FieldNode],
    buffers: List[BufferDesc],
    body: List[UInt8],
) raises -> List[UInt8]:
    """
    Encode an Arrow IPC RecordBatch message.

    length  — total row count.
    nodes   — one FieldNode per column (length + null_count).
    buffers — one BufferDesc per logical buffer (offset + byte length in body).
    body    — raw column data bytes.

    Returns full IPC message bytes (header envelope + body).
    """
    if len(nodes) > 65536:
        raise Error("arrow: encode_record_batch_message: too many nodes (> 65536)")
    if len(buffers) > 65536:
        raise Error("arrow: encode_record_batch_message: too many buffers (> 65536)")

    var est_capacity = (len(nodes) + len(buffers)) * 32 + 256
    if est_capacity < 512:
        est_capacity = 512
    var b = FlatBufferBuilder(est_capacity)

    # Build FieldNode struct vector (16 bytes per element)
    var nodes_bytes = List[UInt8](capacity=len(nodes) * 16)
    for i in range(len(nodes)):
        var nb = _field_node_bytes(nodes[i].copy())
        for j in range(16):
            nodes_bytes.append(nb[j])
    var nodes_vec_off = b.create_vector_structs(nodes_bytes, len(nodes), 16, 8)

    # Build Buffer struct vector (16 bytes per element)
    var buffers_bytes = List[UInt8](capacity=len(buffers) * 16)
    for i in range(len(buffers)):
        var bb = _buffer_desc_bytes(buffers[i].copy())
        for j in range(16):
            buffers_bytes.append(bb[j])
    var buffers_vec_off = b.create_vector_structs(buffers_bytes, len(buffers), 16, 8)

    # RecordBatch table: slot 0=length(i64), slot 1=nodes(vec), slot 2=buffers(vec)
    b.start_table()
    b.add_field_i64(0, length)
    b.add_field_offset(1, nodes_vec_off)
    b.add_field_offset(2, buffers_vec_off)
    var rb_off = b.end_table()

    # Message table: version=4(V5), header_type=3(RecordBatch), header, bodyLength
    b.start_table()
    b.add_field_i16(0, Int16(4))
    b.add_field_u8(1, UInt8(3))
    b.add_field_offset(2, rb_off)
    b.add_field_i64(3, Int64(len(body)))
    var msg_off = b.end_table()

    var flatbuf = b.finish(msg_off)
    return encode_ipc_message(flatbuf, body)


fn decode_record_batch_message(
    buf: List[UInt8],
    pos: Int,
) raises -> Tuple[Int64, List[FieldNode], List[BufferDesc], List[UInt8], Int]:
    """
    Decode an IPC RecordBatch message from buf at pos.
    Returns (length, nodes, buffers, body, next_pos).
    Raises if header_type != 3 (RecordBatch) or if the buffer is malformed.
    """
    if pos < 0:
        raise Error("arrow: decode_record_batch_message: negative pos")

    var ipc_result = decode_ipc_message(buf, pos)
    var metadata = ipc_result[0].copy()
    var body = ipc_result[1].copy()
    var next_pos = ipc_result[2]

    var r = FlatBuffersReader(metadata)
    var msg_tp = r.root()

    var header_type = r.read_u8(msg_tp, 1)
    if header_type != UInt8(3):
        raise Error(
            "arrow: decode_record_batch_message: invalid header_type (expected RecordBatch=3)"
        )

    var rb_tp = r.union_table(msg_tp, 2)
    var length = r.read_i64(rb_tp, 0)

    # Decode FieldNode struct vector
    var nodes_vec = r.read_vector(rb_tp, 1)
    var n_nodes = r.vector_len(nodes_vec)
    if n_nodes > UInt32(65536):
        raise Error("arrow: decode_record_batch_message: node count exceeds limit")
    var nodes = List[FieldNode]()
    for i in range(Int(n_nodes)):
        var nb = r.vec_struct_bytes(nodes_vec, UInt32(i), 16)
        nodes.append(_field_node_from_bytes(nb))

    # Decode Buffer struct vector
    var buffers_vec = r.read_vector(rb_tp, 2)
    var n_buffers = r.vector_len(buffers_vec)
    if n_buffers > UInt32(65536):
        raise Error("arrow: decode_record_batch_message: buffer count exceeds limit")
    var buffers = List[BufferDesc]()
    for i in range(Int(n_buffers)):
        var bb = r.vec_struct_bytes(buffers_vec, UInt32(i), 16)
        buffers.append(_buffer_desc_from_bytes(bb))

    return Tuple[Int64, List[FieldNode], List[BufferDesc], List[UInt8], Int](
        length, nodes^, buffers^, body^, next_pos
    )


# ============================================================================
# Phase 5 — ArrowArray: Typed Column Encoding
# ============================================================================


struct ArrowArray(Copyable, Movable):
    """A typed Arrow column with validity bitmap, optional offsets, and values."""

    var type: ArrowType
    var length: Int
    var null_count: Int
    var validity: List[UInt8]   # packed null bitmap; empty when null_count == 0
    var offsets: List[UInt8]    # variable-length types: (length+1) * 4 LE bytes
    var values: List[UInt8]     # raw value bytes

    fn __init__(
        out self,
        type: ArrowType,
        length: Int,
        null_count: Int,
        validity: List[UInt8],
        offsets: List[UInt8],
        values: List[UInt8],
    ):
        self.type = type.copy()
        self.length = length
        self.null_count = null_count
        self.validity = List[UInt8]()
        for i in range(len(validity)):
            self.validity.append(validity[i])
        self.offsets = List[UInt8]()
        for i in range(len(offsets)):
            self.offsets.append(offsets[i])
        self.values = List[UInt8]()
        for i in range(len(values)):
            self.values.append(values[i])

    fn __copyinit__(out self, copy: Self):
        self.type = copy.type.copy()
        self.length = copy.length
        self.null_count = copy.null_count
        self.validity = List[UInt8]()
        for i in range(len(copy.validity)):
            self.validity.append(copy.validity[i])
        self.offsets = List[UInt8]()
        for i in range(len(copy.offsets)):
            self.offsets.append(copy.offsets[i])
        self.values = List[UInt8]()
        for i in range(len(copy.values)):
            self.values.append(copy.values[i])

    fn __moveinit__(out self, deinit take: Self):
        self.type = take.type^
        self.length = take.length
        self.null_count = take.null_count
        self.validity = take.validity^
        self.offsets = take.offsets^
        self.values = take.values^

    fn copy(self) -> Self:
        return Self(
            self.type, self.length, self.null_count,
            self.validity, self.offsets, self.values,
        )


fn encode_array(
    arr: ArrowArray,
    body_offset: Int,
) raises -> Tuple[FieldNode, List[BufferDesc], List[UInt8]]:
    """
    Encode one Arrow column into a FieldNode, buffer descriptors, and body bytes.

    body_offset: the byte position where this array's data will sit in the
                 shared RecordBatch body. BufferDesc.offset values are absolute
                 (relative to the start of the full RecordBatch body).

    Returns (node, descs, body_bytes).
    body_bytes are padded to 8-byte boundaries internally.
    """
    var node = FieldNode(Int64(arr.length), Int64(arr.null_count))
    var descs = List[BufferDesc]()
    var body = List[UInt8]()

    # Null type: no buffers at all
    if arr.type.tag == TYPE_NULL():
        return Tuple[FieldNode, List[BufferDesc], List[UInt8]](node^, descs^, body^)

    var cur = body_offset

    # ── Validity bitmap ──────────────────────────────────────────────────────
    if arr.null_count > 0:
        var vlen = len(arr.validity)
        descs.append(BufferDesc(Int64(cur), Int64(vlen)))
        for i in range(vlen):
            body.append(arr.validity[i])
        var vpad = ipc_pad8(vlen) - vlen
        for _ in range(vpad):
            body.append(UInt8(0))
        cur += ipc_pad8(vlen)
    else:
        # Absent validity: descriptor with length=0, no bytes in body
        descs.append(BufferDesc(Int64(cur), Int64(0)))

    # ── Offsets (Utf8 / Binary only) ─────────────────────────────────────────
    if arr.type.tag == TYPE_UTF8() or arr.type.tag == TYPE_BINARY():
        var olen = len(arr.offsets)
        descs.append(BufferDesc(Int64(cur), Int64(olen)))
        for i in range(olen):
            body.append(arr.offsets[i])
        var opad = ipc_pad8(olen) - olen
        for _ in range(opad):
            body.append(UInt8(0))
        cur += ipc_pad8(olen)

    # ── Values ───────────────────────────────────────────────────────────────
    var dlen = len(arr.values)
    descs.append(BufferDesc(Int64(cur), Int64(dlen)))
    for i in range(dlen):
        body.append(arr.values[i])
    var dpad = ipc_pad8(dlen) - dlen
    for _ in range(dpad):
        body.append(UInt8(0))
    _ = cur + ipc_pad8(dlen)   # suppress unused-var warning

    return Tuple[FieldNode, List[BufferDesc], List[UInt8]](node^, descs^, body^)


fn decode_array(
    type: ArrowType,
    node: FieldNode,
    descs: List[BufferDesc],
    body: List[UInt8],
) raises -> ArrowArray:
    """
    Reconstruct an ArrowArray from its FieldNode, buffer descriptors, and the
    shared RecordBatch body.  Offsets in descs are absolute body positions.
    """
    # Null type: no buffers
    if type.tag == TYPE_NULL():
        return ArrowArray(
            type, Int(node.length), Int(node.null_count),
            List[UInt8](), List[UInt8](), List[UInt8](),
        )

    var validity = List[UInt8]()
    var offsets  = List[UInt8]()
    var values   = List[UInt8]()

    # ── Validity buffer (descs[0]) ────────────────────────────────────────────
    if len(descs) < 1:
        raise Error("arrow: decode_array: missing validity buffer descriptor")
    # S-P5-1: guard against negative offset/length from corrupt IPC data
    if descs[0].offset < Int64(0):
        raise Error("arrow: decode_array: negative validity buffer offset")
    if descs[0].length < Int64(0):
        raise Error("arrow: decode_array: negative validity buffer length")
    if descs[0].length > Int64(0):
        var start = Int(descs[0].offset)
        var end   = start + Int(descs[0].length)
        if end > len(body):
            raise Error("arrow: decode_array: validity buffer out of bounds")
        for i in range(start, end):
            validity.append(body[i])

    # ── Offsets + values (Utf8 / Binary) or just values (all other types) ────
    if type.tag == TYPE_UTF8() or type.tag == TYPE_BINARY():
        if len(descs) < 3:
            raise Error("arrow: decode_array: expected 3 buffer descriptors for variable-length type")
        if descs[1].offset < Int64(0) or descs[1].length < Int64(0):
            raise Error("arrow: decode_array: negative offsets buffer descriptor value")
        if descs[2].offset < Int64(0) or descs[2].length < Int64(0):
            raise Error("arrow: decode_array: negative values buffer descriptor value")
        var ostart = Int(descs[1].offset)
        var oend   = ostart + Int(descs[1].length)
        if oend > len(body):
            raise Error("arrow: decode_array: offsets buffer out of bounds")
        for i in range(ostart, oend):
            offsets.append(body[i])
        var vstart = Int(descs[2].offset)
        var vend   = vstart + Int(descs[2].length)
        if vend > len(body):
            raise Error("arrow: decode_array: values buffer out of bounds")
        for i in range(vstart, vend):
            values.append(body[i])
    else:
        if len(descs) < 2:
            raise Error("arrow: decode_array: expected 2 buffer descriptors for fixed-width type")
        if descs[1].offset < Int64(0) or descs[1].length < Int64(0):
            raise Error("arrow: decode_array: negative values buffer descriptor value")
        var vstart = Int(descs[1].offset)
        var vend   = vstart + Int(descs[1].length)
        if vend > len(body):
            raise Error("arrow: decode_array: values buffer out of bounds")
        for i in range(vstart, vend):
            values.append(body[i])

    return ArrowArray(
        type, Int(node.length), Int(node.null_count),
        validity, offsets, values,
    )


fn encode_record_batch(
    schema: ArrowSchema,
    arrays: List[ArrowArray],
) raises -> List[UInt8]:
    """
    Encode a full RecordBatch IPC message from a schema and its column arrays.
    Builds nodes + buffers + body from the arrays, then delegates to
    encode_record_batch_message.
    """
    if len(arrays) != len(schema.fields):
        raise Error("arrow: encode_record_batch: column count does not match schema field count")
    if len(arrays) > 65536:
        raise Error("arrow: encode_record_batch: too many columns (> 65536)")

    var row_count = Int64(0)
    if len(arrays) > 0:
        row_count = Int64(arrays[0].length)

    var nodes       = List[FieldNode]()
    var all_buffers = List[BufferDesc]()
    var full_body   = List[UInt8]()
    var cur_offset  = 0

    for i in range(len(arrays)):
        var result = encode_array(arrays[i], cur_offset)
        var col_node  = result[0].copy()
        var col_descs = result[1].copy()
        var col_body  = result[2].copy()

        nodes.append(col_node^)
        for j in range(len(col_descs)):
            all_buffers.append(col_descs[j].copy())
        for j in range(len(col_body)):
            full_body.append(col_body[j])
        # S-P5-2: guard against cur_offset overflow in large multi-column batches
        if len(col_body) > _max_ipc_msg() - cur_offset:
            raise Error("arrow: encode_record_batch: combined column body exceeds 1 GB")
        cur_offset += len(col_body)

    return encode_record_batch_message(row_count, nodes, all_buffers, full_body)


fn decode_record_batch(
    buf: List[UInt8],
    pos: Int,
    schema: ArrowSchema,
) raises -> Tuple[List[ArrowArray], Int]:
    """
    Decode a RecordBatch IPC message and reconstruct typed column arrays.
    schema supplies the Arrow type for each column.
    Returns (arrays, next_pos).
    """
    var ipc_result = decode_record_batch_message(buf, pos)
    var nodes    = ipc_result[1].copy()
    var buffers  = ipc_result[2].copy()
    var body     = ipc_result[3].copy()
    var next_pos = ipc_result[4]
    _ = len(body)

    if len(nodes) != len(schema.fields):
        raise Error("arrow: decode_record_batch: node count does not match schema field count")

    var arrays  = List[ArrowArray]()
    var buf_idx = 0

    for i in range(len(schema.fields)):
        var field_type = schema.fields[i].type.copy()
        var node       = nodes[i].copy()

        # How many buffer descriptors does this type consume?
        var n_bufs: Int
        if field_type.tag == TYPE_NULL():
            n_bufs = 0
        elif field_type.tag == TYPE_UTF8() or field_type.tag == TYPE_BINARY():
            n_bufs = 3
        else:
            n_bufs = 2

        if buf_idx + n_bufs > len(buffers):
            raise Error("arrow: decode_record_batch: buffer descriptor underrun for column " + String(i))

        var field_descs = List[BufferDesc]()
        for j in range(n_bufs):
            field_descs.append(buffers[buf_idx + j].copy())
        buf_idx += n_bufs

        arrays.append(decode_array(field_type, node, field_descs, body))

    return Tuple[List[ArrowArray], Int](arrays^, next_pos)


# ============================================================================
# Phase 6 — IPC File Format (Feather v2 / Arrow IPC File)
# ============================================================================


struct RecordBatch(Copyable, Movable):
    """A batch of typed columns sharing the same row count."""

    var length: Int64
    var columns: List[ArrowArray]

    fn __init__(out self, length: Int64, columns: List[ArrowArray]):
        self.length = length
        self.columns = List[ArrowArray]()
        for i in range(len(columns)):
            self.columns.append(columns[i].copy())

    fn __copyinit__(out self, copy: Self):
        self.length = copy.length
        self.columns = List[ArrowArray]()
        for i in range(len(copy.columns)):
            self.columns.append(copy.columns[i].copy())

    fn __moveinit__(out self, deinit take: Self):
        self.length = take.length
        self.columns = take.columns^

    fn copy(self) -> Self:
        return Self(self.length, self.columns)


fn _arrow_magic() -> List[UInt8]:
    """Return the 8-byte Arrow IPC file magic: b'ARROW1\\0\\0'."""
    var m = List[UInt8](capacity=8)
    m.append(UInt8(0x41))
    m.append(UInt8(0x52))
    m.append(UInt8(0x52))
    m.append(UInt8(0x4F))
    m.append(UInt8(0x57))
    m.append(UInt8(0x31))
    m.append(UInt8(0x00))
    m.append(UInt8(0x00))
    return m^


fn _encode_schema_table(mut b: FlatBufferBuilder, schema: ArrowSchema) raises -> UInt32:
    """
    Build the Schema FlatBuffer table into b (no Message envelope).
    Returns the UOffset of the Schema table.
    """
    if len(schema.fields) > 65536:
        raise Error("arrow: _encode_schema_table: too many fields (> 65536)")

    var field_offs = List[UInt32]()
    for i in range(len(schema.fields)):
        var f = schema.fields[i].copy()
        var type_result = encode_arrow_type(b, f.type)
        var type_disc = type_result[0]
        var type_off  = type_result[1]
        var name_off  = b.create_string(f.name)
        b.start_table()
        b.add_field_offset(0, name_off)
        b.add_field_bool(1, f.nullable)
        b.add_field_u8(2, type_disc)
        b.add_field_offset(3, type_off)
        var foff = b.end_table()
        field_offs.append(foff)

    var fields_vec_off = b.create_vector_offsets(field_offs)
    b.start_table()
    b.add_field_i16(0, schema.endianness)
    b.add_field_offset(1, fields_vec_off)
    return b.end_table()


fn _block_bytes(file_offset: Int64, meta_len: Int32, body_len: Int64) -> List[UInt8]:
    """Serialize one Block struct as 24 LE bytes: [offset:i64][metaLen:i32][pad:i32][bodyLen:i64]."""
    var result = List[UInt8](capacity=24)
    for _ in range(24):
        result.append(UInt8(0))
    write_i64_le(result, 0, file_offset)
    write_i32_le(result, 8, meta_len)
    # bytes 12-15: padding (zero)
    write_i64_le(result, 16, body_len)
    return result^


fn _block_offset_from_bytes(data: List[UInt8]) raises -> Int64:
    return read_i64_le(data, 0)


fn _block_meta_len_from_bytes(data: List[UInt8]) raises -> Int32:
    return read_i32_le(data, 8)


fn _block_body_len_from_bytes(data: List[UInt8]) raises -> Int64:
    return read_i64_le(data, 16)


fn encode_arrow_file(
    schema: ArrowSchema,
    batches: List[RecordBatch],
) raises -> List[UInt8]:
    """
    Encode schema + record batches as an Arrow IPC file (Feather v2).

    Layout:
      [magic: 8]
      [Schema IPC message]
      [RecordBatch IPC message 0]
      ...
      [Footer FlatBuffer]
      [footer_size: i32 LE, 4 bytes]
      [magic: 8]
    """
    var out = List[UInt8]()

    # ── Header magic ─────────────────────────────────────────────────────────
    var magic = _arrow_magic()
    for i in range(8):
        out.append(magic[i])

    # ── Schema IPC message ────────────────────────────────────────────────────
    var schema_msg = encode_schema_message(schema)
    for i in range(len(schema_msg)):
        out.append(schema_msg[i])

    # ── RecordBatch IPC messages ──────────────────────────────────────────────
    # Track Block info for each batch so we can build the Footer.
    var rb_blocks_bytes = List[UInt8]()   # raw 24-byte blocks
    var n_blocks = 0

    for b in range(len(batches)):
        var batch = batches[b].copy()
        var file_offset = Int64(len(out))

        # Build arrays from batch.columns using schema for type info
        var arrays = List[ArrowArray]()
        for c in range(len(batch.columns)):
            arrays.append(batch.columns[c].copy())

        var rb_msg = encode_record_batch(schema, arrays)
        var rb_msg_len = len(rb_msg)

        # metaDataLength = 8 (cont marker + len field) + padded metadata
        # bodyLength = batch body size
        # We encode the full message; body length is stored in the FlatBuffer
        # but for the Block we need it separately.
        # Parse the body length from the encoded message's FlatBuffer header.
        var ipc_result = decode_ipc_message(rb_msg, 0)
        var rb_meta = ipc_result[0].copy()
        var rb_next = ipc_result[2]
        var rb_body_len = Int64(rb_msg_len - rb_next)

        # metaDataLength = total IPC envelope size excluding body and its padding
        # (i.e., the padded header: continuation + meta_len_field + metadata + header_pad)
        var rb_meta_len = Int32(rb_next)

        var blk = _block_bytes(file_offset, rb_meta_len, rb_body_len)
        for i in range(24):
            rb_blocks_bytes.append(blk[i])
        n_blocks += 1

        for i in range(rb_msg_len):
            out.append(rb_msg[i])

        _ = rb_meta

    # ── Footer FlatBuffer ─────────────────────────────────────────────────────
    var est = len(schema.fields) * 100 + n_blocks * 32 + 256
    if est < 512:
        est = 512
    var fb = FlatBufferBuilder(est)

    # Schema table embedded in Footer (no Message envelope)
    var schema_tbl = _encode_schema_table(fb, schema)

    # dictionaries vector (empty)
    var empty_offs = List[UInt32]()
    var dicts_vec  = fb.create_vector_offsets(empty_offs)

    # recordBatches Block struct vector (24 bytes per block)
    var rb_vec_off = fb.create_vector_structs(rb_blocks_bytes, n_blocks, 24, 8)

    # Footer table
    fb.start_table()
    fb.add_field_i16(0, Int16(4))          # version = V5
    fb.add_field_offset(1, schema_tbl)
    fb.add_field_offset(2, dicts_vec)
    fb.add_field_offset(3, rb_vec_off)
    var footer_off = fb.end_table()

    var footer_bytes = fb.finish(footer_off)
    var footer_size  = Int32(len(footer_bytes))

    for i in range(len(footer_bytes)):
        out.append(footer_bytes[i])

    # ── footer_size (i32 LE) ──────────────────────────────────────────────────
    out.append(UInt8(0))
    out.append(UInt8(0))
    out.append(UInt8(0))
    out.append(UInt8(0))
    write_i32_le(out, len(out) - 4, footer_size)

    # ── Trailer magic ─────────────────────────────────────────────────────────
    for i in range(8):
        out.append(magic[i])

    return out^


fn decode_arrow_file(
    buf: List[UInt8],
) raises -> Tuple[ArrowSchema, List[RecordBatch]]:
    """
    Decode an Arrow IPC file (Feather v2).
    Returns (schema, batches).
    Raises on wrong magic, truncated file, or malformed data.
    """
    if len(buf) < 20:
        raise Error("arrow: decode_arrow_file: file too short")

    # ── Verify header magic ───────────────────────────────────────────────────
    var magic = _arrow_magic()
    for i in range(8):
        if buf[i] != magic[i]:
            raise Error("arrow: decode_arrow_file: invalid magic bytes")

    # ── Verify trailer magic ──────────────────────────────────────────────────
    var n = len(buf)
    for i in range(8):
        if buf[n - 8 + i] != magic[i]:
            raise Error("arrow: decode_arrow_file: invalid trailing magic bytes")

    # ── Read footer_size ──────────────────────────────────────────────────────
    var footer_size_i32 = read_i32_le(buf, n - 12)
    if footer_size_i32 <= Int32(0):
        raise Error("arrow: decode_arrow_file: invalid footer_size")
    var footer_size = Int(footer_size_i32)
    var footer_start = n - 12 - footer_size
    if footer_start < 8:
        raise Error("arrow: decode_arrow_file: footer out of bounds")

    # ── Parse Footer FlatBuffer ───────────────────────────────────────────────
    var footer_bytes = List[UInt8](capacity=footer_size)
    for i in range(footer_size):
        footer_bytes.append(buf[footer_start + i])

    var r = FlatBuffersReader(footer_bytes)
    var footer_tp = r.root()

    # Read embedded schema
    var schema_tp = r.union_table(footer_tp, 1)
    var endianness = r.read_i16(schema_tp, 0)
    var fields_vec  = r.read_vector(schema_tp, 1)
    var n_fields    = r.vector_len(fields_vec)
    if n_fields > UInt32(65536):
        raise Error("arrow: decode_arrow_file: field count exceeds limit")

    var fields = List[ArrowField]()
    for i in range(Int(n_fields)):
        var field_tp = r.vec_offset(fields_vec, UInt32(i))
        var name     = r.read_string(field_tp, 0)
        var nullable = r.read_bool(field_tp, 1)
        var disc     = r.union_type(field_tp, 2)
        var type_tp  = r.union_table(field_tp, 3)
        var arrow_type = decode_arrow_type(r, disc, type_tp)
        fields.append(ArrowField(name, arrow_type, nullable))

    var schema = ArrowSchema(fields, endianness)

    # ── Read RecordBatch Block vector (slot 3) ────────────────────────────────
    var rb_vec  = r.read_vector(footer_tp, 3)
    var n_rb    = r.vector_len(rb_vec)
    if n_rb > UInt32(65536):
        raise Error("arrow: decode_arrow_file: record batch count exceeds limit")

    var batches = List[RecordBatch]()
    for i in range(Int(n_rb)):
        var blk = r.vec_struct_bytes(rb_vec, UInt32(i), 24)
        var rb_file_off = _block_offset_from_bytes(blk)
        if rb_file_off < Int64(0) or Int(rb_file_off) >= len(buf):
            raise Error("arrow: decode_arrow_file: block offset out of bounds")

        # Decode the RecordBatch IPC message at rb_file_off
        var rb_start = Int(rb_file_off)
        var rb_result = decode_record_batch_message(buf, rb_start)
        var rb_length  = rb_result[0]
        var rb_next    = rb_result[4]

        # Decode arrays using schema
        var rb_nodes   = rb_result[1].copy()
        var rb_buffers = rb_result[2].copy()
        var rb_body    = rb_result[3].copy()
        _ = rb_next

        if len(rb_nodes) != len(schema.fields):
            raise Error("arrow: decode_arrow_file: node/field count mismatch in batch " + String(i))

        var arrays  = List[ArrowArray]()
        var buf_idx = 0
        for j in range(len(schema.fields)):
            var field_type = schema.fields[j].type.copy()
            var node       = rb_nodes[j].copy()

            var n_bufs: Int
            if field_type.tag == TYPE_NULL():
                n_bufs = 0
            elif field_type.tag == TYPE_UTF8() or field_type.tag == TYPE_BINARY():
                n_bufs = 3
            else:
                n_bufs = 2

            if buf_idx + n_bufs > len(rb_buffers):
                raise Error("arrow: decode_arrow_file: buffer descriptor underrun in batch " + String(i))

            var field_descs = List[BufferDesc]()
            for k in range(n_bufs):
                field_descs.append(rb_buffers[buf_idx + k].copy())
            buf_idx += n_bufs

            arrays.append(decode_array(field_type, node, field_descs, rb_body))

        batches.append(RecordBatch(rb_length, arrays))

    return Tuple[ArrowSchema, List[RecordBatch]](schema^, batches^)
