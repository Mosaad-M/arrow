from dtypes import (
    AnyDataType,
    DataType,
    PrimitiveType,
    NullType,
    BoolType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    BinaryType,
    StringType,
    dtype_to_format_string,
    dtype_from_format_string,
)
from schema import Field, Schema

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

def _ipc_continuation() -> UInt32:
    return UInt32(0xFFFFFFFF)


def ipc_pad8(size: Int) raises -> Int:
    """Return the smallest multiple of 8 >= size. Raises on negative or huge input."""
    if size < 0:
        raise Error("arrow: ipc_pad8: negative size")
    # Guard: size + 7 must not overflow (max safe = 2^62 - 1 on 64-bit)
    if size > 0x3FFF_FFFF_FFFF_FFFF:
        raise Error("arrow: ipc_pad8: size too large")
    return (size + 7) & ~7


# 1 GB hard cap per IPC message — matches flatbuffers._grow() guard philosophy
def _max_ipc_msg() -> Int:
    return 1 << 30


def encode_ipc_message(metadata: List[UInt8], body: List[UInt8]) raises -> List[UInt8]:
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
    out.extend(Span(metadata))

    # Padding after metadata
    for _ in range(meta_pad):
        out.append(UInt8(0))

    # Body bytes
    out.extend(Span(body))

    # Padding after body
    for _ in range(body_pad):
        out.append(UInt8(0))

    return out^


def decode_ipc_message(buf: List[UInt8], pos: Int) raises -> Tuple[List[UInt8], List[UInt8], Int]:
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
    metadata.extend(buf[pos + 8 : pos + 8 + meta_len])

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
        body.extend(buf[padded_header_end : padded_header_end + body_len])
    var padded_body_end = padded_header_end + ipc_pad8(body_len)
    return Tuple[List[UInt8], List[UInt8], Int](metadata^, body^, padded_body_end)


def encode_eos() -> List[UInt8]:
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

def TYPE_NULL() -> UInt8:
    return UInt8(1)

def TYPE_INT() -> UInt8:
    return UInt8(2)

def TYPE_FLOAT() -> UInt8:
    return UInt8(3)

def TYPE_BINARY() -> UInt8:
    return UInt8(4)

def TYPE_UTF8() -> UInt8:
    return UInt8(5)

def TYPE_BOOL() -> UInt8:
    return UInt8(6)

def TYPE_LIST() -> UInt8:
    """Real Arrow Type union tag 12 (List) -- matches upstream numbering
    exactly (7-11 are Decimal/Date/Time/Timestamp/Interval, unsupported
    here), not an arbitrary local choice, since real pyarrow interop
    depends on this matching the actual Arrow IPC spec."""
    return UInt8(12)


struct ArrowInt(Copyable, Movable):
    """Metadata for Int Arrow type."""
    var bit_width: Int32
    var is_signed: Bool

    def __init__(out self, bit_width: Int32, is_signed: Bool):
        self.bit_width = bit_width
        self.is_signed = is_signed

    def __copyinit__(out self, copy: Self):
        self.bit_width = copy.bit_width
        self.is_signed = copy.is_signed

    def __moveinit__(out self, deinit take: Self):
        self.bit_width = take.bit_width
        self.is_signed = take.is_signed

    def copy(self) -> Self:
        return Self(self.bit_width, self.is_signed)


struct ArrowFloat(Copyable, Movable):
    """Metadata for FloatingPoint Arrow type."""
    var precision: UInt16   # 1=Single, 2=Double

    def __init__(out self, precision: UInt16):
        self.precision = precision

    def __copyinit__(out self, copy: Self):
        self.precision = copy.precision

    def __moveinit__(out self, deinit take: Self):
        self.precision = take.precision

    def copy(self) -> Self:
        return Self(self.precision)


struct ArrowType(Copyable, Movable):
    """A tagged Arrow column type with optional Int or Float metadata."""
    var tag: UInt8
    var int_meta: ArrowInt
    var float_meta: ArrowFloat

    def __init__(out self, tag: UInt8, int_bit_width: Int32, int_is_signed: Bool, float_precision: UInt16):
        self.tag = tag
        self.int_meta = ArrowInt(int_bit_width, int_is_signed)
        self.float_meta = ArrowFloat(float_precision)

    def __copyinit__(out self, copy: Self):
        self.tag = copy.tag
        self.int_meta = copy.int_meta.copy()
        self.float_meta = copy.float_meta.copy()

    def __moveinit__(out self, deinit take: Self):
        self.tag = take.tag
        self.int_meta = take.int_meta.copy()
        self.float_meta = take.float_meta.copy()

    def copy(self) -> Self:
        return Self(self.tag, self.int_meta.bit_width, self.int_meta.is_signed, self.float_meta.precision)

    @staticmethod
    def null() -> ArrowType:
        return ArrowType(TYPE_NULL(), 0, False, 0)

    @staticmethod
    def int_(bit_width: Int32, is_signed: Bool) -> ArrowType:
        return ArrowType(TYPE_INT(), bit_width, is_signed, 0)

    @staticmethod
    def float_(precision: UInt16) -> ArrowType:
        return ArrowType(TYPE_FLOAT(), 0, False, precision)

    @staticmethod
    def binary() -> ArrowType:
        return ArrowType(TYPE_BINARY(), 0, False, 0)

    @staticmethod
    def utf8() -> ArrowType:
        return ArrowType(TYPE_UTF8(), 0, False, 0)

    @staticmethod
    def bool_() -> ArrowType:
        return ArrowType(TYPE_BOOL(), 0, False, 0)

    @staticmethod
    def list_utf8() -> ArrowType:
        """List<Utf8> only -- not a general nested-list-of-any-type. The
        List type's own FlatBuffer table carries no scalar metadata in
        the Arrow IPC spec (the child's type lives in the Field's
        children vector, written by encode_schema_message), so this
        reuses the same empty-table encoding every other non-Int/Float
        type already gets."""
        return ArrowType(TYPE_LIST(), 0, False, 0)


def encode_arrow_type(mut b: FlatBufferBuilder, t: ArrowType) raises -> Tuple[UInt8, UInt32]:
    """
    Builds the type table in b and returns (discriminant, table_offset).
    Build order: type table must be built before the Field table that references it.
    """
    # S5: validate tag before encoding to prevent silent type confusion.
    # TYPE_LIST() (12) is allowed as a specific extra value, not by widening
    # the contiguous range, since 7-11 (Decimal/Date/Time/Timestamp/Interval)
    # are still genuinely unsupported.
    var disc = t.tag
    if (disc < TYPE_NULL() or disc > TYPE_BOOL()) and disc != TYPE_LIST():
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


def decode_arrow_type(r: FlatBuffersReader, discriminant: UInt8, type_tp: UInt32) raises -> ArrowType:
    """Reads the type table at type_tp and returns an ArrowType."""
    # P1: locals eliminate repeated function calls
    var T_NULL  = UInt8(1)
    var T_INT   = UInt8(2)
    var T_FLOAT = UInt8(3)
    var T_BIN   = UInt8(4)
    var T_UTF8  = UInt8(5)
    var T_BOOL  = UInt8(6)
    var T_LIST  = TYPE_LIST()

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
    elif discriminant == T_LIST:
        # Scoped to List<Utf8> only: the child's actual type isn't read
        # back from the Field's children vector here, since this project
        # only ever produces/expects a Utf8 child -- the type is implied
        # by the tag itself, not re-derived from the wire.
        return ArrowType.list_utf8()
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

    def __init__(out self, name: String, type: ArrowType, nullable: Bool):
        self.name = name
        self.type = type.copy()
        self.nullable = nullable

    def __copyinit__(out self, copy: Self):
        self.name = copy.name
        self.type = copy.type.copy()
        self.nullable = copy.nullable

    def __moveinit__(out self, deinit take: Self):
        self.name = take.name^
        self.type = take.type^
        self.nullable = take.nullable

    def copy(self) -> Self:
        return Self(self.name, self.type.copy(), self.nullable)


struct ArrowSchema(Copyable, Movable):
    """Schema: ordered list of fields and endianness."""
    var fields: List[ArrowField]
    var endianness: Int16   # 0 = little-endian, 1 = big-endian

    def __init__(out self, fields: List[ArrowField], endianness: Int16 = Int16(0)):
        self.fields = List[ArrowField]()
        for i in range(len(fields)):
            self.fields.append(fields[i].copy())
        self.endianness = endianness

    def __copyinit__(out self, copy: Self):
        self.fields = List[ArrowField]()
        for i in range(len(copy.fields)):
            self.fields.append(copy.fields[i].copy())
        self.endianness = copy.endianness

    def __moveinit__(out self, deinit take: Self):
        self.fields = take.fields^
        self.endianness = take.endianness


def _encode_field(mut b: FlatBufferBuilder, f: ArrowField) raises -> UInt32:
    """Builds one Field table (with its List<Utf8> synthetic "item" child,
    if applicable) and returns its offset. Shared by encode_schema_message
    (the leading Schema message) and _encode_schema_table (the Footer's
    embedded schema, what real pyarrow actually reads for a Feather v2
    file) so the two encoders can't silently diverge."""
    var children_vec_off = UInt32(0)
    var has_children = False
    if f.type.tag == TYPE_LIST():
        var child_type_result = encode_arrow_type(b, ArrowType.utf8())
        var child_type_disc = child_type_result[0]
        var child_type_off = child_type_result[1]
        var child_name_off = b.create_string("item")
        b.start_table()
        b.add_field_offset(0, child_name_off)
        b.add_field_bool(1, True)
        b.add_field_u8(2, child_type_disc)
        b.add_field_offset(3, child_type_off)
        var child_field_off = b.end_table()
        var child_offs = List[UInt32]()
        child_offs.append(child_field_off)
        children_vec_off = b.create_vector_offsets(child_offs)
        has_children = True

    var type_result = encode_arrow_type(b, f.type)
    var type_disc = type_result[0]
    var type_off = type_result[1]
    var name_off = b.create_string(f.name)
    b.start_table()
    b.add_field_offset(0, name_off)
    b.add_field_bool(1, f.nullable)
    b.add_field_u8(2, type_disc)
    b.add_field_offset(3, type_off)
    # Real Arrow's Field table (Schema.fbs) is: name=0, nullable=1,
    # type_type=2, type=3, dictionary=4, children=5, custom_metadata=6 --
    # slot 4 is `dictionary`, NOT `children` (confirmed against a real
    # pyarrow Footer-verification failure when this was first written at
    # slot 4: "Verification of flatbuffer-encoded Footer failed", only
    # fixed once moved to the correct slot 5). No dictionary field is ever
    # written here, so slot 4 stays legitimately absent.
    if has_children:
        b.add_field_offset(5, children_vec_off)
    return b.end_table()


def encode_schema_message(schema: ArrowSchema) raises -> List[UInt8]:
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
        field_offs.append(_encode_field(b, f))

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


def decode_schema_message(buf: List[UInt8], pos: Int) raises -> Tuple[ArrowSchema, Int]:
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

    def __init__(out self, length: Int64, null_count: Int64):
        self.length = length
        self.null_count = null_count

    def __copyinit__(out self, copy: Self):
        self.length = copy.length
        self.null_count = copy.null_count

    def __moveinit__(out self, deinit take: Self):
        self.length = take.length
        self.null_count = take.null_count

    def copy(self) -> Self:
        return Self(self.length, self.null_count)


struct BufferDesc(Copyable, Movable):
    """Describes one buffer's byte offset and length within an IPC message body."""

    var offset: Int64
    var length: Int64

    def __init__(out self, offset: Int64, length: Int64):
        self.offset = offset
        self.length = length

    def __copyinit__(out self, copy: Self):
        self.offset = copy.offset
        self.length = copy.length

    def __moveinit__(out self, deinit take: Self):
        self.offset = take.offset
        self.length = take.length

    def copy(self) -> Self:
        return Self(self.offset, self.length)


def _field_node_bytes(node: FieldNode) -> List[UInt8]:
    """Serialize a FieldNode as 16 LE bytes: [length:i64][null_count:i64]."""
    var result = List[UInt8](capacity=16)
    for _ in range(16):
        result.append(UInt8(0))
    write_i64_le(result, 0, node.length)
    write_i64_le(result, 8, node.null_count)
    return result^


def _buffer_desc_bytes(bd: BufferDesc) -> List[UInt8]:
    """Serialize a BufferDesc as 16 LE bytes: [offset:i64][length:i64]."""
    var result = List[UInt8](capacity=16)
    for _ in range(16):
        result.append(UInt8(0))
    write_i64_le(result, 0, bd.offset)
    write_i64_le(result, 8, bd.length)
    return result^


def _field_node_from_bytes(data: List[UInt8]) raises -> FieldNode:
    """Deserialize a FieldNode from 16 LE bytes."""
    return FieldNode(read_i64_le(data, 0), read_i64_le(data, 8))


def _buffer_desc_from_bytes(data: List[UInt8]) raises -> BufferDesc:
    """Deserialize a BufferDesc from 16 LE bytes."""
    return BufferDesc(read_i64_le(data, 0), read_i64_le(data, 8))


def encode_record_batch_message(
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
    return encode_ipc_message(
        _record_batch_metadata(length, nodes, buffers, len(body)), body
    )


def _record_batch_metadata(
    length: Int64,
    nodes: List[FieldNode],
    buffers: List[BufferDesc],
    body_len: Int,
) raises -> List[UInt8]:
    """The Message FlatBuffer for a RecordBatch (no IPC envelope)."""
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
    b.add_field_i64(3, Int64(body_len))
    var msg_off = b.end_table()

    return b.finish(msg_off)


def decode_record_batch_message(
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

    # List<Utf8> only (type.tag == TYPE_LIST()): the child Utf8 array's own
    # buffers, as flat inline fields rather than a nested ArrowArray --
    # confirmed against the compiler that a directly self-referential
    # `List[Self]` struct field is rejected ("field has non-'Deinitable'
    # type"), so this avoids that recursion entirely. Empty/zero for every
    # non-List type.
    var child_length: Int
    var child_null_count: Int
    var child_validity: List[UInt8]
    var child_offsets: List[UInt8]
    var child_values: List[UInt8]

    def __init__(
        out self,
        type: ArrowType,
        length: Int,
        null_count: Int,
        var validity: List[UInt8],
        var offsets: List[UInt8],
        var values: List[UInt8],
        child_length: Int = 0,
        child_null_count: Int = 0,
        var child_validity: List[UInt8] = List[UInt8](),
        var child_offsets: List[UInt8] = List[UInt8](),
        var child_values: List[UInt8] = List[UInt8](),
    ):
        """Takes ownership of every buffer: pass `buf^` to hand a builder's
        buffer over without copying it, or `buf.copy()` to keep your own."""
        self.type = type.copy()
        self.length = length
        self.null_count = null_count
        self.validity = validity^
        self.offsets = offsets^
        self.values = values^
        self.child_length = child_length
        self.child_null_count = child_null_count
        self.child_validity = child_validity^
        self.child_offsets = child_offsets^
        self.child_values = child_values^

    def __copyinit__(out self, copy: Self):
        self.type = copy.type.copy()
        self.length = copy.length
        self.null_count = copy.null_count
        self.validity = copy.validity.copy()
        self.offsets = copy.offsets.copy()
        self.values = copy.values.copy()
        self.child_length = copy.child_length
        self.child_null_count = copy.child_null_count
        self.child_validity = copy.child_validity.copy()
        self.child_offsets = copy.child_offsets.copy()
        self.child_values = copy.child_values.copy()

    def __moveinit__(out self, deinit take: Self):
        self.type = take.type^
        self.length = take.length
        self.null_count = take.null_count
        self.validity = take.validity^
        self.offsets = take.offsets^
        self.values = take.values^
        self.child_length = take.child_length
        self.child_null_count = take.child_null_count
        self.child_validity = take.child_validity^
        self.child_offsets = take.child_offsets^
        self.child_values = take.child_values^

    def copy(self) -> Self:
        return Self(copy=self)

    def _into_list_child(
        deinit self,
        length: Int,
        null_count: Int,
        var validity: List[UInt8],
        var offsets: List[UInt8],
    ) -> ArrowArray:
        """Consume this Utf8 array as the child of a new List<Utf8> column,
        moving its buffers rather than copying them."""
        return ArrowArray(
            ArrowType.list_utf8(), length, null_count,
            validity^, offsets^, List[UInt8](),
            self.length, self.null_count,
            self.validity^, self.offsets^, self.values^,
        )

    @staticmethod
    def list_utf8(
        length: Int,
        null_count: Int,
        var validity: List[UInt8],
        var offsets: List[UInt8],
        child_length: Int,
        child_null_count: Int,
        var child_validity: List[UInt8],
        var child_offsets: List[UInt8],
        var child_values: List[UInt8],
    ) -> ArrowArray:
        """Convenience constructor for a List<Utf8> column: `offsets` here
        is the LIST's own offsets buffer (int32, length+1 entries, indexing
        into the child array's ELEMENTS, not bytes) -- distinct from a
        plain Utf8 array's offsets, which index into byte content."""
        return ArrowArray(
            ArrowType.list_utf8(), length, null_count,
            validity^, offsets^, List[UInt8](),
            child_length, child_null_count,
            child_validity^, child_offsets^, child_values^,
        )


# ── Column body layout and emission ─────────────────────────────────────────
#
# A RecordBatch body is every column's buffers, in schema order, each padded
# to 8 bytes. _layout_array computes the FieldNodes and BufferDescs from
# buffer LENGTHS alone (no bytes touched), so a message header can be built
# before any body bytes exist; _emit_array then writes the bytes, in the
# same order, into whatever _ByteSink the caller has: the output List for
# encode_arrow_file, or the file itself for ArrowFileWriter. The body is
# never assembled as a separate buffer. Both functions walk the buffers in
# the same order and must be kept in step.


trait _ByteSink:
    def put(mut self, data: List[UInt8]) raises:
        ...

    def put_zeros(mut self, n: Int) raises:
        ...


struct _ListSink(_ByteSink, Movable):
    var buf: List[UInt8]

    def __init__(out self, var buf: List[UInt8]):
        self.buf = buf^

    def take(deinit self) -> List[UInt8]:
        return self.buf^

    def put(mut self, data: List[UInt8]) raises:
        self.buf.extend(Span(data))

    def put_zeros(mut self, n: Int) raises:
        for _ in range(n):
            self.buf.append(UInt8(0))


struct _FileSink(_ByteSink, Movable):
    var file: FileHandle

    def __init__(out self, var file: FileHandle):
        self.file = file^

    def put(mut self, data: List[UInt8]) raises:
        self.file.write_bytes(Span(data))

    def put_zeros(mut self, n: Int) raises:
        var zeros = List[UInt8](length=n, fill=UInt8(0))
        self.file.write_bytes(Span(zeros))


def _layout_buffer(mut descs: List[BufferDesc], mut cur: Int, size: Int) raises:
    descs.append(BufferDesc(Int64(cur), Int64(size)))
    cur += ipc_pad8(size)


def _layout_array(
    arr: ArrowArray, mut nodes: List[FieldNode], mut descs: List[BufferDesc], mut cur: Int
) raises:
    """Append `arr`'s FieldNode(s) and BufferDescs, advancing `cur` (the
    absolute body offset) past its padded buffers. A List<Utf8> column
    contributes two nodes (list, then child: Arrow's depth-first layout)."""
    if arr.type.tag == TYPE_LIST():
        nodes.append(FieldNode(Int64(arr.length), Int64(arr.null_count)))
        _layout_buffer(descs, cur, len(arr.validity) if arr.null_count > 0 else 0)
        _layout_buffer(descs, cur, len(arr.offsets))
        nodes.append(FieldNode(Int64(arr.child_length), Int64(arr.child_null_count)))
        _layout_buffer(descs, cur, len(arr.child_validity) if arr.child_null_count > 0 else 0)
        _layout_buffer(descs, cur, len(arr.child_offsets))
        _layout_buffer(descs, cur, len(arr.child_values))
        return
    nodes.append(FieldNode(Int64(arr.length), Int64(arr.null_count)))
    if arr.type.tag == TYPE_NULL():
        return
    # Absent validity is a zero-length descriptor with no bytes in the body.
    _layout_buffer(descs, cur, len(arr.validity) if arr.null_count > 0 else 0)
    if arr.type.tag == TYPE_UTF8() or arr.type.tag == TYPE_BINARY():
        _layout_buffer(descs, cur, len(arr.offsets))
    _layout_buffer(descs, cur, len(arr.values))


def _emit_buffer[S: _ByteSink](mut sink: S, data: List[UInt8]) raises:
    sink.put(data)
    sink.put_zeros(ipc_pad8(len(data)) - len(data))


def _emit_array[S: _ByteSink](mut sink: S, arr: ArrowArray) raises:
    """Write `arr`'s body bytes in exactly _layout_array's order."""
    if arr.type.tag == TYPE_LIST():
        if arr.null_count > 0:
            _emit_buffer(sink, arr.validity)
        _emit_buffer(sink, arr.offsets)
        if arr.child_null_count > 0:
            _emit_buffer(sink, arr.child_validity)
        _emit_buffer(sink, arr.child_offsets)
        _emit_buffer(sink, arr.child_values)
        return
    if arr.type.tag == TYPE_NULL():
        return
    if arr.null_count > 0:
        _emit_buffer(sink, arr.validity)
    if arr.type.tag == TYPE_UTF8() or arr.type.tag == TYPE_BINARY():
        _emit_buffer(sink, arr.offsets)
    _emit_buffer(sink, arr.values)


def encode_array(
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
    if arr.type.tag == TYPE_LIST():
        raise Error("arrow: encode_array: List columns use encode_list_utf8_array")
    var nodes = List[FieldNode]()
    var descs = List[BufferDesc]()
    var cur = body_offset
    _layout_array(arr, nodes, descs, cur)
    var sink = _ListSink(List[UInt8](capacity=cur - body_offset))
    _emit_array(sink, arr)
    return Tuple[FieldNode, List[BufferDesc], List[UInt8]](nodes[0].copy(), descs^, sink^.take())


def encode_list_utf8_array(
    arr: ArrowArray, body_offset: Int
) raises -> Tuple[List[FieldNode], List[BufferDesc], List[UInt8]]:
    """Encode a List<Utf8> ArrowArray (see ArrowArray.list_utf8) into its
    two FieldNodes (list, then child) and five BufferDescs (list validity,
    list offsets, child validity, child offsets, child values) -- the
    depth-first pre-order layout real Arrow's IPC format uses for nested
    types."""
    if arr.type.tag != TYPE_LIST():
        raise Error("arrow: encode_list_utf8_array: expected List type")
    var nodes = List[FieldNode]()
    var descs = List[BufferDesc]()
    var cur = body_offset
    _layout_array(arr, nodes, descs, cur)
    var sink = _ListSink(List[UInt8](capacity=cur - body_offset))
    _emit_array(sink, arr)
    return Tuple[List[FieldNode], List[BufferDesc], List[UInt8]](nodes^, descs^, sink^.take())


def _checked_slice_bounds(
    offset: Int64, length: Int64, body_len: Int, context: String
) raises -> Tuple[Int, Int]:
    """Overflow-safe [start, end) bounds for a body slice, given an
    attacker-controlled (offset, length) BufferDesc pair straight from a
    parsed IPC file. Real, confirmed exploit (see tasks/lessons.md): the
    previous per-call-site pattern computed `end = start + length` and
    checked `end > body_len` -- but a crafted offset/length pair near
    Int64::MAX makes that addition silently WRAP (Mojo's Int has no
    overflow trap), producing a small or negative `end` that slips past
    the check, then crashes on the out-of-bounds slice with the original
    huge `start` still intact. Subtraction avoids this: `length >
    body_len - start` can't wrap the same way, since body_len is always
    small and non-negative and start is already checked non-negative
    first, so `body_len - start` is always representable. Same pattern
    decode_ipc_message already used for its own meta_len/body_len checks
    (`meta_len > len(buf) - pos - 8`), now applied consistently to every
    buffer-descriptor bounds check that does this same start+length
    computation."""
    if offset < Int64(0):
        raise Error("arrow: " + context + ": negative buffer offset")
    if length < Int64(0):
        raise Error("arrow: " + context + ": negative buffer length")
    var start = Int(offset)
    if Int(length) > body_len - start:
        raise Error("arrow: " + context + ": buffer out of bounds")
    return Tuple[Int, Int](start, start + Int(length))


def decode_list_utf8_array(
    nodes: List[FieldNode],
    descs: List[BufferDesc],
    body: List[UInt8],
) raises -> ArrowArray:
    """Reconstruct a List<Utf8> ArrowArray from its two FieldNodes
    (nodes[0] = list, nodes[1] = child) and five BufferDescs, in the same
    layout encode_list_utf8_array produces. Additive counterpart to
    decode_array."""
    if len(nodes) < 2:
        raise Error("arrow: decode_list_utf8_array: expected 2 field nodes (list + child)")
    if len(descs) < 5:
        raise Error("arrow: decode_list_utf8_array: expected 5 buffer descriptors")

    var list_node = nodes[0].copy()
    var child_node = nodes[1].copy()

    # descs[0] (validity) is always bounds-checked, even when its length is
    # 0 (an absent validity buffer) -- a negative length must still be
    # rejected regardless of whether anything is actually sliced.
    var vbounds = _checked_slice_bounds(
        descs[0].offset, descs[0].length, len(body), "decode_list_utf8_array: validity buffer"
    )
    var validity = List[UInt8]()
    if descs[0].length > Int64(0):
        validity.extend(body[vbounds[0] : vbounds[1]])

    var obounds = _checked_slice_bounds(
        descs[1].offset, descs[1].length, len(body), "decode_list_utf8_array: offsets buffer"
    )
    var offsets = List[UInt8]()
    offsets.extend(body[obounds[0] : obounds[1]])

    var child_descs = List[BufferDesc]()
    child_descs.append(descs[2].copy())
    child_descs.append(descs[3].copy())
    child_descs.append(descs[4].copy())
    var child = decode_array(ArrowType.utf8(), child_node, child_descs, body)
    return child^._into_list_child(
        Int(list_node.length), Int(list_node.null_count), validity^, offsets^
    )


def decode_array(
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
    # Always bounds-checked (via the overflow-safe helper), even when the
    # length is 0 (an absent validity buffer) -- see _checked_slice_bounds.
    var vbounds = _checked_slice_bounds(
        descs[0].offset, descs[0].length, len(body), "decode_array: validity buffer"
    )
    if descs[0].length > Int64(0):
        validity.extend(body[vbounds[0] : vbounds[1]])

    # ── Offsets + values (Utf8 / Binary) or just values (all other types) ────
    if type.tag == TYPE_UTF8() or type.tag == TYPE_BINARY():
        if len(descs) < 3:
            raise Error("arrow: decode_array: expected 3 buffer descriptors for variable-length type")
        var obounds = _checked_slice_bounds(
            descs[1].offset, descs[1].length, len(body), "decode_array: offsets buffer"
        )
        offsets.extend(body[obounds[0] : obounds[1]])
        var vabounds = _checked_slice_bounds(
            descs[2].offset, descs[2].length, len(body), "decode_array: values buffer"
        )
        values.extend(body[vabounds[0] : vabounds[1]])
    else:
        if len(descs) < 2:
            raise Error("arrow: decode_array: expected 2 buffer descriptors for fixed-width type")
        var vabounds = _checked_slice_bounds(
            descs[1].offset, descs[1].length, len(body), "decode_array: values buffer"
        )
        values.extend(body[vabounds[0] : vabounds[1]])

    return ArrowArray(
        type, Int(node.length), Int(node.null_count),
        validity^, offsets^, values^,
    )


def _layout_record_batch(
    schema: ArrowSchema, arrays: List[ArrowArray]
) raises -> Tuple[List[FieldNode], List[BufferDesc], Int]:
    """(nodes, buffer descs, padded body length) for a RecordBatch, from
    buffer lengths only."""
    if len(arrays) != len(schema.fields):
        raise Error("arrow: encode_record_batch: column count does not match schema field count")
    if len(arrays) > 65536:
        raise Error("arrow: encode_record_batch: too many columns (> 65536)")
    var nodes = List[FieldNode]()
    var descs = List[BufferDesc]()
    var cur = 0
    for i in range(len(arrays)):
        _layout_array(arrays[i], nodes, descs, cur)
        # S-P5-2: guard against body offset overflow in large multi-column batches
        if cur > _max_ipc_msg():
            raise Error("arrow: encode_record_batch: combined column body exceeds 1 GB")
    return Tuple[List[FieldNode], List[BufferDesc], Int](nodes^, descs^, cur)


def _record_batch_header(
    schema: ArrowSchema, arrays: List[ArrowArray]
) raises -> Tuple[List[UInt8], Int]:
    """The IPC envelope of a RecordBatch message up to (not including) its
    body: continuation marker, metadata length, metadata, padding. Returns
    (header bytes, body length). Header then _emit_array for each column
    is byte-identical to encode_record_batch's output."""
    var layout = _layout_record_batch(schema, arrays)
    var row_count = Int64(arrays[0].length) if len(arrays) > 0 else Int64(0)
    var metadata = _record_batch_metadata(row_count, layout[0], layout[1], layout[2])
    if len(metadata) > _max_ipc_msg():
        raise Error("arrow: encode_ipc_message: message too large (> 1 GB)")
    var header_size = 8 + len(metadata)
    var header = List[UInt8](capacity=ipc_pad8(header_size))
    for _ in range(8):
        header.append(UInt8(0))
    write_u32_le(header, 0, UInt32(0xFFFFFFFF))
    write_i32_le(header, 4, Int32(len(metadata)))
    header.extend(Span(metadata))
    for _ in range(ipc_pad8(header_size) - header_size):
        header.append(UInt8(0))
    return Tuple[List[UInt8], Int](header^, layout[2])


def encode_record_batch(
    schema: ArrowSchema,
    arrays: List[ArrowArray],
) raises -> List[UInt8]:
    """
    Encode a full RecordBatch IPC message from a schema and its column arrays.
    Column bytes are copied exactly once, straight into the returned buffer.
    """
    var hdr = _record_batch_header(schema, arrays)
    var sink = _ListSink(List[UInt8](capacity=len(hdr[0]) + hdr[1]))
    sink.put(hdr[0])
    for i in range(len(arrays)):
        _emit_array(sink, arrays[i])
    return sink^.take()


def decode_record_batch(
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

    # Unlike scalar types (always exactly 1 FieldNode per schema field),
    # List<Utf8> emits 2 (list + child) -- so node_idx and buf_idx both
    # advance independently per column, not in lockstep with the schema
    # field index the way a strict 1:1 assumption would allow.
    var arrays   = List[ArrowArray]()
    var buf_idx  = 0
    var node_idx = 0

    for i in range(len(schema.fields)):
        var field_type = schema.fields[i].type.copy()

        if field_type.tag == TYPE_LIST():
            if node_idx + 2 > len(nodes):
                raise Error("arrow: decode_record_batch: node underrun for List column " + String(i))
            if buf_idx + 5 > len(buffers):
                raise Error("arrow: decode_record_batch: buffer descriptor underrun for List column " + String(i))
            var field_nodes = List[FieldNode]()
            field_nodes.append(nodes[node_idx].copy())
            field_nodes.append(nodes[node_idx + 1].copy())
            node_idx += 2
            var field_descs = List[BufferDesc]()
            for j in range(5):
                field_descs.append(buffers[buf_idx + j].copy())
            buf_idx += 5
            arrays.append(decode_list_utf8_array(field_nodes, field_descs, body))
            continue

        if node_idx + 1 > len(nodes):
            raise Error("arrow: decode_record_batch: node underrun for column " + String(i))
        var node = nodes[node_idx].copy()
        node_idx += 1

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

    if node_idx != len(nodes):
        raise Error("arrow: decode_record_batch: unconsumed field nodes remain")

    return Tuple[List[ArrowArray], Int](arrays^, next_pos)


# ============================================================================
# Phase 6 — IPC File Format (Feather v2 / Arrow IPC File)
# ============================================================================


struct RecordBatch(Copyable, Movable):
    """A batch of typed columns sharing the same row count."""

    var length: Int64
    var columns: List[ArrowArray]

    def __init__(out self, length: Int64, var columns: List[ArrowArray]):
        """Takes ownership of `columns`: pass `cols^` to avoid a copy."""
        self.length = length
        self.columns = columns^

    def __copyinit__(out self, copy: Self):
        self.length = copy.length
        self.columns = List[ArrowArray]()
        for i in range(len(copy.columns)):
            self.columns.append(copy.columns[i].copy())

    def __moveinit__(out self, deinit take: Self):
        self.length = take.length
        self.columns = take.columns^

    def copy(self) -> Self:
        return Self(copy=self)


def _arrow_magic() -> List[UInt8]:
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


def _encode_schema_table(mut b: FlatBufferBuilder, schema: ArrowSchema) raises -> UInt32:
    """
    Build the Schema FlatBuffer table into b (no Message envelope).
    Returns the UOffset of the Schema table.
    """
    if len(schema.fields) > 65536:
        raise Error("arrow: _encode_schema_table: too many fields (> 65536)")

    var field_offs = List[UInt32]()
    for i in range(len(schema.fields)):
        var f = schema.fields[i].copy()
        field_offs.append(_encode_field(b, f))

    var fields_vec_off = b.create_vector_offsets(field_offs)
    b.start_table()
    b.add_field_i16(0, schema.endianness)
    b.add_field_offset(1, fields_vec_off)
    return b.end_table()


def _block_bytes(file_offset: Int64, meta_len: Int32, body_len: Int64) -> List[UInt8]:
    """Serialize one Block struct as 24 LE bytes: [offset:i64][metaLen:i32][pad:i32][bodyLen:i64]."""
    var result = List[UInt8](capacity=24)
    for _ in range(24):
        result.append(UInt8(0))
    write_i64_le(result, 0, file_offset)
    write_i32_le(result, 8, meta_len)
    # bytes 12-15: padding (zero)
    write_i64_le(result, 16, body_len)
    return result^


def _block_offset_from_bytes(data: List[UInt8]) raises -> Int64:
    return read_i64_le(data, 0)


def _block_meta_len_from_bytes(data: List[UInt8]) raises -> Int32:
    return read_i32_le(data, 8)


def _block_body_len_from_bytes(data: List[UInt8]) raises -> Int64:
    return read_i64_le(data, 16)


def _encode_file_header(schema: ArrowSchema) raises -> List[UInt8]:
    """The bytes an Arrow IPC file starts with: 8-byte magic + Schema IPC
    message. Shared by encode_arrow_file and ArrowFileWriter."""
    var out = _arrow_magic()
    out.extend(encode_schema_message(schema))
    return out^


def _emit_file_batch[S: _ByteSink](
    mut sink: S, schema: ArrowSchema, batch: RecordBatch, file_offset: Int, mut blocks: List[UInt8]
) raises -> Int:
    """Write one RecordBatch IPC message destined for byte `file_offset` of
    the file into `sink`, appending its 24-byte footer Block to `blocks`.
    Returns the number of bytes written. Shared by encode_arrow_file and
    ArrowFileWriter."""
    var hdr = _record_batch_header(schema, batch.columns)
    # Block.metaDataLength is the padded header; Block.bodyLength must equal
    # the Message's own bodyLength (pyarrow rejects a mismatch).
    blocks.extend(_block_bytes(Int64(file_offset), Int32(len(hdr[0])), Int64(hdr[1])))
    sink.put(hdr[0])
    for c in range(len(batch.columns)):
        _emit_array(sink, batch.columns[c])
    return len(hdr[0]) + hdr[1]


def _encode_file_footer(
    schema: ArrowSchema, blocks: List[UInt8], n_blocks: Int
) raises -> List[UInt8]:
    """The bytes an Arrow IPC file ends with: Footer FlatBuffer (schema +
    one Block per RecordBatch) + footer_size + 6-byte trailing magic.
    Shared by encode_arrow_file and ArrowFileWriter."""
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
    var rb_vec_off = fb.create_vector_structs(blocks, n_blocks, 24, 8)

    # Footer table
    fb.start_table()
    fb.add_field_i16(0, Int16(4))          # version = V5
    fb.add_field_offset(1, schema_tbl)
    fb.add_field_offset(2, dicts_vec)
    fb.add_field_offset(3, rb_vec_off)
    var footer_off = fb.end_table()

    var out = fb.finish(footer_off)
    var footer_size = Int32(len(out))

    # ── footer_size (i32 LE) ──────────────────────────────────────────────────
    for _ in range(4):
        out.append(UInt8(0))
    write_i32_le(out, len(out) - 4, footer_size)

    # ── Trailer magic ─────────────────────────────────────────────────────────
    # Per the Arrow IPC File Format spec, the trailing magic is exactly 6
    # bytes ("ARROW1", unpadded) — unlike the leading magic, which is 8
    # bytes padded for alignment. Do not reuse the full 8-byte magic here.
    var magic = _arrow_magic()
    for i in range(6):
        out.append(magic[i])

    return out^


def encode_arrow_file(
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
      [magic: 6]  (unpadded trailing magic, NOT the same 8-byte header magic)

    Holds the whole file in memory; use ArrowFileWriter to write batches
    to disk one at a time instead. Column bytes are copied exactly once,
    straight into the returned buffer.
    """
    # Reserve the whole file up front: body sizes are known from the layout
    # alone, and growing by doubling would transiently hold ~2x the file.
    # Headers and footer are small; the slack covers them.
    var header = _encode_file_header(schema)
    var total = len(header) + 4096 + len(batches) * 1024 + len(schema.fields) * 256
    for b in range(len(batches)):
        total += _layout_record_batch(schema, batches[b].columns)[2]
    var out = List[UInt8](capacity=total)
    out.extend(Span(header))
    var sink = _ListSink(out^)
    var blocks = List[UInt8]()
    for b in range(len(batches)):
        _ = _emit_file_batch(sink, schema, batches[b], len(sink.buf), blocks)
    sink.put(_encode_file_footer(schema, blocks, len(batches)))
    return sink^.take()


struct ArrowFileWriter(Movable):
    """Writes an Arrow IPC file (Feather v2) to disk one RecordBatch at a
    time, producing exactly the bytes encode_arrow_file would for the same
    batches. Only the current batch and the footer's 24-byte-per-batch
    Block list are held in memory, so a file of any size can be written
    in bounded memory.

    Usage: construct (writes magic + schema), write_batch() any number of
    times, then finish() (writes the footer). A file that is never
    finish()ed is left truncated and is not a valid Arrow file.
    """

    var _sink: _FileSink
    var _schema: ArrowSchema
    var _pos: Int
    var _blocks: List[UInt8]
    var _n_blocks: Int
    var _finished: Bool

    def __init__(out self, path: String, schema: ArrowSchema) raises:
        self._sink = _FileSink(open(path, "w"))
        self._schema = schema.copy()
        self._blocks = List[UInt8]()
        self._n_blocks = 0
        self._finished = False
        var header = _encode_file_header(schema)
        self._sink.put(header)
        self._pos = len(header)

    def write_batch(mut self, batch: RecordBatch) raises:
        if self._finished:
            raise Error("arrow: ArrowFileWriter.write_batch: writer already finished")
        self._pos += _emit_file_batch(self._sink, self._schema, batch, self._pos, self._blocks)
        self._n_blocks += 1

    def finish(mut self) raises:
        if self._finished:
            raise Error("arrow: ArrowFileWriter.finish: writer already finished")
        self._finished = True
        self._sink.put(_encode_file_footer(self._schema, self._blocks, self._n_blocks))
        self._sink.file.close()


def decode_arrow_file(
    buf: List[UInt8],
) raises -> Tuple[ArrowSchema, List[RecordBatch]]:
    """
    Decode an Arrow IPC file (Feather v2).
    Returns (schema, batches).
    Raises on wrong magic, truncated file, or malformed data.
    """
    if len(buf) < 18:
        raise Error("arrow: decode_arrow_file: file too short")

    # ── Verify header magic ───────────────────────────────────────────────────
    var magic = _arrow_magic()
    for i in range(8):
        if buf[i] != magic[i]:
            raise Error("arrow: decode_arrow_file: invalid magic bytes")

    # ── Verify trailer magic ──────────────────────────────────────────────────
    # The trailer is exactly 6 bytes ("ARROW1", unpadded), unlike the 8-byte
    # padded header magic — see encode_arrow_file's layout comment.
    var n = len(buf)
    for i in range(6):
        if buf[n - 6 + i] != magic[i]:
            raise Error("arrow: decode_arrow_file: invalid trailing magic bytes")

    # ── Read footer_size ──────────────────────────────────────────────────────
    # Layout from the end: [footer bytes][footer_size: i32, 4 bytes][magic: 6 bytes]
    var footer_size_i32 = read_i32_le(buf, n - 10)
    if footer_size_i32 <= Int32(0):
        raise Error("arrow: decode_arrow_file: invalid footer_size")
    var footer_size = Int(footer_size_i32)
    var footer_start = n - 10 - footer_size
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

        # Same List<Utf8>-aware node/buffer accounting as decode_record_batch:
        # a List column contributes 2 FieldNodes, not 1, so node_idx and
        # buf_idx both advance independently per column rather than in
        # lockstep with the schema field index.
        var arrays   = List[ArrowArray]()
        var buf_idx  = 0
        var node_idx = 0
        for j in range(len(schema.fields)):
            var field_type = schema.fields[j].type.copy()

            if field_type.tag == TYPE_LIST():
                if node_idx + 2 > len(rb_nodes):
                    raise Error("arrow: decode_arrow_file: node underrun for List column " + String(j) + " in batch " + String(i))
                if buf_idx + 5 > len(rb_buffers):
                    raise Error("arrow: decode_arrow_file: buffer descriptor underrun for List column " + String(j) + " in batch " + String(i))
                var field_nodes = List[FieldNode]()
                field_nodes.append(rb_nodes[node_idx].copy())
                field_nodes.append(rb_nodes[node_idx + 1].copy())
                node_idx += 2
                var list_descs = List[BufferDesc]()
                for k in range(5):
                    list_descs.append(rb_buffers[buf_idx + k].copy())
                buf_idx += 5
                arrays.append(decode_list_utf8_array(field_nodes, list_descs, rb_body))
                continue

            if node_idx + 1 > len(rb_nodes):
                raise Error("arrow: decode_arrow_file: node underrun for column " + String(j) + " in batch " + String(i))
            var node = rb_nodes[node_idx].copy()
            node_idx += 1

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

        if node_idx != len(rb_nodes):
            raise Error("arrow: decode_arrow_file: unconsumed field nodes remain in batch " + String(i))

        batches.append(RecordBatch(rb_length, arrays^))

    return Tuple[ArrowSchema, List[RecordBatch]](schema^, batches^)
