# arrays.mojo — Arrow array types (Phase 2).
#
# PrimitiveArray[T: PrimitiveType] — fixed-width arrays backed by a packed
#   validity bitmap and a flat byte buffer (_dtype_bytes(T.native) bytes/element).
# StringArray, BinaryArray — variable-width arrays (validity + offsets + bytes).
# AnyArray — type-erased wrapper over all concrete array types.
#
# NOTE: AnyArray stores the erased data as flat buffers plus an AnyDataType tag.
# Phase 4 will replace List[UInt8] storage with Buffer[False] for GPU support.

from dtypes import (
    AnyDataType,
    PrimitiveType,
    BoolType,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    Float16Type, Float32Type, Float64Type,
    BinaryType, StringType,
)


# ── Byte-size helper ──────────────────────────────────────────────────────────


def _dtype_bytes(dtype: DType) -> Int:
    """Return the byte size of a scalar of the given DType."""
    if dtype == DType.bool or dtype == DType.int8 or dtype == DType.uint8:
        return 1
    if dtype == DType.int16 or dtype == DType.uint16 or dtype == DType.float16:
        return 2
    if dtype == DType.int32 or dtype == DType.uint32 or dtype == DType.float32:
        return 4
    if dtype == DType.int64 or dtype == DType.uint64 or dtype == DType.float64:
        return 8
    return 0


# ── Shared bitmap helper ──────────────────────────────────────────────────────


def _bit_is_set(bitmap: List[UInt8], i: Int) -> Bool:
    """Return True if bit i is set in a packed Arrow validity bitmap."""
    var byte_idx = i // 8
    var bit_idx = UInt8(i % 8)
    return (bitmap[byte_idx] >> bit_idx) & UInt8(1) == UInt8(1)


# ── PrimitiveArray[T] ─────────────────────────────────────────────────────────


struct PrimitiveArray[T: PrimitiveType](Copyable, Movable):
    """
    Immutable fixed-width Arrow array.

    _validity — packed bitmap (empty means all-valid, i.e. null_count == 0).
    _values   — _dtype_bytes(T.native) bytes per element, little-endian.
    """

    var _length: Int
    var _null_count: Int
    var _validity: List[UInt8]
    var _values: List[UInt8]

    fn __init__(
        out self,
        length: Int,
        null_count: Int,
        validity: List[UInt8],
        values: List[UInt8],
    ):
        self._length = length
        self._null_count = null_count
        self._validity = validity.copy()
        self._values = values.copy()

    fn __copyinit__(out self, copy: Self):
        self._length = copy._length
        self._null_count = copy._null_count
        self._validity = copy._validity.copy()
        self._values = copy._values.copy()

    fn __moveinit__(out self, deinit take: Self):
        self._length = take._length
        self._null_count = take._null_count
        self._validity = take._validity^
        self._values = take._values^

    # ── Accessors ─────────────────────────────────────────────────────────────

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) raises -> AnyDataType:
        """Return the AnyDataType corresponding to T."""
        if Self.T.native == DType.bool:
            return AnyDataType.bool_()
        if Self.T.native == DType.int8:
            return AnyDataType.int8()
        if Self.T.native == DType.int16:
            return AnyDataType.int16()
        if Self.T.native == DType.int32:
            return AnyDataType.int32()
        if Self.T.native == DType.int64:
            return AnyDataType.int64()
        if Self.T.native == DType.uint8:
            return AnyDataType.uint8()
        if Self.T.native == DType.uint16:
            return AnyDataType.uint16()
        if Self.T.native == DType.uint32:
            return AnyDataType.uint32()
        if Self.T.native == DType.uint64:
            return AnyDataType.uint64()
        if Self.T.native == DType.float16:
            return AnyDataType.float16()
        if Self.T.native == DType.float32:
            return AnyDataType.float32()
        if Self.T.native == DType.float64:
            return AnyDataType.float64()
        raise Error("arrays: PrimitiveArray.dtype: unhandled native type")

    def is_valid(self, i: Int) -> Bool:
        if self._null_count == 0 or len(self._validity) == 0:
            return True
        return _bit_is_set(self._validity, i)

    def get(self, i: Int) -> Scalar[Self.T.native]:
        """Return the value at index i (unsafe if element is null)."""
        var sz = _dtype_bytes(Self.T.native)
        var offset = i * sz
        return (self._values.unsafe_ptr() + offset).bitcast[Scalar[Self.T.native]]()[]


# ── StringArray ───────────────────────────────────────────────────────────────


struct StringArray(Copyable, Movable):
    """
    Immutable variable-width UTF-8 Arrow array.

    _offsets — length+1 Int32 values; element i spans bytes [_offsets[i], _offsets[i+1]).
    _values  — concatenated UTF-8 bytes.
    """

    var _length: Int
    var _null_count: Int
    var _validity: List[UInt8]
    var _offsets: List[Int32]
    var _values: List[UInt8]

    fn __init__(
        out self,
        length: Int,
        null_count: Int,
        validity: List[UInt8],
        offsets: List[Int32],
        values: List[UInt8],
    ):
        self._length = length
        self._null_count = null_count
        self._validity = validity.copy()
        self._offsets = offsets.copy()
        self._values = values.copy()

    fn __copyinit__(out self, copy: Self):
        self._length = copy._length
        self._null_count = copy._null_count
        self._validity = copy._validity.copy()
        self._offsets = copy._offsets.copy()
        self._values = copy._values.copy()

    fn __moveinit__(out self, deinit take: Self):
        self._length = take._length
        self._null_count = take._null_count
        self._validity = take._validity^
        self._offsets = take._offsets^
        self._values = take._values^

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def is_valid(self, i: Int) -> Bool:
        if self._null_count == 0 or len(self._validity) == 0:
            return True
        return _bit_is_set(self._validity, i)

    def get(self, i: Int) raises -> String:
        """Return the string at index i (undefined behaviour if null)."""
        var start = Int(self._offsets[i])
        var end = Int(self._offsets[i + 1])
        var buf = List[UInt8]()
        for j in range(start, end):
            buf.append(self._values[j])
        return String(unsafe_from_utf8=buf^)


# ── BinaryArray ───────────────────────────────────────────────────────────────


struct BinaryArray(Copyable, Movable):
    """Immutable variable-width opaque binary Arrow array (same layout as StringArray)."""

    var _length: Int
    var _null_count: Int
    var _validity: List[UInt8]
    var _offsets: List[Int32]
    var _values: List[UInt8]

    fn __init__(
        out self,
        length: Int,
        null_count: Int,
        validity: List[UInt8],
        offsets: List[Int32],
        values: List[UInt8],
    ):
        self._length = length
        self._null_count = null_count
        self._validity = validity.copy()
        self._offsets = offsets.copy()
        self._values = values.copy()

    fn __copyinit__(out self, copy: Self):
        self._length = copy._length
        self._null_count = copy._null_count
        self._validity = copy._validity.copy()
        self._offsets = copy._offsets.copy()
        self._values = copy._values.copy()

    fn __moveinit__(out self, deinit take: Self):
        self._length = take._length
        self._null_count = take._null_count
        self._validity = take._validity^
        self._offsets = take._offsets^
        self._values = take._values^

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def is_valid(self, i: Int) -> Bool:
        if self._null_count == 0 or len(self._validity) == 0:
            return True
        return _bit_is_set(self._validity, i)


# ── AnyArray ──────────────────────────────────────────────────────────────────


struct AnyArray(Copyable, Movable):
    """
    Type-erased Arrow array.

    Stores the erased array as flat byte buffers alongside an AnyDataType tag.
    _offsets is non-empty only for string/binary arrays.

    Phase 4 will replace List[UInt8] storage with Buffer[False].
    """

    var _dtype: AnyDataType
    var _length: Int
    var _null_count: Int
    var _validity: List[UInt8]
    var _values: List[UInt8]
    var _offsets: List[Int32]

    fn __init__(
        out self,
        dtype: AnyDataType,
        length: Int,
        null_count: Int,
        validity: List[UInt8],
        values: List[UInt8],
        offsets: List[Int32],
    ):
        self._dtype = dtype
        self._length = length
        self._null_count = null_count
        self._validity = validity.copy()
        self._values = values.copy()
        self._offsets = offsets.copy()

    fn __copyinit__(out self, copy: Self):
        self._dtype = copy._dtype
        self._length = copy._length
        self._null_count = copy._null_count
        self._validity = copy._validity.copy()
        self._values = copy._values.copy()
        self._offsets = copy._offsets.copy()

    fn __moveinit__(out self, deinit take: Self):
        self._dtype = take._dtype^
        self._length = take._length
        self._null_count = take._null_count
        self._validity = take._validity^
        self._values = take._values^
        self._offsets = take._offsets^

    # ── Accessors ─────────────────────────────────────────────────────────────

    def length(self) -> Int:
        return self._length

    def null_count(self) -> Int:
        return self._null_count

    def dtype(self) -> AnyDataType:
        return self._dtype

    def is_valid(self, i: Int) -> Bool:
        if self._null_count == 0 or len(self._validity) == 0:
            return True
        return _bit_is_set(self._validity, i)

    # ── Factory methods ───────────────────────────────────────────────────────

    @staticmethod
    def from_primitive[T: PrimitiveType](arr: PrimitiveArray[T]) raises -> AnyArray:
        var empty_offsets = List[Int32]()
        return AnyArray(
            arr.dtype(),
            arr._length,
            arr._null_count,
            arr._validity,
            arr._values,
            empty_offsets,
        )

    @staticmethod
    def from_string(arr: StringArray) -> AnyArray:
        return AnyArray(
            AnyDataType.string(),
            arr._length,
            arr._null_count,
            arr._validity,
            arr._values,
            arr._offsets,
        )

    @staticmethod
    def from_binary(arr: BinaryArray) -> AnyArray:
        return AnyArray(
            AnyDataType.binary(),
            arr._length,
            arr._null_count,
            arr._validity,
            arr._values,
            arr._offsets,
        )

    # ── Downcast ──────────────────────────────────────────────────────────────

    def downcast_primitive[T: PrimitiveType](self) -> PrimitiveArray[T]:
        """Reinterpret as PrimitiveArray[T] (caller must verify dtype matches T)."""
        return PrimitiveArray[T](
            self._length,
            self._null_count,
            self._validity,
            self._values,
        )

    def downcast_string(self) raises -> StringArray:
        if not self._dtype.is_variable_width():
            raise Error("AnyArray.downcast_string: dtype is not string/binary")
        return StringArray(
            self._length,
            self._null_count,
            self._validity,
            self._offsets,
            self._values,
        )

    def downcast_binary(self) raises -> BinaryArray:
        if not self._dtype.is_variable_width():
            raise Error("AnyArray.downcast_binary: dtype is not string/binary")
        return BinaryArray(
            self._length,
            self._null_count,
            self._validity,
            self._offsets,
            self._values,
        )
