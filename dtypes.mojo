# dtypes.mojo — Arrow data type system
# Phase 1 of the arrow refactor.
#
# Provides a trait-based type hierarchy (DataType, PrimitiveType) and
# the AnyDataType type-erased container (Variant over 15 concrete types).
# Nested types (ListType, FixedSizeListType, StructType) will be added in Phase 2.

from std.utils import Variant


# ── Traits ────────────────────────────────────────────────────────────────────


trait DataType:
    """Base marker trait for all Arrow data types."""
    pass


trait PrimitiveType(DataType):
    """
    Fixed-width Arrow type with a direct hardware SIMD mapping.
    The `native` alias gives the compile-time DType used by compute kernels
    for parametric SIMD dispatch (Phase 2+).
    """
    comptime native: DType


# ── Concrete type structs ─────────────────────────────────────────────────────
# 13 primitive types (PrimitiveType) + NullType + BinaryType + StringType (DataType only).
# Zero-sized structs: no fields, only a type-level identity.


struct NullType(DataType, ImplicitlyCopyable, Movable):
    """Arrow Null type — all values are null; no data buffers."""

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct BoolType(PrimitiveType, ImplicitlyCopyable, Movable):
    """
    Arrow Boolean type.
    NOTE: BoolArray is bit-packed (1 bit/element), NOT byte-per-element.
    PrimitiveArray[BoolType] is therefore prohibited; use BoolArray instead (Phase 2).
    """
    comptime native: DType = DType.bool

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct Int8Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.int8

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct Int16Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.int16

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct Int32Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.int32

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct Int64Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.int64

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct UInt8Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.uint8

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct UInt16Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.uint16

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct UInt32Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.uint32

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct UInt64Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.uint64

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct Float16Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.float16

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct Float32Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.float32

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct Float64Type(PrimitiveType, ImplicitlyCopyable, Movable):
    comptime native: DType = DType.float64

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct BinaryType(DataType, ImplicitlyCopyable, Movable):
    """Variable-length opaque binary data (Arrow Binary type)."""

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


struct StringType(DataType, ImplicitlyCopyable, Movable):
    """Variable-length UTF-8 string data (Arrow Utf8 type)."""

    fn __init__(out self):
        pass

    fn __copyinit__(out self, copy: Self):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


# ── AnyDataType — type-erased container ──────────────────────────────────────


struct AnyDataType(ImplicitlyCopyable, Movable):
    """
    Type-erased Arrow data type backed by a Variant over 15 concrete types.

    Use the @staticmethod factory methods for construction and the `is_*`
    predicate methods for type inspection.

    Nested types (ListType, FixedSizeListType, StructType) will be added
    once the Phase 2 array layer supports them.
    """

    comptime _V = Variant[
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
    ]

    var _data: Self._V

    fn __init__(out self, data: Self._V):
        self._data = data

    fn __copyinit__(out self, copy: Self):
        self._data = copy._data

    fn __moveinit__(out self, deinit take: Self):
        self._data = take._data^

    # ── Factory methods ────────────────────────────────────────────────────────

    @staticmethod
    def null_() -> AnyDataType:
        return AnyDataType(AnyDataType._V(NullType()))

    @staticmethod
    def bool_() -> AnyDataType:
        return AnyDataType(AnyDataType._V(BoolType()))

    @staticmethod
    def int8() -> AnyDataType:
        return AnyDataType(AnyDataType._V(Int8Type()))

    @staticmethod
    def int16() -> AnyDataType:
        return AnyDataType(AnyDataType._V(Int16Type()))

    @staticmethod
    def int32() -> AnyDataType:
        return AnyDataType(AnyDataType._V(Int32Type()))

    @staticmethod
    def int64() -> AnyDataType:
        return AnyDataType(AnyDataType._V(Int64Type()))

    @staticmethod
    def uint8() -> AnyDataType:
        return AnyDataType(AnyDataType._V(UInt8Type()))

    @staticmethod
    def uint16() -> AnyDataType:
        return AnyDataType(AnyDataType._V(UInt16Type()))

    @staticmethod
    def uint32() -> AnyDataType:
        return AnyDataType(AnyDataType._V(UInt32Type()))

    @staticmethod
    def uint64() -> AnyDataType:
        return AnyDataType(AnyDataType._V(UInt64Type()))

    @staticmethod
    def float16() -> AnyDataType:
        return AnyDataType(AnyDataType._V(Float16Type()))

    @staticmethod
    def float32() -> AnyDataType:
        return AnyDataType(AnyDataType._V(Float32Type()))

    @staticmethod
    def float64() -> AnyDataType:
        return AnyDataType(AnyDataType._V(Float64Type()))

    @staticmethod
    def binary() -> AnyDataType:
        return AnyDataType(AnyDataType._V(BinaryType()))

    @staticmethod
    def string() -> AnyDataType:
        return AnyDataType(AnyDataType._V(StringType()))

    # ── Type predicates ────────────────────────────────────────────────────────

    def is_null(self) -> Bool:
        return self._data.isa[NullType]()

    def is_bool(self) -> Bool:
        return self._data.isa[BoolType]()

    def is_signed_integer(self) -> Bool:
        return (
            self._data.isa[Int8Type]()
            or self._data.isa[Int16Type]()
            or self._data.isa[Int32Type]()
            or self._data.isa[Int64Type]()
        )

    def is_unsigned_integer(self) -> Bool:
        return (
            self._data.isa[UInt8Type]()
            or self._data.isa[UInt16Type]()
            or self._data.isa[UInt32Type]()
            or self._data.isa[UInt64Type]()
        )

    def is_integer(self) -> Bool:
        return self.is_signed_integer() or self.is_unsigned_integer()

    def is_floating(self) -> Bool:
        return (
            self._data.isa[Float16Type]()
            or self._data.isa[Float32Type]()
            or self._data.isa[Float64Type]()
        )

    def is_primitive(self) -> Bool:
        """True for bool, all integer, and all floating-point types."""
        return self.is_bool() or self.is_integer() or self.is_floating()

    def is_variable_width(self) -> Bool:
        """True for Binary and String (UTF-8) types."""
        return self._data.isa[BinaryType]() or self._data.isa[StringType]()

    def is_nested(self) -> Bool:
        """True for List, FixedSizeList, Struct. Always False in Phase 1."""
        return False


# ── Arrow C Data Interface format strings ─────────────────────────────────────
# Spec: https://arrow.apache.org/docs/format/CDataInterface.html
# Used by c_data.mojo (Phase 3).


def dtype_to_format_string(dt: AnyDataType) raises -> String:
    """Map an AnyDataType to its Arrow C Data Interface format string."""
    if dt.is_null():
        return "n"
    if dt.is_bool():
        return "b"
    if dt._data.isa[Int8Type]():
        return "c"
    if dt._data.isa[UInt8Type]():
        return "C"
    if dt._data.isa[Int16Type]():
        return "s"
    if dt._data.isa[UInt16Type]():
        return "S"
    if dt._data.isa[Int32Type]():
        return "i"
    if dt._data.isa[UInt32Type]():
        return "I"
    if dt._data.isa[Int64Type]():
        return "l"
    if dt._data.isa[UInt64Type]():
        return "L"
    if dt._data.isa[Float16Type]():
        return "e"
    if dt._data.isa[Float32Type]():
        return "f"
    if dt._data.isa[Float64Type]():
        return "g"
    if dt._data.isa[StringType]():
        return "u"
    if dt._data.isa[BinaryType]():
        return "z"
    raise Error("dtypes: dtype_to_format_string: unhandled type")


def dtype_from_format_string(s: String) raises -> AnyDataType:
    """Reconstruct an AnyDataType from an Arrow C Data format string."""
    if s == "n":
        return AnyDataType.null_()
    if s == "b":
        return AnyDataType.bool_()
    if s == "c":
        return AnyDataType.int8()
    if s == "C":
        return AnyDataType.uint8()
    if s == "s":
        return AnyDataType.int16()
    if s == "S":
        return AnyDataType.uint16()
    if s == "i":
        return AnyDataType.int32()
    if s == "I":
        return AnyDataType.uint32()
    if s == "l":
        return AnyDataType.int64()
    if s == "L":
        return AnyDataType.uint64()
    if s == "e":
        return AnyDataType.float16()
    if s == "f":
        return AnyDataType.float32()
    if s == "g":
        return AnyDataType.float64()
    if s == "u":
        return AnyDataType.string()
    if s == "z":
        return AnyDataType.binary()
    raise Error("dtypes: dtype_from_format_string: unknown format '" + s + "'")
