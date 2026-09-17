# builders.mojo — Mutable Arrow array builders (Phase 2).
#
# PrimitiveBuilder[T: PrimitiveType] — append/append_null, finish → PrimitiveArray[Self.T].
# StringBuilder — append/append_null, finish → StringArray.
#
# Validity bitmaps are accumulated as List[Bool] during building and packed
# into Arrow-format packed bitmaps (1 bit/element, LSB-first) at finish().

from dtypes import (AnyDataType, PrimitiveType)
from arrays import (PrimitiveArray, StringArray, BinaryArray, AnyArray, _dtype_bytes)


# ── Bitmap packing helper ─────────────────────────────────────────────────────


def _pack_validity(null_bits: List[Bool], length: Int) raises -> List[UInt8]:
    """Pack a List[Bool] (True=valid, False=null) into an Arrow validity bitmap."""
    var n_bytes = (length + 7) // 8
    var bitmap = List[UInt8]()
    for _ in range(n_bytes):
        bitmap.append(UInt8(0))
    for i in range(length):
        if null_bits[i]:
            var byte_idx = i // 8
            var bit_idx = UInt8(i % 8)
            bitmap[byte_idx] = bitmap[byte_idx] | (UInt8(1) << bit_idx)
    return bitmap^


# ── PrimitiveBuilder[T] ───────────────────────────────────────────────────────


struct PrimitiveBuilder[T: PrimitiveType](Movable):
    """
    Mutable builder for PrimitiveArray[Self.T].

    append(val)   — append a non-null value.
    append_null() — append a null slot (zero bytes in value buffer).
    finish()      — return an immutable PrimitiveArray[Self.T].
    """

    var _values: List[UInt8]
    var _null_bits: List[Bool]
    var _length: Int
    var _null_count: Int

    def __init__(out self):
        self._values = List[UInt8]()
        self._null_bits = List[Bool]()
        self._length = 0
        self._null_count = 0

    def __moveinit__(out self, deinit take: Self):
        self._values = take._values^
        self._null_bits = take._null_bits^
        self._length = take._length
        self._null_count = take._null_count

    def append(mut self, val: Scalar[Self.T.native]) raises:
        """Append a non-null value."""
        var sz = _dtype_bytes(Self.T.native)
        var old_len = len(self._values)
        # Reserve sz bytes then write val directly into the buffer.
        for _ in range(sz):
            self._values.append(UInt8(0))
        self._values.unsafe_ptr().unsafe_offset(old_len).unsafe_bitcast[Scalar[Self.T.native]]()[] = val
        self._null_bits.append(True)
        self._length += 1

    def append_null(mut self) raises:
        """Append a null slot (value bytes are zeroed)."""
        var sz = _dtype_bytes(Self.T.native)
        for _ in range(sz):
            self._values.append(UInt8(0))
        self._null_bits.append(False)
        self._length += 1
        self._null_count += 1

    def finish(mut self) raises -> PrimitiveArray[Self.T]:
        """Return an immutable PrimitiveArray[Self.T]."""
        var validity = _pack_validity(self._null_bits, self._length)
        var values = self._values^
        self._values = List[UInt8]()
        return PrimitiveArray[Self.T](self._length, self._null_count, validity, values)


# ── StringBuilder ─────────────────────────────────────────────────────────────


struct StringBuilder(Movable):
    """
    Mutable builder for StringArray.

    append(val)   — append a non-null UTF-8 string.
    append_null() — append a null slot.
    finish()      — return an immutable StringArray.
    """

    var _values: List[UInt8]
    var _offsets: List[Int32]
    var _null_bits: List[Bool]
    var _length: Int
    var _null_count: Int

    def __init__(out self):
        self._values = List[UInt8]()
        self._offsets = List[Int32]()
        self._null_bits = List[Bool]()
        self._offsets.append(Int32(0))  # first offset is always 0
        self._length = 0
        self._null_count = 0

    def __moveinit__(out self, deinit take: Self):
        self._values = take._values^
        self._offsets = take._offsets^
        self._null_bits = take._null_bits^
        self._length = take._length
        self._null_count = take._null_count

    def append(mut self, val: String) raises:
        """Append a non-null UTF-8 string."""
        var bytes = val.as_bytes()
        for i in range(len(bytes)):
            self._values.append(bytes[i])
        self._offsets.append(Int32(len(self._values)))
        self._null_bits.append(True)
        self._length += 1

    def append_null(mut self):
        """Append a null slot (empty span — values buffer not advanced)."""
        self._offsets.append(Int32(len(self._values)))
        self._null_bits.append(False)
        self._length += 1
        self._null_count += 1

    def finish(mut self) raises -> StringArray:
        """Return an immutable StringArray."""
        var validity = _pack_validity(self._null_bits, self._length)
        var offsets = self._offsets^
        self._offsets = List[Int32]()
        var values = self._values^
        self._values = List[UInt8]()
        return StringArray(self._length, self._null_count, validity, offsets, values)
