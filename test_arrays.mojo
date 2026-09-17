from arrays import PrimitiveArray, StringArray, BinaryArray, AnyArray
from builders import PrimitiveBuilder, StringBuilder
from dtypes import (
    AnyDataType,
    BoolType,
    Int8Type, Int32Type, Int64Type,
    Float32Type, Float64Type,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def assert_true(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("FAIL: " + msg)


def assert_eq(a: String, b: String, msg: String) raises:
    if a != b:
        raise Error("FAIL: " + msg + " — got '" + a + "', expected '" + b + "'")


def assert_eq_int(a: Int, b: Int, msg: String) raises:
    if a != b:
        raise Error(
            "FAIL: " + msg + " — got " + String(a) + ", expected " + String(b)
        )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_primitive_array_get() raises:
    """PrimitiveBuilder round-trips Int32 values through PrimitiveArray.get()."""
    var b = PrimitiveBuilder[Int32Type]()
    b.append(Int32(10))
    b.append(Int32(20))
    b.append(Int32(30))
    var arr = b.finish()
    assert_eq_int(arr.length(), 3, "length")
    assert_eq_int(arr.null_count(), 0, "null_count")
    assert_eq_int(Int(arr.get(0)), 10, "get(0)")
    assert_eq_int(Int(arr.get(1)), 20, "get(1)")
    assert_eq_int(Int(arr.get(2)), 30, "get(2)")


def test_primitive_array_null_check() raises:
    """Validity bitmap correctly marks null and non-null elements."""
    var b = PrimitiveBuilder[Int32Type]()
    b.append(Int32(1))
    b.append_null()
    b.append(Int32(3))
    var arr = b.finish()
    assert_eq_int(arr.length(), 3, "length")
    assert_eq_int(arr.null_count(), 1, "null_count")
    assert_true(arr.is_valid(0), "element 0 should be valid")
    assert_true(not arr.is_valid(1), "element 1 should be null")
    assert_true(arr.is_valid(2), "element 2 should be valid")
    assert_eq_int(Int(arr.get(0)), 1, "get(0)")
    assert_eq_int(Int(arr.get(2)), 3, "get(2)")


def test_primitive_builder_float64() raises:
    """PrimitiveBuilder round-trips Float64 values."""
    var b = PrimitiveBuilder[Float64Type]()
    b.append(Float64(3.14))
    b.append(Float64(-1.0))
    var arr = b.finish()
    assert_eq_int(arr.length(), 2, "length")
    assert_eq_int(arr.null_count(), 0, "no nulls")
    assert_true(arr.is_valid(0), "valid 0")
    assert_true(arr.is_valid(1), "valid 1")
    # Check approximate value — cast to float to compare
    var v0 = Float64(arr.get(0))
    assert_true(v0 > Float64(3.13) and v0 < Float64(3.15), "get(0) ≈ 3.14")


def test_any_array_from_primitive() raises:
    """AnyArray.from_primitive wraps a PrimitiveArray with correct metadata."""
    var b = PrimitiveBuilder[Int32Type]()
    b.append(Int32(42))
    b.append_null()
    var arr = b.finish()
    var any = AnyArray.from_primitive[Int32Type](arr)
    assert_eq_int(any.length(), 2, "any length")
    assert_eq_int(any.null_count(), 1, "any null_count")
    assert_true(any.dtype().is_integer(), "dtype is integer")
    assert_true(any.dtype().is_signed_integer(), "dtype is signed integer")
    assert_true(any.is_valid(0), "any[0] valid")
    assert_true(not any.is_valid(1), "any[1] null")


def test_any_array_dtype_dispatch() raises:
    """AnyArray.from_primitive preserves correct dtype for each primitive type."""
    var b1 = PrimitiveBuilder[Int32Type]()
    b1.append(Int32(1))
    var any1 = AnyArray.from_primitive[Int32Type](b1.finish())
    assert_true(any1.dtype().is_signed_integer(), "int32 is signed integer")

    var b2 = PrimitiveBuilder[Float64Type]()
    b2.append(Float64(1.0))
    var any2 = AnyArray.from_primitive[Float64Type](b2.finish())
    assert_true(any2.dtype().is_floating(), "float64 is floating")

    var b3 = PrimitiveBuilder[Int64Type]()
    b3.append(Int64(0))
    var any3 = AnyArray.from_primitive[Int64Type](b3.finish())
    assert_true(any3.dtype().is_integer(), "int64 is integer")


def test_any_array_downcast() raises:
    """downcast_primitive reconstructs a usable PrimitiveArray from AnyArray."""
    var b = PrimitiveBuilder[Int32Type]()
    b.append(Int32(99))
    b.append(Int32(100))
    var arr = b.finish()
    var any = AnyArray.from_primitive[Int32Type](arr)
    var back = any.downcast_primitive[Int32Type]()
    assert_eq_int(back.length(), 2, "downcast length")
    assert_eq_int(Int(back.get(0)), 99, "downcast get(0)")
    assert_eq_int(Int(back.get(1)), 100, "downcast get(1)")


def test_string_builder_roundtrip() raises:
    """StringBuilder produces a StringArray with correct values, nulls, and lengths."""
    var b = StringBuilder()
    b.append("hello")
    b.append("world")
    b.append_null()
    b.append("!")
    var arr = b.finish()
    assert_eq_int(arr.length(), 4, "length")
    assert_eq_int(arr.null_count(), 1, "null_count")
    assert_true(arr.is_valid(0), "str[0] valid")
    assert_true(arr.is_valid(1), "str[1] valid")
    assert_true(not arr.is_valid(2), "str[2] null")
    assert_true(arr.is_valid(3), "str[3] valid")
    assert_eq(arr.get(0), "hello", "str[0]")
    assert_eq(arr.get(1), "world", "str[1]")
    assert_eq(arr.get(3), "!", "str[3]")


# ── Runner ────────────────────────────────────────────────────────────────────


def main() raises:
    test_primitive_array_get()
    print("PASS test_primitive_array_get")

    test_primitive_array_null_check()
    print("PASS test_primitive_array_null_check")

    test_primitive_builder_float64()
    print("PASS test_primitive_builder_float64")

    test_any_array_from_primitive()
    print("PASS test_any_array_from_primitive")

    test_any_array_dtype_dispatch()
    print("PASS test_any_array_dtype_dispatch")

    test_any_array_downcast()
    print("PASS test_any_array_downcast")

    test_string_builder_roundtrip()
    print("PASS test_string_builder_roundtrip")

    print("\nAll array tests passed.")
