from dtypes import (
    AnyDataType,
    NullType, BoolType,
    Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    Float16Type, Float32Type, Float64Type,
    BinaryType, StringType,
    dtype_to_format_string, dtype_from_format_string,
)
from schema import Field, Schema


# ── Helpers ───────────────────────────────────────────────────────────────────


def assert_true(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("FAIL: " + msg)


def assert_eq(a: String, b: String, msg: String) raises:
    if a != b:
        raise Error("FAIL: " + msg + " — got '" + a + "', expected '" + b + "'")


def assert_eq_int(a: Int, b: Int, msg: String) raises:
    if a != b:
        raise Error("FAIL: " + msg + " — got " + String(a) + ", expected " + String(b))


# ── Type predicate tests ──────────────────────────────────────────────────────


def test_null_is_null() raises:
    var dt = AnyDataType.null_()
    assert_true(dt.is_null(), "null_() should satisfy is_null()")
    assert_true(not dt.is_bool(), "null_() should not satisfy is_bool()")
    assert_true(not dt.is_integer(), "null_() should not satisfy is_integer()")
    assert_true(not dt.is_primitive(), "null_() should not satisfy is_primitive()")


def test_bool_is_bool() raises:
    var dt = AnyDataType.bool_()
    assert_true(dt.is_bool(), "bool_() should satisfy is_bool()")
    assert_true(dt.is_primitive(), "bool_() should satisfy is_primitive()")
    assert_true(not dt.is_null(), "bool_() should not satisfy is_null()")
    assert_true(not dt.is_integer(), "bool_() should not satisfy is_integer()")


def test_int32_is_integer() raises:
    var dt = AnyDataType.int32()
    assert_true(dt.is_integer(), "int32() should satisfy is_integer()")
    assert_true(dt.is_signed_integer(), "int32() should satisfy is_signed_integer()")
    assert_true(not dt.is_unsigned_integer(), "int32() should not satisfy is_unsigned_integer()")
    assert_true(dt.is_primitive(), "int32() should satisfy is_primitive()")
    assert_true(not dt.is_null(), "int32() should not satisfy is_null()")


def test_uint64_is_unsigned_integer() raises:
    var dt = AnyDataType.uint64()
    assert_true(dt.is_integer(), "uint64() should satisfy is_integer()")
    assert_true(dt.is_unsigned_integer(), "uint64() should satisfy is_unsigned_integer()")
    assert_true(not dt.is_signed_integer(), "uint64() should not satisfy is_signed_integer()")


def test_float64_is_floating() raises:
    var dt = AnyDataType.float64()
    assert_true(dt.is_floating(), "float64() should satisfy is_floating()")
    assert_true(dt.is_primitive(), "float64() should satisfy is_primitive()")
    assert_true(not dt.is_integer(), "float64() should not satisfy is_integer()")
    assert_true(not dt.is_variable_width(), "float64() should not satisfy is_variable_width()")


def test_string_is_variable_width() raises:
    var dt = AnyDataType.string()
    assert_true(dt.is_variable_width(), "string() should satisfy is_variable_width()")
    assert_true(not dt.is_primitive(), "string() should not satisfy is_primitive()")
    assert_true(not dt.is_nested(), "string() should not satisfy is_nested()")


def test_binary_is_variable_width() raises:
    var dt = AnyDataType.binary()
    assert_true(dt.is_variable_width(), "binary() should satisfy is_variable_width()")
    assert_true(not dt.is_primitive(), "binary() should not satisfy is_primitive()")


def test_no_nested_types_in_phase1() raises:
    # All 15 current types should return False for is_nested()
    var types = List[AnyDataType]()
    types.append(AnyDataType.null_())
    types.append(AnyDataType.bool_())
    types.append(AnyDataType.int8())
    types.append(AnyDataType.int16())
    types.append(AnyDataType.int32())
    types.append(AnyDataType.int64())
    types.append(AnyDataType.uint8())
    types.append(AnyDataType.uint16())
    types.append(AnyDataType.uint32())
    types.append(AnyDataType.uint64())
    types.append(AnyDataType.float16())
    types.append(AnyDataType.float32())
    types.append(AnyDataType.float64())
    types.append(AnyDataType.binary())
    types.append(AnyDataType.string())
    for i in range(len(types)):
        assert_true(not types[i].is_nested(), "type[" + String(i) + "] should not be nested")


# ── Schema / Field tests ──────────────────────────────────────────────────────


def test_field_name_and_dtype() raises:
    var f = Field("score", AnyDataType.float64(), True)
    assert_eq(f.name, "score", "field name")
    assert_true(f.dtype.is_floating(), "field dtype should be floating")
    assert_true(f.nullable, "field should be nullable")


def test_schema_field_count() raises:
    var fields = List[Field]()
    fields.append(Field("id", AnyDataType.int32(), False))
    fields.append(Field("name", AnyDataType.string(), True))
    fields.append(Field("value", AnyDataType.float64(), True))
    var s = Schema(fields)
    assert_eq_int(s.n_fields(), 3, "schema field count")
    assert_eq(s.field(0).name, "id", "first field name")
    assert_eq(s.field(1).name, "name", "second field name")
    assert_eq(s.field(2).name, "value", "third field name")


def test_schema_endianness_and_nullable() raises:
    # Verify endianness and nullable fields are preserved correctly.
    var fields = List[Field]()
    fields.append(Field("id", AnyDataType.int32(), False))
    fields.append(Field("tag", AnyDataType.string(), True))
    var s = Schema(fields, Int16(1))  # big-endian
    assert_eq_int(Int(s.endianness), 1, "schema endianness")
    assert_true(not s.field(0).nullable, "id should not be nullable")
    assert_true(s.field(1).nullable, "tag should be nullable")


# ── Format string tests ────────────────────────────────────────────────────────


def test_format_strings_all_primitives() raises:
    var cases = List[Tuple[AnyDataType, String]]()
    cases.append(Tuple[AnyDataType, String](AnyDataType.null_(), "n"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.bool_(), "b"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.int8(), "c"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.uint8(), "C"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.int16(), "s"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.uint16(), "S"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.int32(), "i"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.uint32(), "I"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.int64(), "l"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.uint64(), "L"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.float16(), "e"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.float32(), "f"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.float64(), "g"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.string(), "u"))
    cases.append(Tuple[AnyDataType, String](AnyDataType.binary(), "z"))
    for i in range(len(cases)):
        var fmt = dtype_to_format_string(cases[i][0])
        assert_eq(fmt, cases[i][1], "format string for index " + String(i))


def test_format_string_roundtrip() raises:
    var formats = List[String]()
    formats.append("n")
    formats.append("b")
    formats.append("c")
    formats.append("C")
    formats.append("s")
    formats.append("S")
    formats.append("i")
    formats.append("I")
    formats.append("l")
    formats.append("L")
    formats.append("e")
    formats.append("f")
    formats.append("g")
    formats.append("u")
    formats.append("z")
    for i in range(len(formats)):
        var dt = dtype_from_format_string(formats[i])
        var back = dtype_to_format_string(dt)
        assert_eq(back, formats[i], "roundtrip format[" + String(i) + "]")


def test_format_string_unknown_raises() raises:
    var raised = False
    try:
        _ = dtype_from_format_string("???")
    except:
        raised = True
    assert_true(raised, "dtype_from_format_string should raise on unknown format")


# ── Runner ────────────────────────────────────────────────────────────────────


def main() raises:
    test_null_is_null()
    print("PASS test_null_is_null")

    test_bool_is_bool()
    print("PASS test_bool_is_bool")

    test_int32_is_integer()
    print("PASS test_int32_is_integer")

    test_uint64_is_unsigned_integer()
    print("PASS test_uint64_is_unsigned_integer")

    test_float64_is_floating()
    print("PASS test_float64_is_floating")

    test_string_is_variable_width()
    print("PASS test_string_is_variable_width")

    test_binary_is_variable_width()
    print("PASS test_binary_is_variable_width")

    test_no_nested_types_in_phase1()
    print("PASS test_no_nested_types_in_phase1")

    test_field_name_and_dtype()
    print("PASS test_field_name_and_dtype")

    test_schema_field_count()
    print("PASS test_schema_field_count")

    test_schema_endianness_and_nullable()
    print("PASS test_schema_endianness_and_nullable")

    test_format_strings_all_primitives()
    print("PASS test_format_strings_all_primitives")

    test_format_string_roundtrip()
    print("PASS test_format_string_roundtrip")

    test_format_string_unknown_raises()
    print("PASS test_format_string_unknown_raises")

    print("\nAll dtypes tests passed.")
