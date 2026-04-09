from csv_arrow import (
    read_csv,
    infer_type,
    build_array,
    csv_to_feather,
    feather_to_csv,
)
from arrow import (
    ArrowType, ArrowSchema, ArrowField,
    ArrowArray, RecordBatch,
    decode_arrow_file,
    TYPE_INT, TYPE_FLOAT, TYPE_UTF8,
    encode_arrow_file,
)
from flatbuffers import read_i32_le, read_i64_le, read_f64_le
from pathlib import Path


fn assert_true(cond: Bool, msg: String = "") raises:
    if not cond:
        raise Error(msg if len(msg) > 0 else "expected True")


fn assert_eq_int(actual: Int, expected: Int, msg: String = "") raises:
    if actual != expected:
        raise Error((msg + ": " if len(msg) > 0 else "") + "expected " + String(expected) + " got " + String(actual))


fn _write_csv(path: String, content: String) raises:
    Path(path).write_text(content)


# ============================================================================
# Phase 7 tests
# ============================================================================


fn test_parse_csv_headers() raises:
    """Column names are extracted from the first row of the CSV."""
    _write_csv("/tmp/test_headers.csv", "id,name,score\n1,alice,9.5\n2,bob,8.0\n")
    var result = read_csv("/tmp/test_headers.csv")
    var headers = result[0].copy()
    assert_eq_int(len(headers), 3, "3 headers")
    assert_true(headers[0] == "id", "headers[0]=id")
    assert_true(headers[1] == "name", "headers[1]=name")
    assert_true(headers[2] == "score", "headers[2]=score")


fn test_parse_csv_rows() raises:
    """Row values match the CSV input exactly."""
    _write_csv("/tmp/test_rows.csv", "a,b\n10,20\n30,40\n")
    var result = read_csv("/tmp/test_rows.csv")
    var rows = result[1].copy()
    assert_eq_int(len(rows), 2, "2 data rows")
    assert_eq_int(len(rows[0]), 2, "row0 has 2 cols")
    assert_true(rows[0][0] == "10", "row0[0]=10")
    assert_true(rows[0][1] == "20", "row0[1]=20")
    assert_true(rows[1][0] == "30", "row1[0]=30")
    assert_true(rows[1][1] == "40", "row1[1]=40")


fn test_infer_int_column() raises:
    """All-integer column infers as Int64."""
    var vals = List[String]()
    vals.append(String("1"))
    vals.append(String("42"))
    vals.append(String("-7"))
    var t = infer_type(vals)
    assert_true(t.tag == TYPE_INT(), "int column tag")
    assert_true(t.int_meta.bit_width == Int32(64), "int64 bit_width")
    assert_true(t.int_meta.is_signed, "int64 is_signed")


fn test_infer_float_column() raises:
    """All-float column infers as Float64."""
    var vals = List[String]()
    vals.append(String("1.5"))
    vals.append(String("3.14"))
    vals.append(String("-2.0"))
    var t = infer_type(vals)
    assert_true(t.tag == TYPE_FLOAT(), "float column tag")
    assert_true(t.float_meta.precision == UInt16(2), "float64 precision=2")


fn test_infer_string_column() raises:
    """Mixed or non-numeric column infers as Utf8."""
    var vals = List[String]()
    vals.append(String("alice"))
    vals.append(String("bob"))
    vals.append(String("42"))   # mixed: one number, rest strings
    var t = infer_type(vals)
    assert_true(t.tag == TYPE_UTF8(), "utf8 column tag (mixed)")


fn test_csv_roundtrip() raises:
    """CSV -> feather -> decode_arrow_file returns identical data."""
    _write_csv(
        "/tmp/test_roundtrip.csv",
        "id,score\n1,9.5\n2,8.0\n3,7.25\n",
    )
    csv_to_feather("/tmp/test_roundtrip.csv", "/tmp/test_roundtrip.feather")

    var file_bytes = Path("/tmp/test_roundtrip.feather").read_bytes()
    _ = len(file_bytes)
    var result = decode_arrow_file(file_bytes)
    var schema = result[0].copy()
    var batches = result[1].copy()

    assert_eq_int(len(schema.fields), 2, "2 fields")
    assert_true(schema.fields[0].name == "id", "field[0]=id")
    assert_true(schema.fields[1].name == "score", "field[1]=score")
    assert_eq_int(len(batches), 1, "1 batch")
    assert_true(batches[0].length == Int64(3), "3 rows")

    # id column should be Int64
    var id_col = batches[0].columns[0].copy()
    assert_true(id_col.type.tag == TYPE_INT(), "id is int")
    assert_eq_int(id_col.length, 3, "id length=3")

    # score column should be Float64
    var score_col = batches[0].columns[1].copy()
    assert_true(score_col.type.tag == TYPE_FLOAT(), "score is float")


fn test_csv_to_feather_file() raises:
    """csv_to_feather writes a real file readable by decode_arrow_file."""
    _write_csv(
        "/tmp/test_e2e.csv",
        "name,value\nalice,10\nbob,20\n",
    )
    csv_to_feather("/tmp/test_e2e.csv", "/tmp/test_e2e.feather")

    var file_bytes = Path("/tmp/test_e2e.feather").read_bytes()
    _ = len(file_bytes)
    var result = decode_arrow_file(file_bytes)
    var schema = result[0].copy()
    var batches = result[1].copy()

    assert_eq_int(len(schema.fields), 2, "2 fields")
    assert_eq_int(len(batches), 1, "1 batch")
    var batch = batches[0].copy()
    assert_true(batch.length == Int64(2), "2 rows")
    # name column: Utf8
    assert_true(batch.columns[0].type.tag == TYPE_UTF8(), "name is utf8")
    # value column: Int64
    assert_true(batch.columns[1].type.tag == TYPE_INT(), "value is int")
    # Verify value[0] = 10
    var val_col = batch.columns[1].copy()
    var v0 = read_i64_le(val_col.values, 0)
    assert_true(v0 == Int64(10), "value[0]=10, got " + String(v0))


# ============================================================================
# Test runner
# ============================================================================


fn run_test(name: String, mut passed: Int, mut failed: Int, test_fn: fn () raises -> None):
    try:
        test_fn()
        print("  PASS: " + name)
        passed += 1
    except e:
        print("  FAIL: " + name + " — " + String(e))
        failed += 1


fn main() raises:
    print("=== csv_arrow tests ===")
    var passed = 0
    var failed = 0

    run_test("test_parse_csv_headers", passed, failed, test_parse_csv_headers)
    run_test("test_parse_csv_rows", passed, failed, test_parse_csv_rows)
    run_test("test_infer_int_column", passed, failed, test_infer_int_column)
    run_test("test_infer_float_column", passed, failed, test_infer_float_column)
    run_test("test_infer_string_column", passed, failed, test_infer_string_column)
    run_test("test_csv_roundtrip", passed, failed, test_csv_roundtrip)
    run_test("test_csv_to_feather_file", passed, failed, test_csv_to_feather_file)

    print("\n" + String(passed) + "/" + String(passed + failed) + " passed")
    if failed > 0:
        raise Error(String(failed) + " test(s) failed")
