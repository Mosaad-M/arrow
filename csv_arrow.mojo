from std.pathlib import Path
from arrow import (
    ArrowType, ArrowField, ArrowSchema, ArrowArray, RecordBatch,
    encode_arrow_file, decode_arrow_file,
    TYPE_INT, TYPE_FLOAT, TYPE_UTF8, TYPE_BINARY, TYPE_NULL,
)
from flatbuffers import (
    write_i32_le, write_i64_le, write_f64_le,
    read_i32_le, read_i64_le, read_f64_le,
)


# ============================================================================
# CSV reading
# ============================================================================


fn read_csv(
    path: String,
) raises -> Tuple[List[String], List[List[String]]]:
    """
    Read a CSV file at path.
    Returns (column_names, rows) where all values are strings.
    First row is treated as the header.  Trailing empty lines are skipped.
    """
    var content = Path(path).read_text()
    var raw_lines = content.split("\n")

    # Collect non-empty lines (split returns StringSlice, convert to String)
    var lines = List[String]()
    for i in range(len(raw_lines)):
        if len(raw_lines[i]) > 0:
            lines.append(String(raw_lines[i]))

    if len(lines) == 0:
        raise Error("csv_arrow: read_csv: empty file")

    # First line: headers
    var header_parts = lines[0].split(",")
    var headers = List[String]()
    for i in range(len(header_parts)):
        headers.append(String(header_parts[i]))

    var n_cols = len(headers)
    var rows = List[List[String]]()

    for line_idx in range(1, len(lines)):
        var parts = lines[line_idx].split(",")
        if len(parts) != n_cols:
            raise Error(
                "csv_arrow: read_csv: row " + String(line_idx)
                + " has " + String(len(parts))
                + " columns, expected " + String(n_cols)
            )
        var row = List[String]()
        for i in range(len(parts)):
            row.append(String(parts[i]))
        rows.append(row^)

    return Tuple[List[String], List[List[String]]](headers^, rows^)


# ============================================================================
# Type inference
# ============================================================================


fn infer_type(values: List[String]) -> ArrowType:
    """
    Infer Arrow type for a column of string values.
    Empty strings are treated as nulls and skipped during inference.
    Priority: Int64 > Float64 > Utf8.
    """
    var has_non_empty = False
    var all_int = True
    var all_float = True

    for i in range(len(values)):
        if len(values[i]) == 0:
            continue  # null — skip for type inference
        has_non_empty = True

        if all_int:
            try:
                _ = Int(values[i])
            except:
                all_int = False

        if not all_int and all_float:
            try:
                _ = Float64(values[i])
            except:
                all_float = False

        if not all_int and not all_float:
            break   # no need to keep checking

    if not has_non_empty:
        return ArrowType.utf8()   # all-null column defaults to Utf8

    if all_int:
        return ArrowType.int_(64, True)
    elif all_float:
        return ArrowType.float_(2)
    else:
        return ArrowType.utf8()


fn infer_schema(
    names: List[String],
    rows: List[List[String]],
) raises -> ArrowSchema:
    """
    Infer an ArrowSchema from column names and CSV data rows.
    A column is nullable if any of its values is the empty string "".
    """
    var n_cols = len(names)
    var fields = List[ArrowField]()

    for col in range(n_cols):
        # Collect this column's values
        var col_values = List[String]()
        var nullable = False
        for row in range(len(rows)):
            var v = rows[row][col]
            col_values.append(v)
            if len(v) == 0:
                nullable = True

        var arrow_type = infer_type(col_values)
        fields.append(ArrowField(names[col], arrow_type, nullable))

    return ArrowSchema(fields, Int16(0))


# ============================================================================
# Array building from strings
# ============================================================================


fn _build_validity(values: List[String], null_count: Int) raises -> List[UInt8]:
    """Build a packed validity bitmap for a column with some null values."""
    var n = len(values)
    var n_bytes = (n + 7) // 8
    var validity = List[UInt8](capacity=n_bytes)
    for _ in range(n_bytes):
        validity.append(UInt8(0))
    for i in range(n):
        if len(values[i]) > 0:
            # Bit i in LSB-first order
            validity[i // 8] |= UInt8(1 << (i % 8))
    return validity^


fn build_array(
    type: ArrowType,
    nullable: Bool,
    values: List[String],
) raises -> ArrowArray:
    """
    Build an ArrowArray from string values, using `type` for binary encoding.
    Empty strings are treated as null values.
    """
    var length = len(values)

    # Count nulls
    var null_count = 0
    for i in range(length):
        if len(values[i]) == 0:
            null_count += 1

    # Build validity bitmap (only when there are nulls)
    var validity = List[UInt8]()
    if null_count > 0:
        validity = _build_validity(values, null_count)

    # ── Null type ────────────────────────────────────────────────────────────
    if type.tag == TYPE_NULL():
        return ArrowArray(type, length, length, List[UInt8](), List[UInt8](), List[UInt8]())

    # ── Int64 ────────────────────────────────────────────────────────────────
    if type.tag == TYPE_INT() and type.int_meta.bit_width == Int32(64):
        var value_bytes = List[UInt8]()
        for _ in range(length * 8):
            value_bytes.append(UInt8(0))
        for i in range(length):
            var int_val = Int64(0)
            if len(values[i]) > 0:
                int_val = Int64(Int(values[i]))
            write_i64_le(value_bytes, i * 8, int_val)
        return ArrowArray(type, length, null_count, validity, List[UInt8](), value_bytes)

    # ── Int32 and smaller ────────────────────────────────────────────────────
    if type.tag == TYPE_INT():
        var byte_width = Int(type.int_meta.bit_width) // 8
        var value_bytes = List[UInt8]()
        for _ in range(length * byte_width):
            value_bytes.append(UInt8(0))
        for i in range(length):
            var int_val = Int(0)
            if len(values[i]) > 0:
                int_val = Int(values[i])
            # Write LE bytes for the int value
            for b in range(byte_width):
                value_bytes[i * byte_width + b] = UInt8((int_val >> (b * 8)) & 0xFF)
        return ArrowArray(type, length, null_count, validity, List[UInt8](), value_bytes)

    # ── Float64 ─────────────────────────────────────────────────────────────
    if type.tag == TYPE_FLOAT() and type.float_meta.precision == UInt16(2):
        var value_bytes = List[UInt8]()
        for _ in range(length * 8):
            value_bytes.append(UInt8(0))
        for i in range(length):
            var f_val = Float64(0.0)
            if len(values[i]) > 0:
                f_val = Float64(values[i])
            write_f64_le(value_bytes, i * 8, f_val)
        return ArrowArray(type, length, null_count, validity, List[UInt8](), value_bytes)

    # ── Float32 ─────────────────────────────────────────────────────────────
    if type.tag == TYPE_FLOAT():
        # Encode as Float64 for simplicity (widen)
        var value_bytes = List[UInt8]()
        for _ in range(length * 8):
            value_bytes.append(UInt8(0))
        for i in range(length):
            var f_val = Float64(0.0)
            if len(values[i]) > 0:
                f_val = Float64(values[i])
            write_f64_le(value_bytes, i * 8, f_val)
        return ArrowArray(type, length, null_count, validity, List[UInt8](), value_bytes)

    # ── Utf8 / Binary ────────────────────────────────────────────────────────
    # Offsets: (length + 1) Int32 values
    var offsets = List[UInt8]()
    for _ in range((length + 1) * 4):
        offsets.append(UInt8(0))
    write_i32_le(offsets, 0, Int32(0))

    var value_bytes = List[UInt8]()
    var cur_byte = 0
    for i in range(length):
        if len(values[i]) > 0:
            var sb = values[i].as_bytes()
            for j in range(len(sb)):
                value_bytes.append(sb[j])
            cur_byte += len(sb)
        write_i32_le(offsets, (i + 1) * 4, Int32(cur_byte))

    return ArrowArray(type, length, null_count, validity, offsets, value_bytes)


# ============================================================================
# CSV → Feather
# ============================================================================


fn csv_to_feather(csv_path: String, feather_path: String) raises:
    """
    Full pipeline: read CSV → infer schema → encode as Arrow IPC file → write .feather.
    """
    var csv_result = read_csv(csv_path)
    var col_names = csv_result[0].copy()
    var rows = csv_result[1].copy()

    var schema = infer_schema(col_names, rows)

    # Build arrays — one per column
    var arrays = List[ArrowArray]()
    for col in range(len(col_names)):
        var col_values = List[String]()
        for row in range(len(rows)):
            col_values.append(rows[row][col])
        var field_type = schema.fields[col].type.copy()
        var nullable = schema.fields[col].nullable
        arrays.append(build_array(field_type, nullable, col_values))

    # Pack into one RecordBatch
    var n_rows = Int64(len(rows))
    var batch = RecordBatch(n_rows, arrays)
    var batches = List[RecordBatch]()
    batches.append(batch.copy())

    var file_bytes = encode_arrow_file(schema, batches)
    Path(feather_path).write_bytes(file_bytes)


# ============================================================================
# Feather → CSV
# ============================================================================


fn _array_value_to_string(col: ArrowArray, row: Int) raises -> String:
    """Return the string representation of element `row` in an ArrowArray."""
    # Check validity (null check)
    if col.null_count > 0 and len(col.validity) > 0:
        var byte_idx = row // 8
        var bit_idx = row % 8
        if byte_idx < len(col.validity):
            if ((col.validity[byte_idx] >> UInt8(bit_idx)) & UInt8(1)) == UInt8(0):
                return String("")  # null → empty string

    if col.type.tag == TYPE_NULL():
        return String("")

    if col.type.tag == TYPE_INT() and col.type.int_meta.bit_width == Int32(64):
        var v = read_i64_le(col.values, row * 8)
        return String(v)

    if col.type.tag == TYPE_INT():
        var byte_width = Int(col.type.int_meta.bit_width) // 8
        var v = Int64(0)
        for b in range(byte_width):
            v |= Int64(col.values[row * byte_width + b]) << Int64(b * 8)
        return String(v)

    if col.type.tag == TYPE_FLOAT() and col.type.float_meta.precision == UInt16(2):
        var v = read_f64_le(col.values, row * 8)
        return String(v)

    if col.type.tag == TYPE_FLOAT():
        var v = read_f64_le(col.values, row * 8)
        return String(v)

    # Utf8 / Binary: use offsets
    if col.type.tag == TYPE_UTF8() or col.type.tag == TYPE_BINARY():
        var off_start = Int(read_i32_le(col.offsets, row * 4))
        var off_end   = Int(read_i32_le(col.offsets, (row + 1) * 4))
        if off_end > len(col.values):
            raise Error("csv_arrow: _array_value_to_string: offsets out of bounds")
        var sb = List[UInt8]()
        for i in range(off_start, off_end):
            sb.append(col.values[i])
        return String(unsafe_from_utf8=sb^)

    return String("")


fn feather_to_csv(feather_path: String, csv_path: String) raises:
    """
    Inverse pipeline: read .feather → decode Arrow IPC file → write CSV.
    All batches are concatenated row-by-row.
    """
    var file_bytes = Path(feather_path).read_bytes()
    _ = len(file_bytes)
    var result = decode_arrow_file(file_bytes)
    var schema  = result[0].copy()
    var batches = result[1].copy()

    var n_cols = len(schema.fields)
    var out = String("")

    # Header row
    for c in range(n_cols):
        if c > 0:
            out += ","
        out += schema.fields[c].name
    out += "\n"

    # Data rows from all batches
    for b in range(len(batches)):
        var batch = batches[b].copy()
        var n_rows = Int(batch.length)
        for row in range(n_rows):
            for c in range(n_cols):
                if c > 0:
                    out += ","
                out += _array_value_to_string(batch.columns[c], row)
            out += "\n"

    Path(csv_path).write_text(out)
