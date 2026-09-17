# schema.mojo — Arrow Field and Schema types using the new type system.
# Phase 1 of the arrow refactor.

from dtypes import AnyDataType


struct Field(ImplicitlyCopyable, Movable):
    """A single column descriptor: name, Arrow type, and nullability."""

    var name: String
    var dtype: AnyDataType
    var nullable: Bool

    fn __init__(out self, name: String, dtype: AnyDataType, nullable: Bool = True):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable

    fn __copyinit__(out self, copy: Self):
        self.name = copy.name
        self.dtype = copy.dtype
        self.nullable = copy.nullable

    fn __moveinit__(out self, deinit take: Self):
        self.name = take.name^
        self.dtype = take.dtype^
        self.nullable = take.nullable


struct Schema(Copyable, Movable):
    """An ordered list of Fields plus an endianness indicator."""

    var fields: List[Field]
    var endianness: Int16  # 0 = little-endian, 1 = big-endian

    fn __init__(out self, fields: List[Field], endianness: Int16 = Int16(0)):
        # Explicit element copy — List[Field] is not ImplicitlyCopyable.
        self.fields = List[Field]()
        for i in range(len(fields)):
            self.fields.append(fields[i])
        self.endianness = endianness

    fn __copyinit__(out self, copy: Self):
        self.fields = List[Field]()
        for i in range(len(copy.fields)):
            self.fields.append(copy.fields[i])
        self.endianness = copy.endianness

    fn __moveinit__(out self, deinit take: Self):
        self.fields = take.fields^
        self.endianness = take.endianness

    def n_fields(self) -> Int:
        return len(self.fields)

    def field(self, i: Int) -> Field:
        return self.fields[i]
