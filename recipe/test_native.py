from pathlib import Path
import ctypes
import io
import struct

import numpy as np

import awkward_cpp
import awkward_cpp.libawkward
from awkward_cpp.lib._ext import ArrayBuilder, ForthMachine32, fromjsonobj

# The tests-cpu-kernels suites cover the CPU kernels. These checks exercise the
# remaining native code (the pybind11 extension and the libawkward C API)
# without importing the awkward Python package.

# pybind11 extension: ArrayBuilder
builder = ArrayBuilder()
for sublist in [[1, 2], [], [3, 4, 5]]:
    builder.beginlist()
    for x in sublist:
        builder.integer(x)
    builder.endlist()
assert len(builder) == 3
form, length, buffers = builder.to_buffers()
assert length == 3
assert '"ListOffsetArray"' in form, form
offsets, content = (np.asarray(buffers[key]) for key in sorted(buffers))
np.testing.assert_array_equal(offsets.view(np.int64), [0, 2, 2, 5])
np.testing.assert_array_equal(content.view(np.int64), [1, 2, 3, 4, 5])

# pybind11 extension: JSON reader (RapidJSON)
json_builder = ArrayBuilder()
fromjsonobj(
    io.BytesIO(b'{"x": 1.5}\n{"x": 2.5}'),
    json_builder,
    read_one=False,
    buffersize=1024,
    nan_string=None,
    posinf_string=None,
    neginf_string=None,
)
json_form, json_length, json_buffers = json_builder.to_buffers()
assert json_length == 2
assert '"RecordArray"' in json_form, json_form
(x_values,) = json_buffers.values()
np.testing.assert_allclose(np.asarray(x_values).view(np.float64), [1.5, 2.5])

# pybind11 extension: AwkwardForth virtual machine
vm = ForthMachine32("1 2 + 3 *")
vm.run()
assert vm.stack == [9], vm.stack

# libawkward C API, called through ctypes on the pybind11 ArrayBuilder
c_builder = ArrayBuilder()
assert awkward_cpp.libawkward.ArrayBuilder_real(c_builder._ptr, 1.5) == 0
assert awkward_cpp.libawkward.ArrayBuilder_real(c_builder._ptr, 2.5) == 0
c_length = ctypes.c_int64(-1)
assert awkward_cpp.libawkward.ArrayBuilder_length(c_builder._ptr, ctypes.byref(c_length)) == 0
assert c_length.value == 2

# All shipped binaries are native ARM64 PE files
root = Path(awkward_cpp.__file__).parent
binaries = [p for p in root.rglob("*") if p.suffix in (".dll", ".pyd")]
assert any(p.suffix == ".dll" for p in binaries), "Missing native CPU kernels"
assert any(p.suffix == ".pyd" for p in binaries), "Missing native Python extension"
for path in binaries:
    data = path.read_bytes()
    assert data[:2] == b"MZ", path
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[offset:offset + 4] == b"PE\0\0", path
    assert struct.unpack_from("<H", data, offset + 4)[0] == 0xAA64, path
    print("Verified ARM64:", path.name)
