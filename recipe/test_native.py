from pathlib import Path
import struct
import awkward as ak
import awkward_cpp

array = ak.Array([[1, 2], [], [3, 4, 5]])
assert ak.to_list(ak.sum(array, axis=-1)) == [3, 0, 12]
assert ak.to_list(ak.flatten(array)) == [1, 2, 3, 4, 5]
assert ak.to_list(array[array > 2]) == [[], [], [3, 4, 5]]

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
