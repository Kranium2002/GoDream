# Converts a binary P6 PPM to PNG using only the standard library.
import sys, zlib, struct
data = open(sys.argv[1], "rb").read()
hdr, _, rest = data.partition(b"\n")
_, w, h, _ = hdr.split(); w, h = int(w), int(h)
raw = b"".join(b"\x00" + rest[y*w*3:(y+1)*w*3] for y in range(h))
def chunk(t, b): return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
open(sys.argv[2], "wb").write(png)
