"""Decode the text of Horowitz ISSCC 2014 Fig. 1.1.9 ("Rough energy costs for various
operations in 45nm 0.9V") directly from the PDF page content stream.

Why: pdftotext prints the figure's labels shifted ("S-" for "pJ", "ELW" for "bit") and drops
the digits, because the figure's subset font maps its byte codes 29 below ASCII and the
digit codes (0x13-0x1C) are control bytes. This script inflates every FlateDecode stream
(stdlib zlib only), walks each BT...ET text object, records the active font (Tf) and the
literal-string operands of Tj/TJ, and prints, for every text object that contains bytes
below 0x20 (the shifted font), the raw bytes and the bytes shifted by +29.

Usage: python -I decode_horowitz_fig119.py <path-to-pdf>
Reads the PDF only; writes nothing.
"""
import re
import sys
import zlib

SHIFT = 29


def unescape(s: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == 0x5C and i + 1 < len(s):
            n = s[i + 1]
            if 0x30 <= n <= 0x37:
                j = i + 1
                digits = b""
                while j < len(s) and len(digits) < 3 and 0x30 <= s[j] <= 0x37:
                    digits += bytes([s[j]])
                    j += 1
                out.append(int(digits, 8) & 0xFF)
                i = j
                continue
            table = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}
            out.append(table.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return bytes(out)


def shifted(b: bytes) -> str:
    return "".join(chr(x + SHIFT) for x in b)


def main(path: str) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    data = open(path, "rb").read()
    n_stream = 0
    for m in re.finditer(rb"(?<!end)stream\r?\n", data):
        n_stream += 1
        start = m.end()
        end = data.find(b"endstream", start)
        try:
            x = zlib.decompressobj().decompress(data[start:end])
        except zlib.error:
            continue
        if b"BT" not in x:
            continue
        for bt in re.finditer(rb"BT\b(.*?)\bET", x, re.S):
            body = bt.group(1)
            fonts = re.findall(rb"(/[A-Za-z0-9]+) [0-9.]+ Tf", body)
            tm = re.findall(rb"([-0-9. ]+) Tm", body)
            pieces = [unescape(sm.group(0)[1:-1]) for sm in re.finditer(rb"\((?:\\.|[^\\)])*\)", body, re.S)]
            raw = b"".join(pieces)
            # Keep only text objects drawn in the shifted font: a large share of bytes < 0x60
            # and no lowercase ASCII run typical of normal body text.
            if not raw or sum(c < 0x20 for c in raw) == 0 or re.search(rb"[a-z]{3}", raw):
                continue
            print(f"stream={n_stream} fonts={[f.decode() for f in fonts]} Tm={[t.decode().strip() for t in tm][:1]}")
            print(f"   raw={raw!r}")
            print(f"   +29={shifted(raw)!r}")
            # The Fig. 1.1.9 fonts use 2-byte codes (high byte 0x00); drop the high bytes.
            print(f"   clean(+29, 0x00 high bytes dropped)={shifted(raw.replace(bytes([0]), b''))!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
