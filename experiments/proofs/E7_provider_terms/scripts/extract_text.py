"""Extract visible text from saved HTML pages (stdlib only, read-only).

Usage: python -I scripts/extract_text.py <raw_file> <out_txt>
Skips <script>, <style>, <noscript>, <svg>, <template>, <head> content.
Block-level tags produce line breaks so clause numbering stays on separate lines.
"""
import html
import re
import sys
from html.parser import HTMLParser

SKIP = {"script", "style", "noscript", "svg", "template", "head"}
BLOCK = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "td", "th", "table", "section", "article", "header", "footer",
    "nav", "blockquote", "pre", "dt", "dd", "dl", "hr", "main", "aside",
}


class Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip_depth += 1
        elif tag in BLOCK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth == 0:
            self.parts.append(data)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    raw = open(src, "rb").read().decode("utf-8", errors="replace")
    p = Extractor()
    p.feed(raw)
    text = "".join(p.parts)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if not ln:
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out).strip() + "\n")
    print(f"{src} -> {dst}: {len(raw)} raw chars, {sum(len(x) for x in out)} text chars")


if __name__ == "__main__":
    main()
