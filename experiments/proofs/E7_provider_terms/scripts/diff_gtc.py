"""Diff the clause body (from 'Status:' line to clause 15.3) of two extracted GTC text files.
Usage: python -I scripts/diff_gtc.py <old.txt> <new.txt> <label_old> <label_new>"""
import difflib, sys
def body(p):
    L = open(p, encoding="utf-8").read().splitlines()
    s = next(i for i, x in enumerate(L) if x.startswith("Status:"))
    e = next(i for i, x in enumerate(L) if x.startswith("15.3."))
    return L[s:e + 1]
a, b = body(sys.argv[1]), body(sys.argv[2])
d = list(difflib.unified_diff(a, b, sys.argv[3], sys.argv[4], lineterm="", n=0))
print("\n".join(d) if d else "NO DIFFERENCES")
print(f"# lines old={len(a)} new={len(b)} changed_hunks={sum(1 for x in d if x.startswith('@@'))}")
