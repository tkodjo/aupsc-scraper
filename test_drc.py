from analyze import extract_text_pages, extract_countries_from_text, _norm
import re, sys
path = sys.argv[1]
pages = extract_text_pages(path)
print("Pages:", len(pages))
for i, p in enumerate(pages, 1):
    blob = _norm(p)
    has_drc = any(x in blob for x in ["democratic republic of the congo","democratic republic of congo","drc","republique democratique du congo","rdc"])
    print(f"Page {i}: DRC-ish? {has_drc} | len={len(blob)}")
    if has_drc:
        print(p[:800], "\n---\n")
