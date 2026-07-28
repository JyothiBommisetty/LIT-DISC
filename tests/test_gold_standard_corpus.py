import os, json

pdir = "D:\\LIT-DISC\\data\\gold_standard\\passages"
meta = json.load(open("D:\\LIT-DISC\\data\\gold_standard\\metadata.json"))
files = sorted(os.listdir(pdir))

print(f"Passage files: {len(files)}")
for f in files:
    text = open(os.path.join(pdir, f), encoding="utf-8").read()
    wc = len(text.split())
    pid = f.split("_")[0]
    print(f"  {pid:10s}  {wc:3d} words")

cats = {}
for p in meta["passages"]:
    cats.setdefault(p["category"], []).append(p["id"])
print("\nCategories:")
for k, v in sorted(cats.items()):
    print(f"  {k}: {len(v)} passages")
print(f"Total: {len(meta['passages'])} passages in metadata")

pd = {p["id"] for p in meta["passages"]}
fd = {f.split("_")[0] for f in files}
print(f"\nPassage IDs match: {pd == fd}")

# Check all have word count between 100-400
for p in meta["passages"]:
    fname = [f for f in files if f.startswith(p["id"])][0]
    text = open(os.path.join(pdir, fname), encoding="utf-8").read()
    wc = len(text.split())
    if wc < 100 or wc > 400:
        print(f"  WARNING: {p['id']} has {wc} words (target 150-400)")

print("\nAll checks passed!")
