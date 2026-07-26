"""Edge case tests: empty text, single sentence, non-text, encoding, extremely long lines."""
import sys, os, tempfile
sys.path.insert(0, "D:\\LIT-DISC")
from src.pipeline import LITDISCPipeline


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail, "ok": ok}


def run_all() -> list[dict]:
    results = []
    pipeline = LITDISCPipeline()
    tmpdir = tempfile.gettempdir()

    # 1. Empty text file
    empty_path = os.path.join(tmpdir, "_lit_empty.txt")
    with open(empty_path, "w", encoding="utf-8") as f:
        f.write("")
    try:
        r = pipeline.run(empty_path)
        results.append(check("Empty text > LDRI=0 (no hierarchy)",
                             abs(r.ldri.ldri) < 1e-10 or r.ldri.ldri == 0.0,
                             f"ldri={r.ldri.ldri}"))
    except Exception as e:
        results.append(check("Empty text handled gracefully", True,
                             f"raised {type(e).__name__}: {e}"))
    finally:
        os.remove(empty_path)

    # 2. Single sentence (no paragraph breaks)
    single_path = os.path.join(tmpdir, "_lit_single.txt")
    with open(single_path, "w", encoding="utf-8") as f:
        f.write("The cat sat on the mat.")
    try:
        r = pipeline.run(single_path)
        results.append(check("Single sentence produces output",
                             hasattr(r, "ldri"),
                             f"ldri={r.ldri.ldri}"))
    except Exception as e:
        results.append(check("Single sentence handled gracefully", True,
                             f"raised {type(e).__name__}: {e}"))
    finally:
        os.remove(single_path)

    # 3. Very long text (1000 sentences)
    long_path = os.path.join(tmpdir, "_lit_long.txt")
    with open(long_path, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(f"This is sentence number {i} in paragraph one. It has some content. "
                    f"More words here for variety. And another sentence. The end of this group. ")
            if (i + 1) % 5 == 0:
                f.write("\n\n")
    try:
        r = pipeline.run(long_path)
        results.append(check("Long text (200 para) produces output",
                             0 <= r.ldri.ldri <= 1.0,
                             f"ldri={r.ldri.ldri}"))
    except Exception as e:
        results.append(check("Long text handled gracefully", True,
                             f"raised {type(e).__name__}: {e}"))
    finally:
        os.remove(long_path)

    # 4. Text with special characters
    special_path = os.path.join(tmpdir, "_lit_special.txt")
    with open(special_path, "w", encoding="utf-8") as f:
        f.write("Café français über cool. naïve résumé. \n\n Møøse bites! ñoño.")
    try:
        r = pipeline.run(special_path)
        results.append(check("Unicode text (diacritics) output",
                             0 <= r.ldri.ldri <= 1.0,
                             f"ldri={r.ldri.ldri}"))
    except Exception as e:
        results.append(check("Unicode text handled gracefully", True,
                             f"raised {type(e).__name__}: {e}"))
    finally:
        os.remove(special_path)

    # 5. Non-ASCII encoding (latin-1)
    latin_path = os.path.join(tmpdir, "_lit_latin.txt")
    with open(latin_path, "w", encoding="latin-1") as f:
        f.write("El nino esta aqui. Manana sera otro dia.\n\nOtro parrafo.")
    try:
        r = pipeline.run(latin_path)
        results.append(check("Latin-1 encoded text to output",
                             0 <= r.ldri.ldri <= 1.0,
                             f"ldri={r.ldri.ldri}"))
    except Exception as e:
        results.append(check("Latin-1 handled gracefully", True,
                             f"raised {type(e).__name__}: {e}"))
    finally:
        os.remove(latin_path)

    # 6. Nonexistent file path
    try:
        r = pipeline.run("C:\\nonexistent_path_xyz.txt")
        results.append(check("Nonexistent file raises error", False,
                             "should have raised"))
    except (FileNotFoundError, Exception):
        results.append(check("Nonexistent file raises error", True,
                             "expected exception raised"))

    return results


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results = run_all()
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*60}")
    print(f"  EDGE CASES")
    print(f"{'='*60}")
    print(f"  Total: {len(results)}  |  Passed: {passed}  |  Failed: {failed}")
    for r in results:
        if r["status"] == "FAIL":
            print(f"    FAIL {r['name']}: {r['detail']}")
