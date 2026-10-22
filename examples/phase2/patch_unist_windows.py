"""Apply minimal Windows/KnowAir patches to third_party/UniST (idempotent).

- DataLoader: num_workers=0 (Windows spawn issues)
- DataLoader: treat TrafficKnowAir like Traffic* timestamps (already via name)
- train.py / main: leave as-is unless needed

Usage:
  .venv-ml\\Scripts\\python.exe examples/phase2/patch_unist_windows.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "third_party" / "UniST" / "src"
LOADER = ROOT / "DataLoader.py"

MARKER = "# PHASE2_WINDOWS_PATCH"


def main() -> int:
    text = LOADER.read_text(encoding="utf-8")
    if MARKER in text:
        print("DataLoader already patched")
        return 0

    old = "data = th.utils.data.DataLoader(data, num_workers=4, batch_size=batch_size, shuffle=True) \n    test_data = th.utils.data.DataLoader(test_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)\n    val_data = th.utils.data.DataLoader(val_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)"
    new = (
        f"{MARKER}\n"
        "    # Windows-friendly: num_workers=0\n"
        "    data = th.utils.data.DataLoader(data, num_workers=0, batch_size=batch_size, shuffle=True)\n"
        "    test_data = th.utils.data.DataLoader(test_data, num_workers=0, batch_size=4 * batch_size, shuffle=False)\n"
        "    val_data = th.utils.data.DataLoader(val_data, num_workers=0, batch_size=4 * batch_size, shuffle=False)"
    )
    if old not in text:
        # try without trailing space variants
        old2 = "num_workers=4"
        if old2 not in text:
            print("ERROR: unexpected DataLoader.py; patch manually")
            return 1
        text = text.replace("num_workers=4", "num_workers=0")
        text = f"{MARKER}\n" + text
        LOADER.write_text(text, encoding="utf-8")
        print(f"patched num_workers via replace → {LOADER}")
        return 0

    LOADER.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched {LOADER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
