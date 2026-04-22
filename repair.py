# repair.py — run this once, then run dnabert.py
from pathlib import Path
import importlib.util, re

dmu = Path(importlib.util.find_spec("transformers").origin).parent / "dynamic_module_utils.py"
txt = dmu.read_text(encoding="utf-8")

# Strip ALL our previous patch attempts
txt = re.sub(r"# WIN_SKIP_TRITON\n.*?(?=\n        raise ImportError)", 
             "", txt, flags=re.DOTALL)
txt = re.sub(r"        # WIN_SKIP_TRITON\n", "", txt)
txt = re.sub(r"        _skip.*?\n", "", txt)
txt = re.sub(r"        if not _skip.*?\n", "", txt)
txt = re.sub(r"    # WIN_SKIP_TRITON\n", "", txt)

# Find exact indentation and raise line
idx = txt.find("raise ImportError(")
if idx == -1:
    print("raise ImportError not found — already clean or different version")
else:
    # Get the line
    line_start = txt.rfind("\n", 0, idx) + 1
    indent = len(txt[line_start:idx])
    ind = " " * indent
    
    # Find end of the raise block (closing paren on its own line)
    end = txt.find("\n" + ind + ")", idx)
    end = txt.find("\n", end + 1)  # include the closing paren line
    
    old_block = txt[line_start:end]
    new_block = (
        ind + "# WIN_SKIP_TRITON\n" +
        ind + "_skip_pkgs = {'triton', 'flash_attn', 'flash_attn_2'}\n" +
        ind + "if not _skip_pkgs.issuperset(set(missing_packages)):\n" +
        "\n".join("    " + l for l in old_block.splitlines())
    )
    txt = txt[:line_start] + new_block + txt[end:]
    dmu.write_text(txt, encoding="utf-8")
    print(f"✓ Fixed. Indentation={indent}. Snippet:")
    print(txt[line_start:line_start+300])