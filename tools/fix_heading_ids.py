"""给含英文字母的小节标题和实验标题补上纯数字锚点，使“回看 1.3.2”这类链接稳定可用。

用法：python tools/fix_heading_ids.py        （处理全部章节，可重复运行）
规则：## 1.3 / ### 1.3.2 这类标题 → { #13 } / { #132 }；### 实验 1-2 → { #1-2 }。已有 { #… } 的不改。
"""
import re
from pathlib import Path

for p in sorted(Path("docs/chapters").glob("*/index.md")):
    lines = p.read_text(encoding="utf-8").split("\n")
    changed = 0
    for i, line in enumerate(lines):
        if "{ #" in line or "{#" in line:
            continue
        m = re.match(r"^(##+) (\d+(?:\.\d+)+) (.*)$", line)
        if m and re.search(r"[A-Za-z]", m.group(3)):
            lines[i] = f"{line} {{ #{m.group(2).replace('.', '')} }}"
            changed += 1
            continue
        m = re.match(r"^### 实验 (\d+-\d+) ", line)
        if m:
            lines[i] = f"{line} {{ #{m.group(1)} }}"
            changed += 1
    if changed:
        p.write_text("\n".join(lines), encoding="utf-8")
        print(f"{p}: 补了 {changed} 个锚点")
