# src/libs/Hunk.py
from __future__ import annotations

from unidiff.patch import Hunk as _Hunk


class NumberedHunk(_Hunk):
    """Hunk.__str__를 확장해 원본/대상 라인 번호를 함께 보여주는 구현."""

    def __str__(self) -> str:
        source_line = self.source_start
        target_line = self.target_start
        out_lines = []

        for line in self:
            text = line.value.rstrip("\n")
            if line.is_removed:
                # 원본 라인 번호만 증가, 대상은 공백
                out_lines.append(f"{source_line:4d} {'':4} -{text}")
                source_line += 1
            elif line.is_added:
                # 대상 라인 번호만 증가, 원본은 공백
                out_lines.append(f"{'':4} {target_line:4d} +{text}")
                target_line += 1
            else:  # context line
                out_lines.append(f"{source_line:4d} {target_line:4d}  {text}")
                source_line += 1
                target_line += 1

        return "\n".join(out_lines)
