
---

# ✅ `src/services/llms/base.py` (완성본 – 파서 안정/헬퍼 포함)
> `gemini.py`가 이 헬퍼들에 의존합니다. 아래 그대로 교체하세요.

```python
# src/services/llms/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import List, Dict
from unidiff.patch import Hunk, PatchedFile
from ...core.models import PRDetails


class BaseLLMService(ABC):
    """LLM 서비스 공통 인터페이스 및 유틸."""

    @abstractmethod
    def create_prompt(self, file: PatchedFile, hunk: Hunk, pr_details: PRDetails) -> str:
        """각 LLM에 맞는 프롬프트 생성."""
        raise NotImplementedError

    @abstractmethod
    def get_ai_response(self, prompt: str) -> List[Dict[str, str]]:
        """LLM 호출을 수행하고 표준 리뷰 리스트 반환."""
        raise NotImplementedError

    # ===== 공통 헬퍼 =====
    def _clean_response_text(self, text: str) -> str:
        """
        모델 출력에서 코드펜스와 주변 설명 제거 후,
        첫 번째 JSON 오브젝트/배열만 추출.
        """
        if not text:
            return ""

        # 코드펜스 제거(앞/뒤 한 번씩)
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text.strip())
        text = re.sub(r"\n```$", "", text)

        # JSON 블록만 추출: {...} 또는 [...]
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        return match.group(1).strip() if match else text

    def _parse_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        LLM이 반환한 JSON 문자열을 파싱하여 표준 스키마로 정제.
        표준 스키마: [{ lineNumber:int, side:'LEFT'|'RIGHT', reviewComment:str, severity?:str }]
        """
        if not response_text:
            return []

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return []

        if isinstance(data, dict) and "reviews" in data:
            items = data.get("reviews")
        elif isinstance(data, list):
            items = data
        else:
            return []

        results: List[Dict[str, str]] = []
        for it in items or []:
            if not isinstance(it, dict):
                continue
            try:
                line = int(it.get("lineNumber"))
                side = str(it.get("side", "RIGHT")).upper()
                if side not in ("LEFT", "RIGHT"):
                    side = "RIGHT"
                comment = str(it.get("reviewComment", "")).strip()
                if not comment:
                    continue
                out = {"lineNumber": line, "side": side, "reviewComment": comment}
                sev = it.get("severity")
                if isinstance(sev, str) and sev:
                    out["severity"] = sev
                results.append(out)
            except Exception:
                continue
        return results
