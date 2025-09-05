# src/services/llms/base.py
from abc import ABC, abstractmethod
import json
import re
from typing import List, Dict
from unidiff.patch import Hunk, PatchedFile
from ...core.models import PRDetails


class BaseLLMService(ABC):
"""
공통 LLM 서비스 인터페이스 및 유틸.
하위 클래스는 `create_prompt`와 `get_ai_response`를 구현해야 합니다.
"""


@abstractmethod
def create_prompt(self, file: PatchedFile, hunk: Hunk, pr_details: PRDetails) -> str:
"""각 LLM에 맞는 프롬프트를 생성합니다."""
raise NotImplementedError


@abstractmethod
def get_ai_response(self, prompt: str) -> List[Dict[str, str]]:
"""LLM 호출을 수행하고, 표준화된 리뷰 리스트를 반환합니다."""
raise NotImplementedError


# ===== 공통 헬퍼 =====
def _clean_response_text(self, text: str) -> str:
"""
모델 출력에서 코드펜스와 주변 설명을 제거하고, 첫 번째 JSON 오브젝트/배열만 추출.
"""
if not text:
return ""


# 코드펜스 제거
text = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", text.strip())


# JSON 블록만 추출: { ... } 또는 [ ... ]
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