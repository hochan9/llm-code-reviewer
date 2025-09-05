# src/services/llms/gemini.py
from __future__ import annotations

from typing import List, Dict
import google.generativeai as genai
from unidiff.patch import Hunk, PatchedFile

from ...core.config import Config
from ...core.models import PRDetails
from .base import BaseLLMService


class GeminiService(BaseLLMService):
    """Google Gemini API 구현."""

    def __init__(self) -> None:
        api_key = getattr(Config, "GEMINI_API_KEY", None)
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 가 설정되어 있지 않습니다.")
        genai.configure(api_key=api_key)

        # 1.5 Flash는 단계적 종료 예정 → 기본 모델을 2.0 Flash로 권장
        model_name = getattr(Config, "GEMINI_MODEL", None) or "gemini-2.0-flash"
        self.model = genai.GenerativeModel(model_name)

    def create_prompt(self, file: PatchedFile, hunk: Hunk, pr_details: PRDetails) -> str:
        """
        코드 리뷰 지시 프롬프트. 반드시 JSON만 반환하도록 요구.
        표준 스키마:
        {
          "reviews": [
            { "lineNumber": <int>, "side": "LEFT|RIGHT", "reviewComment": "...", "severity": "nit|warn|crit" }
          ]
        }
        """
        file_path = getattr(file, "path", "") or getattr(file, "source_file", "") or ""
        return f"""
당신은 숙련된 코드 리뷰어입니다. 다음 변경에서 버그/보안/성능/가독성 문제와 개선점을 찾아주세요.
PR 제목: {pr_details.title}
PR 설명: {pr_details.description}
파일 경로: {file_path}

반드시 아래 **JSON만** 출력하세요(코드펜스/설명 금지):
{{
  "reviews": [
    {{ "lineNumber": 0, "side": "RIGHT", "reviewComment": "설명", "severity": "nit|warn|crit" }}
  ]
}}

변경(diff):
```diff
{str(hunk)}
    """.strip()

def _extract_text(self, resp) -> str:
    """
    SDK 버전별 응답 포맷 차이에 대응하여 텍스트를 안전하게 추출.
    우선 resp.text, 없으면 candidates[].content.parts[].text를 조합.
    """
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    try:
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            collected: List[str] = []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    collected.append(t)
            if collected:
                return "\n".join(collected)
    except Exception:
        pass
    return ""

def get_ai_response(self, prompt: str) -> List[Dict[str, str]]:
    """Gemini 호출 → 텍스트 추출 → 정제(JSON 파싱) → 표준 스키마로 반환."""
    try:
        resp = self.model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1024, "temperature": 0.3},
        )
        raw = self._extract_text(resp)
        cleaned = self._clean_response_text(raw)
        return self._parse_response(cleaned)
    except Exception as e:
        # 필요 시 로거로 교체 가능
        print(f"[GeminiService] Error during API call: {e}")
        return []
