from typing import List, Dict, Optional
import re
import time
import random
import threading
from functools import lru_cache

import google.generativeai as Client
from google.api_core import exceptions as gexc
from unidiff import Hunk, PatchedFile

from ...core.config import Config
from ...core.models import PRDetails
from .base import BaseLLMService


# ----- 분당 호출 제한기 (폭주 방지) -----
class MinuteRateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.lock = threading.Lock()
        self.count = 0
        self.window_start = time.time()

    def acquire(self):
        with self.lock:
            now = time.time()
            # 새 60초 윈도우로 교체
            if now - self.window_start >= 60:
                self.window_start = now
                self.count = 0
            # 꽉 찼으면 다음 윈도우까지 대기
            if self.count >= self.rpm:
                sleep_for = 60 - (now - self.window_start) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self.window_start = time.time()
                self.count = 0
            self.count += 1


def _extract_retry_seconds(err: Exception) -> Optional[int]:
    """
    Gemini SDK 에러 문자열에서 `retry_delay { seconds: N }` 값을 파싱.
    없으면 None.
    """
    m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(err))
    return int(m.group(1)) if m else None


class GeminiService(BaseLLMService):
    """
    Google의 Gemini 모델용 BaseLLMService 구현 (429/일시적 오류 대비 강화판)
    """

    def __init__(self):
        """구성값으로 Gemini 서비스를 초기화합니다."""
        Client.configure(api_key=Config.GEMINI_API_KEY)
        self.model = Client.GenerativeModel(Config.GEMINI_MODEL)

        # 환경에서 조절 가능. 값이 없으면 보수적으로 10 RPM 사용.
        rpm = getattr(Config, "GEMINI_RPM", 10)
        self.ratelimiter = MinuteRateLimiter(rpm)

        # 생성 파라미터도 Config에서 주입 가능 (없으면 기본값 사용)
        self._max_output_tokens = getattr(Config, "GEMINI_MAX_OUTPUT_TOKENS", 768)
        self._temperature = getattr(Config, "GEMINI_TEMPERATURE", 0.3)

    def create_prompt(self, file: PatchedFile, hunk: Hunk, pr_details: PRDetails) -> str:
        """Gemini의 기대 형식에 맞춘 프롬프트를 생성합니다."""
        return f"""
            Your task is to review the following code changes. Please follow these guidelines:
            Provide your response in this JSON format:
            {{"reviews": [{{"lineNumber": <line_number>, "reviewComment": "<review comment>", "side": "<left or right>", "filepath": "<file path>"}}]}}
            Important Rules:
            1. Line Number Validation:
            - For "left" side: {hunk.source_start} ≤ lineNumber < {hunk.source_start + hunk.source_length}
            - For "right" side: {hunk.target_start} ≤ lineNumber < {hunk.target_start + hunk.target_length}

            2. Review Focus Areas:
            - Critical bugs and errors
            - Security vulnerabilities and risks
            - Performance optimization opportunities
            - Code architecture and maintainability issues
            - Suggest code for improvement and optimization

            3. Key Requirements:
            - Return empty "reviews" array if no issues found
            - Use GitHub Markdown formatting in your comments
            - Do NOT suggest adding code comments
            - Provide feedback in language: {Config.HUMAN_LANGUAGE}

            Context Information:
            File: {file.path}
            PR Title: {pr_details.title}
            PR Description:
            ---
            {pr_details.description or 'No description provided'}
            ---

            Git Diff Details:
            - Source Start: {hunk.source_start}
            - Source Length: {hunk.source_length}
            - Target Start: {hunk.target_start}
            - Target Length: {hunk.target_length}

            Code Diff to Review:
            ```diff
            {hunk.__str__()}
            ```
        """

    # 동일 프롬프트 중복 호출을 제거 (최대 256개 캐시)
    @lru_cache(maxsize=256)
    def _cached_generate(self, prompt: str) -> str:
        """
        실제 Gemini 호출부.
        - 분당 호출 제한 (QPS 스로틀)
        - 429/일시적 오류에 대해 retry_delay 준수 + 지수 백오프(+지터)
        """
        self.ratelimiter.acquire()

        max_attempts = 6
        backoff = 2.0  # seconds

        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": self._max_output_tokens,
                        "temperature": self._temperature,
                    },
                )
                return resp.text or ""
            except (gexc.ResourceExhausted, gexc.TooManyRequests) as e:
                # 429: SDK가 제공한 retry_delay 우선
                delay = _extract_retry_seconds(e)
                if delay is None:
                    # 지수 백오프 + 지터
                    delay = min(backoff, 60.0) + random.uniform(0, 0.25 * backoff)
                    backoff = min(backoff * 2, 60.0)
                time.sleep(delay)
                continue
            except (gexc.ServiceUnavailable, gexc.DeadlineExceeded) as e:
                # 일시적 장애/타임아웃: 지수 백오프
                delay = min(backoff, 30.0) + random.uniform(0, 0.25 * backoff)
                backoff = min(backoff * 2, 60.0)
                time.sleep(delay)
                continue
            except Exception:
                # 그 외는 즉시 전파
                raise
        raise RuntimeError("Gemini generate_content: 재시도 한도를 초과했습니다.")

    def get_ai_response(self, prompt: str) -> List[Dict[str, str]]:
        """Gemini 모델에서 응답을 가져옵니다. (429/일시적 오류 대비)"""
        try:
            response_text = self._cached_generate(prompt)
            response_text = self._clean_response_text(response_text)
            return self._parse_response(response_text)
        except Exception as e:
            print(f"Gemini API 호출 실패: {e}")
            return []
