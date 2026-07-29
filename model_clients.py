from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()


@dataclass
class ModelResult:
    """Speichert das Ergebnis eines einzelnen Modellaufrufs."""

    decision: str | None
    raw_response: str
    latency_seconds: float
    status: str
    error: str | None = None


def parse_decision(text: str) -> str | None:
    """
    Extrahiert eine eindeutige Entscheidung aus der Modellantwort.

    Akzeptierte Antworten:
    - A
    - B
    - Option A
    - Option B
    - Seçenek A
    - Seçenek B

    Andere oder mehrdeutige Antworten werden nicht akzeptiert.
    """
    cleaned = text.strip().upper()

    if cleaned in {"A", "B"}:
        return cleaned

    match = re.fullmatch(
        r"(?:OPTION|SEÇENEK)?\s*([AB])[\.\)]?",
        cleaned,
    )

    return match.group(1) if match else None


class OpenAIModel:
    """Client für das OpenAI-Modell."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-4o",
        )

        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    def _request(self, prompt: str) -> str:
        """
        Sendet einen Prompt an die OpenAI API.

        Bei einem Fehler wird der Aufruf höchstens viermal wiederholt.
        """
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0,
            max_output_tokens=16,
        )

        return response.output_text or ""

    def run(self, prompt: str) -> ModelResult:
        """Führt einen Modellaufruf durch und misst dessen Laufzeit."""
        start = time.perf_counter()

        try:
            text = self._request(prompt)
            decision = parse_decision(text)

            return ModelResult(
                decision=decision,
                raw_response=text,
                latency_seconds=time.perf_counter() - start,
                status="ok" if decision else "invalid_response",
                error=(
                    None
                    if decision
                    else "The model response could not be parsed as A or B."
                ),
            )

        except Exception as exc:
            return ModelResult(
                decision=None,
                raw_response="",
                latency_seconds=time.perf_counter() - start,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )


class GeminiModel:
    """Client für das Google-Gemini-Modell."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.1-pro-preview",
        )

        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    def _request(self, prompt: str) -> str:
        """
        Sendet einen Prompt an die Gemini API.

        Bei einem Fehler wird der Aufruf höchstens viermal wiederholt.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0,
                "max_output_tokens": 1024,
            },
        )

        return response.text or ""

    def run(self, prompt: str) -> ModelResult:
        """Führt einen Modellaufruf durch und misst dessen Laufzeit."""
        start = time.perf_counter()

        try:
            text = self._request(prompt)
            decision = parse_decision(text)

            return ModelResult(
                decision=decision,
                raw_response=text,
                latency_seconds=time.perf_counter() - start,
                status="ok" if decision else "invalid_response",
                error=(
                    None
                    if decision
                    else "The model response could not be parsed as A or B."
                ),
            )

        except Exception as exc:
            return ModelResult(
                decision=None,
                raw_response="",
                latency_seconds=time.perf_counter() - start,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )