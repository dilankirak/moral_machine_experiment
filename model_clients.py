import json
import os
import re
import time
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


load_dotenv()


class Decision(str, Enum):
    A = "A"
    B = "B"


@dataclass
class ModelResult:
    decision: str | None
    raw_response: str
    latency_seconds: float
    status: str
    error: str | None = None


def parse_decision(text):
    cleaned = text.strip().upper()

    if cleaned in {"A", "B"}:
        return cleaned

    match = re.fullmatch(
        r"(?:OPTION|SEÇENEK)\s+([AB])[\.\)]?",
        cleaned,
    )

    return match.group(1) if match else None


def run_request(request, prompt):
    start = time.perf_counter()

    try:
        text = request(prompt)
        decision = parse_decision(text)

        return ModelResult(
            decision=decision,
            raw_response=text,
            latency_seconds=time.perf_counter() - start,
            status="ok" if decision else "invalid_response",
            error=None if decision else "Response is not A or B.",
        )

    except Exception as error:
        return ModelResult(
            decision=None,
            raw_response="",
            latency_seconds=time.perf_counter() - start,
            status="error",
            error=f"{type(error).__name__}: {error}",
        )


class OpenAIModel:
    def __init__(self, model=None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=30),
        reraise=True,
    )
    def _request(self, prompt):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0,
            max_output_tokens=32,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "enum": ["A", "B"],
                            }
                        },
                        "required": ["decision"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        raw_text = response.output_text or ""
        data = json.loads(raw_text)
        decision = data.get("decision")

        if decision not in {"A", "B"}:
            raise ValueError(
                f"OpenAI returned an invalid decision: {raw_text!r}"
            )

        return decision

    def run(self, prompt):
        return run_request(self._request, prompt)


class GeminiModel:
    def __init__(self, model=None):
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
    def _request(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=128,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                ),
                response_mime_type="text/x.enum",
                response_schema=Decision,
            ),
        )

        text = (response.text or "").strip().upper()

        if text not in {"A", "B"}:
            raise ValueError(
                f"Gemini returned an invalid decision: {text!r}"
            )

        return text

    def run(self, prompt):
        return run_request(self._request, prompt)