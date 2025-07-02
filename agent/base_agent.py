
# base_llm_agent.py
from __future__ import annotations
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Union
import json
import re
from transformers import AutoProcessor
from langchain.chat_models.base import BaseChatModel
from langchain_community.chat_models import ChatOllama
from qwen_vl_utils import process_vision_info 
from langchain_openai import ChatOpenAI  


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


@dataclass
class BaseLLMAgent:
 
    llm: Union[BaseChatModel, str, None] = None
    system_prompt: str = ""
    temperature: float = 0.1

   
    response_format: Dict[str, str] = field(
        default_factory=lambda: {"type": "json_object"}
    )

    _model: BaseChatModel = field(init=False, repr=False)
    _backend: str = field(init=False, repr=False) #need when using openai
    def __post_init__(self) -> None:
        if isinstance(self.llm, BaseChatModel):
            self._model = self.llm
            self._backend = "custom"
        model_name = self.llm or "gpt-4o-mini"
        
        self._model = ChatOpenAI(
            model=model_name,
            temperature=self.temperature,
            response_format=self.response_format,
        )
        self._backend = "openai"
        # else:
        #     model_name = self.llm or "qwen2.5vl:3b"
        #     print(f"[Agent Init] Using ChatOllama model: {model_name}")
        #     self._model = ChatOllama(model=model_name, base_url="http://localhost:11434")
        try:
            self._processor = AutoProcessor.from_pretrained(model_name)
        except Exception:
            self._processor = None 


    def _encode_image(self, image_path: Path| str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


    def _strip_fences(self, text: str) -> str:
        """Remove leading/trailing ```json fences so json.loads() can succeed."""
        return _FENCE_RE.sub("", text.strip())


    def call(
        self,
        user_text: str,
        *,
        screenshots: list[Path | str] | None = None,
        system_prompt: str | None = None,
    ) -> dict:
        """
        If `screenshot` is provided **and** the model is multimodal, send
        text + image; otherwise fall back to text-only.  Always returns a dict
        parsed from the model’s JSON output.
        """
        img_blocks = []
        if screenshots and self._processor:
            if len(screenshots) > 2:
                raise ValueError("Pass at most 2 screenshots")
            for path in screenshots:
                img_blocks.append(
                    {"type": "image",
                    "image": f"data:image/png;base64,{self._b64(path)}"}
                )

            user_block = img_blocks + [{"type": "text", "text": user_text.strip()}]
            msgs = [
                {"role": "system",
                    "content": [{"type": "text", "text": (system_prompt or self.system_prompt).strip()}]},
                {"role": "user",
                    "content": user_block}
            ]
            prompt = self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            img_inputs, _, kw = process_vision_info(msgs, True)
            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": img_inputs},
                "mm_processor_kwargs": kw,
            }
            raw = self._model.invoke(llm_inputs, temperature=self.temperature, response_format=self.response_format)
        else:
            # text-only mode 
            raw = self._model.invoke(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user",   "content": user_text}],
                temperature=self.temperature,
                response_format=self.response_format,
            )

        # Same JSON-parsing logic you already had:
        text_out = getattr(raw, "content", raw).strip()
        try:
            return json.loads(self._strip_fences(text_out))
        except json.JSONDecodeError as e:
            raise ValueError(f"Model output not valid JSON:\n{text_out}") from e
        
    """
    def _call(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    
        try:
            # Try native JSON mode first (OpenAI, Anthropic, etc.)
            resp = self._model.invoke(
                messages,
                temperature=self.temperature,
                response_format=self.response_format,
            )
            # Some back-ends already return a Mapping (e.g. OpenAI 2024-04+)
            if isinstance(resp, Mapping):
                return dict(resp)

            raw_text = resp.content  # type: ignore[attr-defined]
        except TypeError:
            # Provider doesn't accept `response_format`; fall back to plain call.
            raw_text = self._model.invoke(
                messages, temperature=self.temperature
            ).content  # type: ignore[attr-defined]

        # Fallback parse
        try:
            return json.loads(self._strip_fences(raw_text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Action output not valid JSON:\n{raw_text}") from exc
"""