from langchain.llms.base import LLM
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
from typing import Optional, List
from pydantic import PrivateAttr


class PegasusLLM(LLM):
    # Pydantic-safe private attributes
    _tokenizer: PegasusTokenizer = PrivateAttr()
    _model: PegasusForConditionalGeneration = PrivateAttr()
    _device: str = PrivateAttr()
    _max_tokens: int = PrivateAttr()

    def __init__(self, model_path: str, max_tokens: int = 1024):
        super().__init__()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = PegasusTokenizer.from_pretrained(model_path)
        self._model = PegasusForConditionalGeneration.from_pretrained(model_path).to(self._device)
        self._max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "pegasus"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self._device)
        output_ids = self._model.generate(
            inputs["input_ids"],
            max_length=self._max_tokens,
            num_beams=4,
            early_stopping=True
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
