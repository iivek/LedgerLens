"""HuggingFace transformers runner for Qwen2-VL family (teacher & student, GPU box).

Requires the `gpu` extra. Works for stock, LoRA-merged, AWQ/GPTQ-quantized and
pruned checkpoints alike — anything loadable via AutoModelForImageTextToText.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import MAX_NEW_TOKENS, MAX_PIXELS, MIN_PIXELS


class HFVLMRunner:
    def __init__(self, model_id: str, max_new_tokens: int = MAX_NEW_TOKENS,
                 device_map: str = "auto", torch_dtype: str = "auto",
                 load_in_4bit: bool = False, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS):
        import torch  # noqa: F401  (fail fast with a clear error if gpu extra missing)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self._model_id = model_id
        self._load_in_4bit = load_in_4bit
        self._min_pixels, self._max_pixels = min_pixels, max_pixels
        kwargs: dict = {"device_map": device_map, "torch_dtype": torch_dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16"
            )
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.eval()
        # Bounded pixel budget => bounded vision tokens => comparable latency numbers
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def provenance(self) -> dict:
        import torch
        import transformers

        return {
            "runtime": "transformers/HFVLMRunner",
            "model_id": self._model_id,
            "load_in_4bit": self._load_in_4bit,
            "min_pixels": self._min_pixels, "max_pixels": self._max_pixels,
            "max_new_tokens": self.max_new_tokens,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

    def generate(self, image_path: Path, prompt: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
