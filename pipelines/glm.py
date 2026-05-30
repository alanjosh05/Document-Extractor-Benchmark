import gc
import time
import torch

from pathlib import Path
from pdf2image import convert_from_path

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)

from pipelines.base import BasePipeline


class GlmOCRPipeline(BasePipeline):
    name = "glmocr"

    def __init__(self):
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.dtype = (
            torch.float32
            if self.device == "mps"
            else torch.float16
        )

        print(f"[{self.name}] Loading model on {self.device}...")
        self.model_path = "zai-org/GLM-OCR"
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

    def run(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)

        output_dir = Path("outputs") / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            print(f"\n[{self.name}] Converting PDF pages to images...")

            pages = convert_from_path(
                str(pdf_path),
                dpi=200,
            )

            markdown_parts = []

            for idx, page in enumerate(pages):
                print(
                    f"[{self.name}] Processing page "
                    f"{idx + 1}/{len(pages)}"
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": page,
                            },
                            {
                                "type": "text",
                                "text": "Text Recognition:"
                            }
                        ],
                    }
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                
                inputs = {
                    k: (
                        v.to(
                            device=self.model.device,
                            dtype=self.dtype,
                        )
                        if v.is_floating_point()
                        else v.to(self.model.device)
                    )
                    for k, v in inputs.items()
                }
                
                inputs.pop("token_type_ids", None)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                    )

                generated_ids = generated_ids[
                    0,
                    inputs["input_ids"].shape[1]:,
                ]

                output_text = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=False,
                )

                markdown_parts.append(
                    f"\n\n# Page {idx + 1}\n\n"
                    f"{output_text}"
                )

                # Memory cleanup per page
                del inputs
                del generated_ids
                gc.collect()

                if self.device == "mps":
                    torch.mps.empty_cache()

            markdown = "\n".join(markdown_parts)

            final_md_path = (
                output_dir / f"{pdf_path.stem}.md"
            )

            final_md_path.write_text(
                markdown,
                encoding="utf-8",
            )

            elapsed = time.time() - start_time

            return {
                "success": True,
                "pipeline": self.name,
                "pdf": pdf_path.name,
                "markdown_path": str(final_md_path),
                "time_sec": round(elapsed, 2),
            }

        except Exception as e:
            return {
                "success": False,
                "pipeline": self.name,
                "pdf": pdf_path.name,
                "error": str(e),
            }

        finally:
            gc.collect()

            if self.device == "mps":
                torch.mps.empty_cache()