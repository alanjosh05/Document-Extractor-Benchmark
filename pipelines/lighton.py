import gc
import time
import torch

from pathlib import Path
from pdf2image import convert_from_path

from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
)

from pipelines.base import BasePipeline


class LightOnOCRPipeline(BasePipeline):
    name = "lightonocr"

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
            else torch.bfloat16
        )

        print(f"[{self.name}] Loading model on {self.device}...")

        self.model = (
            LightOnOcrForConditionalGeneration
            .from_pretrained(
                "lightonai/LightOnOCR-2-1B",
                torch_dtype=self.dtype,
            )
            .to(self.device)
        )

        self.processor = (
            LightOnOcrProcessor.from_pretrained(
                "lightonai/LightOnOCR-2-1B"
            )
        )

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

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": page,
                            }
                        ],
                    }
                ]

                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                inputs = {
                    k: (
                        v.to(
                            device=self.device,
                            dtype=self.dtype,
                        )
                        if v.is_floating_point()
                        else v.to(self.device)
                    )
                    for k, v in inputs.items()
                }

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                    )

                generated_ids = output_ids[
                    0,
                    inputs["input_ids"].shape[1]:,
                ]

                output_text = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                markdown_parts.append(
                    f"\n\n# Page {idx + 1}\n\n"
                    f"{output_text}"
                )

                # Memory cleanup per page
                del inputs
                del output_ids
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