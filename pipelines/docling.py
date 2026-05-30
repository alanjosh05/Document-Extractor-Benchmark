import gc
import time
from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    LayoutOptions,
    TableStructureOptions,
)

from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_HERON,
)

from docling.datamodel.base_models import InputFormat

from pipelines.base import BasePipeline


class DoclingPipeline(BasePipeline):
    name = "docling"

    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            layout_options=LayoutOptions(
                model_spec=DOCLING_LAYOUT_HERON
            ),
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE,
                do_cell_matching=True,
            ),
            do_ocr=True,   # Docling handles OCR itself
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def run(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)

        output_dir = Path("outputs") / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            print(f"\n[{self.name}] Running Docling...")

            result = self.converter.convert(str(pdf_path))

            markdown = result.document.export_to_markdown(strict_text=True)

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