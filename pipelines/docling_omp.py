import gc
import os
import shutil
import tempfile
import time

from pathlib import Path

import ocrmypdf

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


class DoclingOCRmyPDFPipeline(BasePipeline):
    name = "docling_ocrmypdf"

    def run(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)

        output_dir = Path("outputs") / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        tmp_pdf = None

        try:
            # -----------------------------------------
            # STEP 1: OCRmyPDF
            # -----------------------------------------
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as tmp:
                tmp_pdf = tmp.name

            print(f"\n[{self.name}] Running OCRmyPDF...")

            ocrmypdf.ocr(
                str(pdf_path),
                tmp_pdf,
                language="eng",
                skip_text=True,
                optimize=1,
                output_type="pdf",
                progress_bar=False,
            )

            # -----------------------------------------
            # STEP 2: Docling
            # -----------------------------------------
            print(f"[{self.name}] Running Docling...")

            pipeline_options = PdfPipelineOptions(
                layout_options=LayoutOptions(
                    model_spec=DOCLING_LAYOUT_HERON
                ),
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE,
                    do_cell_matching=True,
                ),
                do_ocr=False,
            )

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )

            result = converter.convert(tmp_pdf)

            markdown = result.document.export_to_markdown(strict_text=True)

            # -----------------------------------------
            # STEP 3: Save Markdown
            # -----------------------------------------
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
            # Cleanup temp OCR PDF
            if tmp_pdf and os.path.exists(tmp_pdf):
                os.unlink(tmp_pdf)

            gc.collect()