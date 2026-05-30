import gc
import os
import shutil
import tempfile
import time
import subprocess

from pathlib import Path

import ocrmypdf

from pipelines.base import BasePipeline


class MinerUOCRmyPDFPipeline(BasePipeline):
    name = "mineru_ocrmypdf"

    def run(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)

        output_dir = Path("outputs") / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = Path("temp") / self.name
        temp_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        ocr_pdf_path = None

        try:
            # -----------------------------------------
            # STEP 1: OCRmyPDF
            # -----------------------------------------
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as tmp:
                ocr_pdf_path = tmp.name

            print(f"\n[{self.name}] Running OCRmyPDF...")

            ocrmypdf.ocr(
                str(pdf_path),
                ocr_pdf_path,
                language="eng",
                skip_text=True,
                optimize=0,
                output_type="pdf",
                progress_bar=False,
            )

            print(f"[{self.name}] OCRmyPDF completed.")

            # -----------------------------------------
            # STEP 2: MinerU
            # -----------------------------------------
            print(f"[{self.name}] Running MinerU...")

            cmd_name = "mineru"
            if not shutil.which("mineru") and shutil.which("magic-pdf"):
                cmd_name = "magic-pdf"

            cmd = [
                cmd_name,
                "-p", str(ocr_pdf_path),
                "-o", str(temp_dir),
            ]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
            )

            # -----------------------------------------
            # STEP 3: Find markdown output
            # -----------------------------------------
            md_files = list(temp_dir.rglob("*.md"))

            if not md_files:
                raise FileNotFoundError(
                    "No markdown file generated."
                )

            md_file = md_files[0]

            final_md_path = (
                output_dir / f"{pdf_path.stem}.md"
            )

            shutil.copy(
                md_file,
                final_md_path,
            )

            elapsed = time.time() - start_time

            return {
                "success": True,
                "pipeline": self.name,
                "pdf": pdf_path.name,
                "markdown_path": str(final_md_path),
                "time_sec": round(elapsed, 2),
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "pipeline": self.name,
                "pdf": pdf_path.name,
                "error": f"{str(e)}\nStdout: {e.stdout.decode('utf-8', errors='ignore') if e.stdout else 'None'}\nStderr: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'None'}",
            }
        except Exception as e:
            return {
                "success": False,
                "pipeline": self.name,
                "pdf": pdf_path.name,
                "error": str(e),
            }

        finally:
            # Cleanup OCR PDF
            if (
                ocr_pdf_path
                and os.path.exists(ocr_pdf_path)
            ):
                os.unlink(ocr_pdf_path)

            # Cleanup MinerU temp output
            if temp_dir.exists():
                shutil.rmtree(
                    temp_dir,
                    ignore_errors=True,
                )

            gc.collect()