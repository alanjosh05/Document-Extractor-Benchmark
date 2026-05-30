from pathlib import Path
import subprocess
import shutil
import time

from pipelines.base import BasePipeline


class MinerUPipeline(BasePipeline):
    name = "mineru"

    def run(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)

        output_dir = Path("outputs") / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = Path("temp") / self.name
        temp_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            cmd_name = "mineru"
            if not shutil.which("mineru") and shutil.which("magic-pdf"):
                cmd_name = "magic-pdf"

            cmd = [
                cmd_name,
                "-p", str(pdf_path),
                "-o", str(temp_dir),
            ]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True, # Prevent spamming stdout
            )

            # Find generated markdown
            md_files = list(temp_dir.rglob("*.md"))

            if not md_files:
                raise FileNotFoundError("No markdown file generated.")

            md_file = md_files[0]

            final_md_path = output_dir / f"{pdf_path.stem}.md"

            shutil.copy(md_file, final_md_path)

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
            # Clean temp
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)