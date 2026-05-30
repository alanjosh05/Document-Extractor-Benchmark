# benchmark.py
"""Benchmark runner for PDF extraction pipelines.

Discovers PDF files in `input_pdfs/`, runs each pipeline defined in the
`pipelines` package sequentially, and writes a CSV file with the results.
"""

import csv
import gc
import time
from pathlib import Path
from typing import List, Dict

# Import all pipeline classes
from pipelines.docling import DoclingPipeline
from pipelines.docling_omp import DoclingOCRmyPDFPipeline
from pipelines.miner import MinerUPipeline
from pipelines.miner_omp import MinerUOCRmyPDFPipeline
from pipelines.docling_rapid import RapidOCRDoclingPipeline
from pipelines.glm import GlmOCRPipeline
# from pipelines.lighton import LightOnOCRPipeline

# Mapping of pipeline name to class for easy iteration
PIPELINES = [
    #DoclingPipeline(),
    #DoclingOCRmyPDFPipeline(),
    MinerUPipeline(),
    MinerUOCRmyPDFPipeline(),
    #RapidOCRDoclingPipeline(),
    #GlmOCRPipeline(),
    # LightOnOCRPipeline(),
]


def discover_pdfs() -> List[Path]:
    """Return a list of all PDF files in the `input_pdfs/` directory."""
    pdf_dir = Path("input_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return list(pdf_dir.rglob("*.pdf"))


import threading
import subprocess
from pdf2image import pdfinfo_from_path

def get_gpu_vram() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            text=True
        )
        return sum(float(x.strip()) for x in out.splitlines())
    except Exception:
        return 0.0

class VramMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.peak_vram = 0.0
        
    def run(self):
        while not self.stop_event.is_set():
            vram = get_gpu_vram()
            if vram > self.peak_vram:
                self.peak_vram = vram
            time.sleep(0.1)
            
    def stop(self):
        self.stop_event.set()
        self.join()
        return self.peak_vram

def run_pipeline(pipeline, pdf_path: Path) -> Dict:
    """Execute a pipeline on a single PDF and return the result dict."""
    try:
        pdf_info = pdfinfo_from_path(str(pdf_path))
        n_pages = pdf_info.get("Pages", 1)
    except Exception:
        n_pages = 1

    baseline_vram = get_gpu_vram()
    monitor = VramMonitor()
    monitor.start()

    result = pipeline.run(str(pdf_path))
    
    peak_vram = monitor.stop()
    vram_delta = peak_vram - baseline_vram

    # Ensure required keys exist
    result.setdefault("pipeline", pipeline.name)
    result.setdefault("pdf", pdf_path.name)
    result.setdefault("timestamp", time.time())
    result["n_pages"] = n_pages
    result["peak_vram_mb"] = max(0.0, round(vram_delta, 2))
    
    time_sec = result.get("time_sec", 0.0)
    if time_sec > 0:
        result["throughput"] = round(n_pages / time_sec, 4)
        if "latency_p95" not in result:
            # Fallback to average latency for bulk pipelines
            result["latency_p95"] = round(time_sec / n_pages, 4)
    else:
        result["throughput"] = 0.0
        result.setdefault("latency_p95", 0.0)
        
    return result


def write_csv(results: List[Dict], csv_path: Path) -> None:
    """Write a list of result dictionaries to ``csv_path``."""
    if not results:
        return
    # Determine column order
    columns = [
        "timestamp", "pipeline", "pdf", "success", "markdown_path", 
        "time_sec", "n_pages", "throughput", "peak_vram_mb", "latency_p95", "error"
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    """Main entry point for the benchmark.

    1. Discover PDFs.
    2. For each pipeline:
       a. Instantiate pipeline (loads model).
       b. Run on all PDFs.
       c. Destroy pipeline & clear memory.
    3. Save results to CSV iteratively.
    """
    pdf_files = discover_pdfs()
    if not pdf_files:
        print("[benchmark] No PDF files found in 'input_pdfs/'. Exiting.")
        return

    # Using the classes directly so we only load one into memory at a time
    pipeline_classes = [
        #DoclingPipeline,
        #DoclingOCRmyPDFPipeline,
        MinerUPipeline,
        MinerUOCRmyPDFPipeline,
        #RapidOCRDoclingPipeline,
        #GlmOCRPipeline,
        # LightOnOCRPipeline,
    ]

    all_results: List[Dict] = []
    csv_path = Path("outputs") / "benchmark_results.csv"

    for pipeline_cls in pipeline_classes:
        print(f"\n[benchmark] === Starting pipeline: {pipeline_cls.name} ===")
        # Instantiate pipeline (this loads the heavy models for Docling/LightOnOCR)
        pipeline = pipeline_cls()
        
        for pdf_path in pdf_files:
            print(f"[benchmark] Processing PDF: {pdf_path.name}")
            result = run_pipeline(pipeline, pdf_path)
            all_results.append(result)
            
            # Save results iteratively in case of a crash
            write_csv(all_results, csv_path)
            
            # Per-PDF garbage collection
            gc.collect()

        # Delete the pipeline instance to free up memory (models, etc.)
        pipeline_name = pipeline.name
        del pipeline
        gc.collect()

        # Clean up PyTorch memory if available (Apple Silicon or CUDA)
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        print(f"[benchmark] === Finished pipeline: {pipeline_name} ===")

    print(f"\n[benchmark] Benchmark completely finished. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
