# Document Extractor Benchmark

A comprehensive benchmarking framework for evaluating and comparing state-of-the-art PDF text and table extraction pipelines. 

This tool processes a folder of PDFs through multiple OCR and document parsing pipelines, dynamically capturing runtime metrics, and evaluates the outputs against a Ground Truth using both deterministic algorithms and an intelligent LLM-as-a-Judge.

## Supported Pipelines
- **MinerU**
- **Docling**
- **MinerU + OCRmyPDF**
- **Docling + OCRmyPDF** (RapidOCR)
- **GLM-OCR** (Zhipu AI)
- **LightonOCR**

## Metrics Tracked
- **LLM-as-a-Judge** (powered by `llama3-70b-8192` via Groq API): Contextual Text Accuracy, Table Structure, and Reading Order scoring.
- **NED** (Normalized Edit Distance): Programmatic Levenshtein distance for raw text accuracy.
- **Reading Order**: Programmatic Kendall's Tau on shared token positions.
---

## How to Run on Google Colab (Recommended)
Because some extraction pipelines (like GLM-OCR) require significant GPU resources, running this benchmark on Google Colab's T4 instances is highly recommended.

1. **Zip the Project**: Compress this entire folder into a `Flux_bench.zip` file.
2. **Open Colab**: Navigate to [Google Colab](https://colab.research.google.com/) and upload the provided `colab_benchmark.ipynb` file.
3. **Hardware Setup**: Go to `Runtime > Change runtime type` and select **T4 GPU**.
4. **Upload Data**: Open the Colab files pane (left sidebar) and drag-and-drop your `Flux_bench.zip` file into it.
5. **Run All**: Go to `Runtime > Run all`. 
   - *Note:* The notebook will securely prompt you to enter your **Groq API Key** for the LLM evaluation phase.
6. **Download Results**: Once finished, Colab will automatically download a `benchmark_results.zip` containing your final CSV reports!

---

## How to Run Locally

If you have a powerful local machine (or are testing lighter pipelines), you can run the benchmark locally.

### 1. System Requirements
You will need to install system-level dependencies for PDF processing and OCR:
- **Mac (Homebrew)**: `brew install ghostscript tesseract poppler`
- **Linux (Apt)**: `sudo apt-get install ghostscript tesseract-ocr poppler-utils libgl1`

### 2. Python Setup
Create a conda/virtual environment and install the required dependencies:
```bash
pip install docling ocrmypdf mineru pdf2image transformers accelerate rapidfuzz scipy pypdf tiktoken sentencepiece groq python-dotenv
```

### 3. Execution
Set your Groq API key in your environment (or inside a `.env` file at the root of the project):
```bash
export GROQ_API_KEY="your_api_key_here"
```

Run the extraction benchmark:
```bash
python benchmark.py
```
*(This processes all PDFs and generates `outputs/benchmark_results.csv`)*

Run the evaluation:
```bash
python evaluate.py
```
*(This compares the outputs to the `ground_truth/` folder and generates `outputs/evaluation_results.csv`)*

## Project Structure
- `benchmark.py`: Main orchestrator script for running the extraction pipelines.
- `evaluate.py`: The evaluation script utilizing Groq and deterministic metrics.
- `generate_colab.py`: Generates the `colab_benchmark.ipynb` file based on the latest dependencies.
- `pipelines/`: Contains the modular class definitions for each OCR pipeline.
- `ground_truth/`: Place your reference Markdown files here.
- `input_pdfs/`: (Create this if missing) Place your raw PDFs to be benchmarked here.
- `outputs/`: Auto-generated folder containing the extracted Markdown files and CSV results.
