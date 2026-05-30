import json

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Flux Bench - Google Colab Setup\n",
        "\n",
        "**Important:** Go to `Runtime > Change runtime type` and ensure Hardware Accelerator is set to **T4 GPU**."
      ],
      "metadata": {"id": "markdown-intro"}
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 1. System Requirements\n",
        "Install Ghostscript, Tesseract OCR, and other system dependencies required by OCRmyPDF and Docling."
      ],
      "metadata": {"id": "markdown-sys"}
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {"id": "cell-sys"},
      "outputs": [],
      "source": [
        "!apt-get update\n",
        "!apt-get install -y ghostscript tesseract-ocr poppler-utils libgl1"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 2. Python Dependencies\n",
        "Install all pipeline libraries including MinerU, Docling, OCRmyPDF, GLM-OCR, and evaluation metrics (rapidfuzz, scipy, groq)."
      ],
      "metadata": {"id": "markdown-pip"}
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {"id": "cell-pip"},
      "outputs": [],
      "source": [
        "!pip install docling ocrmypdf mineru pdf2image transformers accelerate rapidfuzz scipy pypdf tiktoken sentencepiece groq"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 3. Upload Project Files\n",
        "Zip your entire `Flux bench` folder on your Mac, and upload `Flux_bench.zip` to the Colab files pane (on the left).\n",
        "Run this cell to extract it."
      ],
      "metadata": {"id": "markdown-upload"}
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {"id": "cell-upload"},
      "outputs": [],
      "source": [
        "import os\n",
        "import zipfile\n",
        "\n",
        "# Unzip if the file exists\n",
        "if os.path.exists('Flux_bench.zip'):\n",
        "    with zipfile.ZipFile('Flux_bench.zip', 'r') as zip_ref:\n",
        "        zip_ref.extractall('flux_bench')\n",
        "    print(\"Unzipped to 'flux_bench/'\")\n",
        "else:\n",
        "    print(\"Flux_bench.zip not found! Please upload it to the main directory.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 4. Setup Groq API Key (For LLM Evaluation)\n",
        "Enter your Groq API key when prompted to run the intelligent evaluation."
      ],
      "metadata": {"id": "markdown-groq"}
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {"id": "cell-groq"},
      "outputs": [],
      "source": [
        "import os\n",
        "import getpass\n",
        "os.environ['GROQ_API_KEY'] = getpass.getpass('Enter your Groq API key: ')"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 5. Run Benchmark & Evaluator\n",
        "This will execute your sequential benchmark across all pipelines and run the evaluation script."
      ],
      "metadata": {"id": "markdown-run"}
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {"id": "cell-run"},
      "outputs": [],
      "source": [
        "import os\n",
        "\n",
        "# Change directory to the extracted folder to run the benchmark\n",
        "os.chdir('/content/flux_bench')\n",
        "if os.path.exists('Flux bench'):\n",
        "    os.chdir('Flux bench')\n",
        "\n",
        "!python benchmark.py\n",
        "!python evaluate.py"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 5. Download Results\n",
        "Zip the `outputs` directory back up so you can download the results CSVs."
      ],
      "metadata": {"id": "markdown-download"}
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {"id": "cell-download"},
      "outputs": [],
      "source": [
        "!zip -r /content/benchmark_results.zip outputs/\n",
        "\n",
        "from google.colab import files\n",
        "if os.path.exists('/content/benchmark_results.zip'):\n",
        "    files.download('/content/benchmark_results.zip')"
      ]
    }
  ]
}

with open("colab_benchmark.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated colab_benchmark.ipynb")
