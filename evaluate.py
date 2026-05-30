import os
import re
import csv
import json
import math
from pathlib import Path
from rapidfuzz.distance import Levenshtein
import scipy.stats
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Programmatic Reading Order Score
# ==========================================
def get_tokens(text):
    return [t for t in re.split(r'\W+', text.lower()) if t]

def reading_order_score(hyp, ref):
    hyp_tokens = get_tokens(hyp)
    ref_tokens = get_tokens(ref)
    
    ref_map = {}
    for i, t in enumerate(ref_tokens):
        if t not in ref_map:
            ref_map[t] = i
            
    hyp_mapped = []
    seen = set()
    for t in hyp_tokens:
        if t in ref_map and t not in seen:
            seen.add(t)
            hyp_mapped.append(ref_map[t])
            
    if len(hyp_mapped) < 2:
        return 0.0
        
    x = list(range(len(hyp_mapped)))
    tau, p_value = scipy.stats.kendalltau(x, hyp_mapped)
    return tau if not math.isnan(tau) else 0.0

# ==========================================
# 2. Programmatic NED
# ==========================================
def calc_ned(hyp, ref):
    if not hyp and not ref:
        return 1.0
    dist = Levenshtein.distance(hyp, ref)
    max_len = max(len(hyp), len(ref))
    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)

# ==========================================
# 3. LLM Evaluator (Groq)
# ==========================================
def get_llm_evaluation(client, hyp_text, ref_text):
    prompt = f"""You are an expert OCR and document extraction evaluation system.
Compare the GROUND TRUTH markdown with the EXTRACTED markdown below.

GROUND TRUTH:
{ref_text}

EXTRACTED:
{hyp_text}

Evaluate the extraction on a scale of 0.0 to 1.0 for three dimensions:
1. text_accuracy_score: How accurate is the raw text content?
2. table_structure_score: How well are tables extracted? (Note: Both HTML and Markdown tables are perfectly acceptable as long as the data and structure are correct. If no tables exist in both, score 1.0).
3. reading_order_score: Is the layout and reading order preserved logically?

Output ONLY a raw JSON object with the following exact keys, with no markdown formatting or backticks:
{{
  "text_accuracy_score": 0.0,
  "table_structure_score": 0.0,
  "reading_order_score": 0.0,
  "reasoning": "Brief explanation of the scores."
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return data
    except Exception as e:
        print(f"  [LLM Error] {e}")
        return None

# ==========================================
# Main Evaluation Loop
# ==========================================
def main():
    gt_dir = Path("ground_truth")
    out_dir = Path("outputs")
    
    if not gt_dir.exists():
        print("[evaluate] ground_truth folder not found!")
        return
        
    if not out_dir.exists():
        print("[evaluate] outputs folder not found! Benchmark likely crashed before generating any outputs.")
        return
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[evaluate] Warning: GROQ_API_KEY not found in environment. LLM evaluation will be skipped.")
        client = None
    else:
        print("[evaluate] Groq API Key found. Enabling LLM Evaluation.")
        client = Groq(api_key=api_key)
        
    results = []
    
    # Iterate through each pipeline folder
    for pipeline_dir in out_dir.iterdir():
        if not pipeline_dir.is_dir() or pipeline_dir.name.startswith("."):
            continue
            
        pipeline = pipeline_dir.name
        
        for hyp_file in pipeline_dir.rglob("*.md"):
            doc_name = hyp_file.name
            gt_file = gt_dir / doc_name
            
            if not gt_file.exists():
                print(f"[evaluate] Missing ground truth for {doc_name}")
                continue
                
            print(f"[evaluate] Scoring {pipeline}/{doc_name}...")
            try:
                with open(hyp_file, "r", encoding="utf-8") as f:
                    hyp_text = f.read()
                with open(gt_file, "r", encoding="utf-8") as f:
                    ref_text = f.read()
                    
                ned = calc_ned(hyp_text, ref_text)
                ro_score = reading_order_score(hyp_text, ref_text)
                
                row = {
                    "pipeline": pipeline,
                    "pdf": doc_name.replace(".md", ".pdf"),
                    "Programmatic_NED": round(ned, 4),
                    "Programmatic_ReadingOrder": round(ro_score, 4),
                    "LLM_Text_Score": "",
                    "LLM_Table_Score": "",
                    "LLM_ReadingOrder_Score": "",
                    "LLM_Reasoning": ""
                }
                
                if client:
                    llm_data = get_llm_evaluation(client, hyp_text, ref_text)
                    if llm_data:
                        row["LLM_Text_Score"] = llm_data.get("text_accuracy_score", "")
                        row["LLM_Table_Score"] = llm_data.get("table_structure_score", "")
                        row["LLM_ReadingOrder_Score"] = llm_data.get("reading_order_score", "")
                        row["LLM_Reasoning"] = llm_data.get("reasoning", "")
                        
                results.append(row)
            except Exception as e:
                print(f"[evaluate] Error evaluating {doc_name} for {pipeline}: {e}")
                
    if not results:
        print("[evaluate] No results to save.")
        return
        
    csv_path = out_dir / "evaluation_results.csv"
    columns = [
        "pipeline", "pdf", "Programmatic_NED", "Programmatic_ReadingOrder", 
        "LLM_Text_Score", "LLM_Table_Score", "LLM_ReadingOrder_Score", "LLM_Reasoning"
    ]
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"[evaluate] Evaluation complete. Saved to {csv_path}")

if __name__ == "__main__":
    main()
