import openai
import argparse
import json
import os
import statistics
from pathlib import Path
from collections import defaultdict

# === OpenAI Key ===
openai.api_key_path = os.path.expanduser("./openai/openai_key.txt")

# === System Prompt ===
SYSTEM_PROMPT = """
You are an expert in AI lifecycle development and software compliance analysis. 
Your task is to analyze the provided CODE and assess its compliance with a set of predefined Compliance Checks related to AI development best practices.

You will receive:
1. A CODE that needs to be analyzed.
2. A JSON file that contains a list of common tasks required in an AI lifecycle, based on their stage in the pipeline. Your objective is to evaluate whether each compliance requirement is fully implemented, partially implemented, or missing in the provided code. Each task includes:

Your objective is to evaluate to which extent each compliance requirement is implemented in the provided code, using a 6-point Likert scale (from 0 to 5). Each task includes:
    Task Name: The specific AI-related task.
    Compliance Check: The aspect of the task that must be evaluated.
    Compliance Description: The criteria that define proper implementation.

Evaluation Criteria:
For each Compliance Check, analyze the provided CODE and assess its level of implementation using the following 6-point Likert scale, by giving a Score of:
0 → The task IS NOT applicable to the context of the problem and code.
1 → The task IS applicable to the context of the code but is not implemented at all.
2 → The task IS applicable to the context of the code but is barely present and/or lacks key components.
3 → The task IS applicable to the context of the code and somewhat implemented, but is missing some required elements.
4 → The task IS applicable to the context of the code and mostly implemented, but is with minor deficiencies.
5 → The task IS applicable to the context of the code and is fully implemented with no need for improvement.

Additionally, provide:
1. Reasoning: A detailed explanation of why the task received the given score. If it is a 0 explain why it is not applicable to the context of the problem and code, and if the score is 1 or higher, explain the given score. Whenever is possible, if the score is 2 or higher, indicate where in the code the task is present.
2. Recommendation: If the task is not applicable (score = 0) explain why there is no need for that specific task to be implemented into the code. If the task is not fully implemented (score < 5), suggest specific improvements to achieve full compliance. If the task is fully implemented (score = 5), state that no further improvement is needed.
"""

# Output format for the AI to follow
OUTPUT_FORMAT_INSTRUCTION = """
Return the output as a JSON array where each element follows this structure:
[
    {
        "Task Name": the name of the task,
        "Compliance Check": the name of the compliance check,
        "Score": value from 0 to 5,
        "Reasoning": A detailed explanation of why the task received the given score,
        "Recommendation": Recommended improvement or reason why none is needed
    }
]
"""

def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in name)

# Load compliance checks
def load_compliance_checks(json_dir, stage=None):
    checks = {}
    if stage:
        filename = os.path.join(json_dir, f"{stage.lower().replace(' ', '_')}.json")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Stage file not found: {filename}")
        with open(filename, "r") as f:
            data = json.load(f)
        checks[stage] = data
    else:
        for file in os.listdir(json_dir):
            if file.endswith(".json"):
                stage_name = file.replace("_", " ").replace(".json", "").title()
                with open(os.path.join(json_dir, file), "r") as f:
                    data = json.load(f)
                checks[stage_name] = data
    return checks

def load_code(code_path):
    with open(code_path, 'r') as file:
        return file.read()

def build_user_prompt(code, compliance_dict):
    prompt = f"CODE TO ANALYZE:\n{code}\n\n"
    prompt += "COMPLIANCE CHECKS:\n"
    for stage, tasks in compliance_dict.items():
        prompt += f"\n--- {stage.upper()} ---\n"
        prompt += "\n".join([
            f"- Task: {t['Task Name']} | Check: {t['Compliance Check']} | Description: {t['Compliance description']}"
            for t in tasks
        ]) + "\n"
    prompt += "\nTASK:\nFor each Compliance Check, analyze the provided CODE and assess its level of implementation using the Likert scale. Return output in the exact JSON format below:\n"
    prompt += OUTPUT_FORMAT_INSTRUCTION
    return prompt

def analyze_single_file(code_path, compliance_dict, model="gpt-4"):
    code = load_code(code_path)
    user_prompt = build_user_prompt(code, compliance_dict)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    )
    return response['choices'][0]['message']['content']

def analyze_multiple_codes(code_inputs, json_dir, stage=None, model="gpt-4", output_root="./compliance_results"):
    code_files = []
    for path in code_inputs:
        if os.path.isdir(path):
            code_files += [str(p) for p in Path(path).rglob("*.py")]
            parent_folder_name = Path(path).name
        elif os.path.isfile(path):
            code_files.append(path)
            parent_folder_name = Path(path).parent.name
        else:
            print(f"Skipped invalid path: {path}")

    if not code_files:
        raise ValueError("No valid code files found to analyze.")

    if stage == "all":
        compliance_dict = load_compliance_checks(json_dir)
    elif stage:
        compliance_dict = load_compliance_checks(json_dir, stage)
    else:
        raise ValueError("Please specify a stage or use 'all' for all stages.")

    parent_output_dir = os.path.join(output_root, sanitize_filename(parent_folder_name))
    os.makedirs(parent_output_dir, exist_ok=True)

    # === Step 1: Individual results ===
    for code_path in code_files:
        code_name = sanitize_filename(Path(code_path).stem)
        print(f"\nAnalyzing: {code_path}")

        for stage_name, tasks in compliance_dict.items():
            print(f"   → Stage: {stage_name}")
            try:
                result = analyze_single_file(code_path, {stage_name: tasks}, model=model)
                parsed = json.loads(result)

                output_dir = os.path.join(parent_output_dir, code_name)
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{sanitize_filename(stage_name.lower())}.json")
                with open(out_path, "w") as f:
                    json.dump(parsed, f, indent=2)
                print(f"Saved: {out_path}")

            except json.JSONDecodeError:
                print("Output not valid JSON. Raw result:")
                print(result)
            except Exception as e:
                print(f"Error analyzing {code_path}: {e}")

    # === Step 2: Aggregation for final summary ===
    print("\nAggregating results for final summary...")

    for stage_name in compliance_dict.keys():
        task_scores = defaultdict(list)
        task_sources = defaultdict(list)
        stage_files = list(Path(parent_output_dir).rglob(f"{sanitize_filename(stage_name.lower())}.json"))

        for file_path in stage_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                for item in data:
                    key = (item["Task Name"], item["Compliance Check"])
                    if item["Score"] != 0:  # ignore zeros in mean
                        task_scores[key].append(item["Score"])
                        # relative path for traceability
                        rel_path = os.path.relpath(file_path, parent_output_dir)
                        task_sources[key].append(rel_path)

        aggregated = []
        for (task_name, check), scores in sorted(task_scores.items()):
            if scores:
                mean_score = round(statistics.mean(scores), 1)
                aggregated.append({
                    "Task Name": task_name,
                    "Compliance Check": check,
                    "Mean Score": mean_score,
                    "Count": len(scores),
                    "Sources": sorted(task_sources[(task_name, check)])
                })

        final_path = os.path.join(parent_output_dir, f"{sanitize_filename(stage_name.lower())}_final.json")
        with open(final_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"Saved final aggregated file: {final_path}")
        
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Stage AI Code Compliance Checker",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--code", nargs="+", required=True, help="Code file(s) or folder(s) to analyze")
    parser.add_argument("--json_dir", required=True, help="Directory with compliance JSONs")
    parser.add_argument("--stage", required=False, help="Analyze one stage, or 'all'")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    args = parser.parse_args()

    try:
        analyze_multiple_codes(args.code, args.json_dir, stage=args.stage, model=args.model)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    #Analyze entire folder
    #python data_retrieval_processes/code_analysis_UPDATED.py --code scripts_folder --json_dir tasks_check_jsons --stage "Data Ingestion and Data Versioning"

    #Analyze multiple files
    #python data_retrieval_processes/code_analysis_UPDATED.py --code scripts_folder/a_few_good.py file2.py --json_dir tasks_check_jsons --stage "Data Ingestion and Data Versioning"

    #Analyze specific stage
    #python data_retrieval_processes/code_analysis_UPDATED.py --code scripts_folder/a_few_good.py --json_dir tasks_check_jsons --stage "Data Ingestion and Data Versioning"  

    #Analyze all stages from a specific code
    #python data_retrieval_processes/code_analysis_UPDATED.py --code scripts_folder/a_few_good.py --json_dir tasks_check_jsons --stage all


    #python data_retrieval_processes/code_analysis_UPDATED.py --code scripts_folder/ --json_dir tasks_check_jsons --stage "all