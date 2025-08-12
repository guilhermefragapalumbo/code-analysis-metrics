import os
import json

input_folder = "compliance_check_distribution/compliance_check_distribution_reasoning_json"
output_folder = "compliance_check_distribution/compliance_check_distribution_reasoning_markdown"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".json", ".md"))
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Start markdown content with table header
        markdown_lines = [
            "| Task Name | Compliance Check | Reasoning |",
            "|-----------|------------------|-----------|"
        ]
        
        # We expect data to be a list or dict containing entries with these keys
        # Adapt as needed if your JSON structure differs
        # For example, if data is a list of dicts:
        if isinstance(data, dict):
            entries = data.values()  # if keys are not important, get values
        elif isinstance(data, list):
            entries = data
        else:
            entries = []
        
        for entry in entries:
            # Attempt to extract relevant fields with fallback empty strings if missing
            task_name = entry.get("Task Name") or entry.get("task_name") or entry.get("task") or ""
            compliance_check = entry.get("Compliance Check") or entry.get("compliance_check") or ""
            reasoning = entry.get("Reasoning") or entry.get("reasoning") or ""
            
            # Escape pipe characters '|' in content to avoid breaking markdown table
            task_name = task_name.replace("|", "\\|")
            compliance_check = compliance_check.replace("|", "\\|")
            reasoning = reasoning.replace("|", "\\|")
            
            markdown_lines.append(f"| {task_name} | {compliance_check} | {reasoning} |")
        
        # Write markdown content to file
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(markdown_lines))

print("Markdown files created successfully.")
