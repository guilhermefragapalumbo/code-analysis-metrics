"""
Write the exact prompts used by code_analysis.py to prompts/.

code_analysis.py is imported with the openai package replaced by a stub, so
no API call is made and no key is needed. The user prompt is built with the
script's own build_user_prompt() for one example: the Model Training and
Tuning stage of the Food Allergy main.py before the interventions.

Usage:
    python export_prompts.py
"""
import importlib.util
import sys
import types
from pathlib import Path

def write_text(path, text):
    """Write UTF-8 text with LF line endings on every platform."""
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


HERE = Path(__file__).resolve().parent
EXAMPLE_CODE = HERE.parent.parent / 'ML Projects' / 'FOOD_ALLERGY_ANALYSIS_BEFORE' / 'Projeto' / 'Algoritms' / 'main.py'
EXAMPLE_STAGE = 'model_training_and_tuning'

sys.modules['openai'] = types.ModuleType('openai')  # the script only references it at call time
spec = importlib.util.spec_from_file_location('code_analysis', HERE / 'code_analysis.py')
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

out = HERE / 'prompts'
out.mkdir(exist_ok=True)
write_text(out / 'system_prompt.txt', ca.SYSTEM_PROMPT)
write_text(out / 'output_format_instruction.txt', ca.OUTPUT_FORMAT_INSTRUCTION)

# When run with --stage all, the script loads every stage file and sends one stage per call,
# keyed by the file name in title case, e.g. "Model Training And Tuning".
checks = ca.load_compliance_checks(str(HERE / 'compliance_checks'))
stage_name = EXAMPLE_STAGE.replace('_', ' ').title()
# load_code() opens files with the platform default encoding. The evaluations ran on Windows, where that
# is cp1252, so the UTF-8 source reached the model with non-ASCII characters mis-decoded; reading with
# cp1252 here reproduces the prompt as sent, on any platform.
code = EXAMPLE_CODE.read_bytes().decode('cp1252').replace('\r\n', '\n')
user_prompt = ca.build_user_prompt(code, {stage_name: checks[stage_name]})
write_text(out / 'example_user_prompt_food_allergy_before_main_training.txt', user_prompt)
print('wrote', *sorted(p.name for p in out.iterdir()), sep='\n  ')
