import json
import random

from utils import prompt_templates


if __name__ == '__main__':

    with open('data/person_occupation_match_f_multiple.json', 'r') as json_file:
        data = json.load(json_file)
    prompt_data = prompt_templates.generate_prompt_data(data)
    decently_sized = [p for p in prompt_data if len(p['input'].split()) < 500]
    prompt_sample = random.sample(decently_sized, 1500)
    prompt_sample_most = random.sample(decently_sized, 8000)
    with open("prompt_library/prompt_data_f.json", "w") as f:
        json.dump(prompt_sample, f, indent=4)
    with open("prompt_library/prompt_data_all.json", "w") as f:
        json.dump(prompt_sample_most, f, indent=4)

    with open('data/annotated_data_aggr.json', 'r') as json_file:
        data = json.load(json_file)
    prompt_data = prompt_templates.generate_prompt_annotated_data(data)
    decently_sized = [p for p in prompt_data if len(p['input'].split()) < 500]
    with open("prompt_library/prompt_data_annotated.json", "w") as f:
        json.dump(decently_sized, f, indent=4)

    with open('data/person_occupation_gpt3_5_completions.json', 'r') as json_file:
        data = json.load(json_file)
    prompt_data = prompt_templates.generate_prompt_generated_data([{k: v} for k, v in data.items()])
    decently_sized = [p for p in prompt_data if len(p['input'].split()) < 500]
    with open("prompt_library/prompt_data_generated.json", "w") as f:
        json.dump(decently_sized, f, indent=4)
