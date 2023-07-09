def generate_prompt_data(data: list) -> list:
    dataset = [
        {
            "instruction": "Based on the information given in the input, predict this person's future occupation.",
            "input": d_point["early_life"],
            "output": ', '.join(d_point["occupation"]).strip()
        }
        for d_point in data
    ]
    return dataset


def generate_prompt_annotated_data(data: list) -> list:
    dataset = []
    for d in data:
      for k, d_point in d.items():
        evidence = ' '.join([(str(n) + ') ' + e) for n, e in enumerate(d_point['evidence'])])
        dataset.append({
              "instruction": "Based on the information given in the input, predict this person's future occupation.",
              "input": d_point["early_life"],
              "output": f"""Based on the following pieces of evidence: {evidence}
                        This person should become: {', '.join(d_point['occupation']).strip()}
                        """
          })
    return dataset


def generate_prompt_generated_data(data: list) -> list:
    dataset = []
    for d in data:
      for k, d_point in d.items():
        dataset.append({
              "instruction": "Based on the information given in the input, predict this person's future occupation.",
              "input": d_point["early_life"],
              "output": f"""{d_point['gpt_completion']}
                        This person should become: {', '.join(d_point['occupation']).strip()}
                        """
          })
    return dataset


def generate_prompt(data_point: dict) -> str:
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
