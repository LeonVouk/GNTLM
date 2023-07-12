from peft import (
    PeftModel,
)


EXPERIMENT = 'experiment_4'


def print_trainable_parameters(model: PeftModel) -> None:
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_params} || trainable %: {100 * trainable_params / all_params}'
    )


def get_prompts_for_correct_experiment(experiment_name: str=EXPERIMENT) -> str:
    map_experiment_name_to_prompts = {
        'experiment_1': 'prompt_library/prompt_data_all.json',
        'experiment_2': 'prompt_library/prompt_data_f.json',
        'experiment_3': 'prompt_library/prompt_data_annotated.json',
        'experiment_4': 'prompt_library/prompt_data_generated.json'
    }
    try:
      collection_name = map_experiment_name_to_prompts[experiment_name]
    except KeyError:
      raise KeyError(f'Please choose a name '
                     f'in {list(map_experiment_name_to_prompts.keys())} or set experiment_name to ""')
    return f'prompt_library/{collection_name}'


def set_different_steps_per_experiment(experiment_name: str=EXPERIMENT) -> str:
    map_experiment_name_to_prompts = {
        'experiment_1': (1, 5, 1540),
        'experiment_2': (1, 4, 1200),
        'experiment_3': (1, 5, 845),
        'experiment_4': (1, 4, 768)
    }
    try:
      return map_experiment_name_to_prompts[experiment_name]
    except KeyError:
      raise KeyError(f'Please choose a name '
                     f'in {list(map_experiment_name_to_prompts.keys())} or set experiment_name to ""')


def get_local_peft_model(experiment_name: str=EXPERIMENT, initial: bool=True) -> str:
    map_experiment_name_to_model = {
          'experiment_1': 'models/experiment_1_model',
          'experiment_2': 'models/experiment_2_model',
          'experiment_3': 'models/experiment_3_model',
          'experiment_4': 'models/experiment_4_model'
    }
    try:
      return map_experiment_name_to_model[experiment_name]
    except KeyError:
      if initial:
        raise KeyError(f'Please choose a name in {list(map_experiment_name_to_model.keys())}. '
                       f'If you want to load a different model, please provide the model directory and set initial=False')
      else:
        return experiment_name