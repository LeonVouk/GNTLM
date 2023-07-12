import json
import os

import fire
import bitsandbytes as bnb
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from utils.model_utils import set_different_steps_per_experiment, get_prompts_for_correct_experiment, \
    print_trainable_parameters
from utils.prompt_templates import generate_prompt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda:0'

OUTPUT_DIR = '/checkpoints'
EXPERIMENT = 'experiment_4'
MODEL_NAME = 'tiiuae/falcon-7b'

SAVE_MODEL_AS = 'GNTLM'


def load_4bit_model_and_tokenizer(model_name: str=MODEL_NAME) -> tuple:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map='auto',
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def generate_and_tokenize_prompt(data_point: dict, tokenizer) -> transformers.tokenization_utils_base.BatchEncoding:
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt,
                                      padding=True,
                                      truncation=True)
    return tokenized_full_prompt


def prepare_training_set(prompt_location: str='', experiment_name: str=EXPERIMENT) -> tuple:
    if experiment_name:
        prompt_location = f'/{get_prompts_for_correct_experiment()}'

    data = load_dataset('json', data_files=prompt_location)
    train_val = data["train"].train_test_split(
        test_size=100, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].map(generate_and_tokenize_prompt)
    )

    return train_data, val_data


def set_training_arguments(micro_batch_size: int=None,
                           batch_size: int=None,
                           train_steps: int=None,
                           learning_rate: int=None,
                           optimizer: str='paged_adamw_8bit',
                           experiment_name: str=EXPERIMENT) -> transformers.TrainingArguments:

    if experiment_name:
        micro_batch_size, batch_size, train_steps = set_different_steps_per_experiment(experiment_name)
        learning_rate = 3e-4
        optimizer = 'paged_adamw_8bit'

    grad_accumulation_steps = batch_size // micro_batch_size

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        save_total_limit=3,
        logging_steps=50,
        output_dir=OUTPUT_DIR,
        max_steps=train_steps,
        optim=optimizer,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        report_to='tensorboard'
    )

    return training_args


def main(micro_batch_size: int=None,
         batch_size: int=None,
         train_steps: int=None,
         learning_rate: int=None,
         optimizer: str='paged_adamw_8bit',
         prompt_location: str='',
         lora_config: str='',
         experiment_name: str=EXPERIMENT,
         save_model_as=SAVE_MODEL_AS) -> None:

    model, tokenizer = load_4bit_model_and_tokenizer()

    if not lora_config:
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=['query_key_value'],
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )
    else:
        config = lora_config


    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    if not experiment_name:
        train_data, val_data = prepare_training_set(prompt_location=prompt_location,
                                                    experiment_name=experiment_name)
        training_args = set_training_arguments(micro_batch_size=micro_batch_size,
                                               batch_size=batch_size,
                                               train_steps=train_steps,
                                               learning_rate=learning_rate,
                                               optimizer=optimizer,
                                               experiment_name=experiment_name)
    else:
        train_data, val_data = prepare_training_set(experiment_name=experiment_name)
        training_args = set_training_arguments(experiment_name=experiment_name)

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=data_collator
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(save_model_as)


if __name__ == '__main__':
    fire.Fire(main)