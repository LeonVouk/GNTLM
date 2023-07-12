import os

import bitsandbytes as bnb
import fire
import langchain
from langchain.llms import HuggingFacePipeline
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
import torch
import transformers
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

from utils.model_utils import get_local_peft_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = 'cuda:0'
EXPERIMENT = 'experiment_4'


class StopOnTokens(transformers.StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in kwargs['stop_token_ids']:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


def normalize_response(chat_chain: langchain.chains.ConversationChain, query: str) -> str:
    chat_chain.predict(input=query)

    chat_chain.memory.chat_memory.messages[-1].content = chat_chain.memory.chat_memory.messages[-1].content.split('\n\n')[0]
    chat_chain.memory.chat_memory.messages[-1].content = chat_chain.memory.chat_memory.messages[-1].content.strip()

    for stop_text in ['Human:', 'AI:']:
        chat_chain.memory.chat_memory.messages[-1].content = chat_chain.memory.chat_memory.messages[-1].content.removesuffix(stop_text)

    chat_chain.memory.chat_memory.messages[-1].content = chat_chain.memory.chat_memory.messages[-1].content.strip()

    return chat_chain.memory.chat_memory.messages[-1].content


def load_local_model_and_tokenizer(lora_config: str='', experiment_name: str=EXPERIMENT) -> tuple:
    GNTLM = get_local_peft_model(experiment_name=experiment_name)
    peft_model = f'/{GNTLM}'

    if not lora_config:
        bnb_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=['query_key_value'],
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )
    else:
        bnb_config = lora_config

    config = PeftConfig.from_pretrained(peft_model)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        load_in_4bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token_id = 1

    model = PeftModel.from_pretrained(model, peft_model)
    return model, tokenizer


def create_stopping_criterion(tokenizer) -> transformers.StoppingCriteriaList:
    stop_token_ids = [
        tokenizer.convert_tokens_to_ids(x) for x in [
            ['Human', ':'], ['AI', ':']
        ]
    ]
    stop_token_ids = [torch.LongTensor(x).to(DEVICE) for x in stop_token_ids]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens(**{"stop_token_ids": stop_token_ids})])
    return stopping_criteria


def generate_pipeline(model,
                  tokenizer,
                  stopping_criteria,
                  task: str='text-generation',
                  temperature: float=0.7,
                  max_new_tokens: int=128) -> transformers.pipeline():

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task=task,
        stopping_criteria=stopping_criteria,
        temperature=temperature,
        top_p=0.7,
        top_k=0,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5
    )
    return generate_text


def langchain_config(pipeline, initial_prompt_template: str=''):

    prompt = langchain.PromptTemplate(
        input_variables=["instruction"],
        template="{instruction}"
    )

    llm = HuggingFacePipeline(pipeline=pipeline)
    llm_chain = langchain.LLMChain(llm=llm, prompt=prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="history",
        k=5,
        return_only_outputs=True
    )

    chat = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )

    if not initial_prompt_template:
        chat.prompt.template = \
            """The following is a conversation between a human and an guidance counseling AI. The AI is talkative and provides lots of specific details from its context and focuses on answering the question posed by the human. If the AI does not know the answer to a question, it truthfully says it does not know.
    
            Current conversation:
            {history}
            Human: {input}
            AI:"""
    else:
        chat.prompt.template = initial_prompt_template

    return chat


def main(text: str='',
         temperature: float=0.7,
         max_new_tokens: int=128,
         initial_prompt_template: str='',
         lora_config: str='',
         task: str='text-generation',
         experiment_name: str=EXPERIMENT):

    model, tokenizer = load_local_model_and_tokenizer(lora_config=lora_config, experiment_name=experiment_name)

    stopping_criteria = create_stopping_criterion(tokenizer)

    pipeline = generate_pipeline(model=model,
                                 tokenizer=tokenizer,
                                 stopping_criteria=stopping_criteria,
                                 task=task,
                                 temperature=temperature,
                                 max_new_tokens=max_new_tokens)

    # TODO fix implemetation to not overwrite memory for every use
    # TODO add gradio support
    chat = langchain_config(pipeline=pipeline, initial_prompt_template=initial_prompt_template)

    return normalize_response(chat,
                              text)


if __name__ == '__main__':
    fire.Fire(main)