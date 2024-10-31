from typing import Dict
import copy
import json
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import sys
sys.path.append("..")
from conversation import SeparatorStyle
from model.model_adapter import get_conversation_template
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        padding="max_length",
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
def InstructionDataset(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    IGNORE_TOKEN_ID = -100  # The default setting in CrossEntropyLoss
    examples = []
    prompts = []
    for ann in sources:
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        examples.append(example)
        prompts.append(prompt)
    input_ids = tokenizer(
        examples,
        padding="max_length",
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    labels = copy.deepcopy(input_ids)
    for prompt,example, target in zip(prompts,input_ids, labels):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        prompt = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.int64
        )
        cur_len = len(prompt)
        target[:cur_len] = IGNORE_TOKEN_ID
        cur_len = int(example.ne(tokenizer.pad_token_id).sum())
        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask":input_ids.ne(tokenizer.pad_token_id),
    }
if __name__ == "__main__":
    train_json = json.load(open('../data/output.json', "r"))
    sources = [example for example in train_json[0:1]]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '../Llama-2-7b-chat-hf',
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_dict = InstructionDataset(sources,tokenizer)
    print(data_dict['attention_mask'])