import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
# from attention_sinks import AutoModelForCausalLM
from transformers import AutoModelForCausalLM
from typing import Dict
import copy
import json
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import sys
from transformers import default_data_collator
from tqdm import tqdm
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("..")
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
def InstructionDataset(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    IGNORE_TOKEN_ID = -100  # The default setting in CrossEntropyLoss
    examples = []
    prompts = []
    label = []
    for ann in sources:
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt
        if ann["output"].split(",") == 'Yes':
            label.append('Yes')
        else:
            label.append('No')
        examples.append(example)
        prompts.append(prompt)
    input_ids = tokenizer(examples , return_tensors="pt")
    labels_ids = tokenizer(label,padding="max_length",return_tensors="pt",max_length=tokenizer.model_max_length,truncation=True,).input_ids
    return {
        "input_ids": input_ids,
    }
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        # sources = [example["conversations"] for example in raw_data]
        sources = [example for example in raw_data]
        # data_dict = preprocess(sources, tokenizer)
        data_dict = InstructionDataset(sources,tokenizer)

        self.input_ids = data_dict["input_ids"]
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
        )
# model_id = "meta-llama/Llama-2-7b-hf"
# model_id = "mistralai/Mistral-7B-v0.1"
model_id = "llama2/Llama-2-7b-chat-hf-lora"
# Load the chosen model and corresponding tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # for efficiency:
    device_map="auto",
    # `attention_sinks`-specific arguments:
    # attention_sink_size=4,
    # attention_sink_window_size=252, # <- Low for the sake of faster generation
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id,model_max_length=1024)
tokenizer.pad_token_id = tokenizer.eos_token_id

test_json = json.load(open("/data/datasets/eva_2k.json", "r"))
sources = [example for example in test_json]
label = []
input = []
explain_labels = []
for example in sources:
    prompt = PROMPT_DICT["prompt_input"].format_map(example)
    input.append(prompt)
    explain_labels.append(example["output"][4:])
    if example["output"].split(",")[0] == 'Yes':
        # print(example["output"][4:])
        label.append('Yes')
    else:
        label.append('No')

# with open("explanation_label.txt",'w') as fp2:
#     for i in explain_labels:
#         fp2.write(i+'\n')


# data_dict = SupervisedDataset(test_json,tokenizer=tokenizer)
# eval_dataloader = torch.utils.data.DataLoader(
#             data_dict,
#             batch_size=1,
#             num_workers=8,
#             pin_memory=True,
#             drop_last=True,
#             collate_fn=default_data_collator,
# )
TP=0
TN=0
FP=0
FN=0
def Calculate_f1(TP,TN,FP,FN)-> Dict:
    precision = TP/(TP+FP+0.00001)
    recall = TP/(TP+FN+0.00001)
    F1_score = 2*(precision*recall)/(precision+recall+0.00001)
    return {
        "precision": precision,
        "recall": recall,
        "F1_score": F1_score,
        "ACC": ((TP+TN)/(TP+TN+FP+FN+0.00001))
    }
dic = []
explanations = []
label_explanations = []
count = 0
count_Y = 0
count_N = 0
count2 = 0
for step, batch in enumerate(tqdm(input,colour="green", desc="evaluating Epoch")):
    with torch.no_grad():
        ids = tokenizer.encode(batch, return_tensors="pt").to(model.device)
        # Forward pass and compute loss
        outputs = model.generate(
        ids,
        generation_config=GenerationConfig(
            # use_cache=True is required, the rest can be changed up.
            use_cache=True,
            max_new_tokens=20,
            penalty_alpha=0.6,
            top_k=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_explanation = output_text.split("### Response:")[-1]
        if len(output_explanation) < 3: continue
        # if output_explanation[0:3] == 'Yes':
        if output_explanation[0] == 'Y' or "is one contradiction" in output_explanation or "is a contradiction" in output_explanation or "are several contradictions" in output_explanation or "are contradictions" in output_explanation or "a few inconsistencies" in output_explanation or "are 3 contradiction" in output_explanation or "are two contradictions" in output_explanation:
            count_Y += 1
            if label[step] =='Yes':
                TP+=1
            else :FP+=1
        # elif output_explanation[0] == 'N':
        elif output_explanation[0] == 'N' or "no contradiction" in output_explanation or "not contradictory" in output_explanation or "does not contain a contradiction" in output_explanation or "any contradictions" in output_explanation:
            count_N += 1
            if label[step] =='No':
                TN+=1
            else :FN+=1
        # elif "a:" in output_explanation and "b:" in output_explanation:
        #     count2+=1
            # if label[step] == 'Yes':
            #     FP += 1
            # if label[step] == 'No':
            #     FN += 1
        output = Calculate_f1(TP,TN,FP,FN)
        pre = output["precision"]
        recal = output["recall"]
        f1_score = output["F1_score"]
        acc_score = output["ACC"]
        dic.append([label[step],output_text.split("### Response:")[-1][:3]])
        print(label[step],output_text.split("### Response:")[-1])
        # print(output["precision"],output["recall"],output["F1_score"],output["ACC"])
        print(f"precision:{pre}, Recall:{recal}, F1_score:{f1_score}, ACC:{acc_score}")
        print("count_Y is {}".format(count_Y))
        print("count_N is {}".format(count_N))
        print(f"all detection:{count_Y+count_N}")
        with open("eva_Y_and_N.txt","w") as fp:
            fp.write(f"precision:{pre}, Recall:{recal}, F1_score:{f1_score}, ACC:{acc_score}")
            fp.write('\n')
            fp.write(f"count_Y:{count_Y}, count_N:{count_N}, all_count:{count_Y+count_N}")
        # print(count2)


