from bert_score import score
from BARTScore.bart_score import BARTScorer
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
bart_scorer = BARTScorer(device='cuda:2', checkpoint='facebook/bart-large-cnn')
cands = []
refs = []
bert_f1_result = []
bart_results = []
count = 0

label_explanation_json = json.load(open("label_explanation/original_llama_v2_label_explanation.json","r"))
for i in label_explanation_json:
    cand = i['explanation']
    ref = i['label']
    cands.append(cand)
    refs.append(ref)
# with open("explanation_vicuna_finetune_result/explanation_label-eva_2k.txt","r") as f1:
#     for line1 in f1:
#         cand = line1
#         cands.append(cand)
# with open("explanation_vicuna_finetune_result/explanation_vicuna-7b-finetuned.txt","r") as f2:
#     for line2 in f2:
#         ref = line2
#         refs.append(ref)

# print(len(cands))
# print(len(refs))


for i in range(len(cands)):
    c = [cands[i]]
    r = [refs[i]]
    P, R, F1 = score(c, r, lang='en', verbose = True) #bertscore
    bart_result = bart_scorer.score(c,r,batch_size=4) #bartscore
    print(f"F1 score:{F1.mean():3f}")
    # print(F1.item())
    # print(type(F1.item())) float
    print(bart_result)
    # print(type(bart_result)) list

    bert_f1_result.append(F1.item())
    bart_results.append(bart_result)

    # print(bart_result)
    
tau = 0.1
with open("explanation_result/result_original_llama.txt","w") as fp:
    for bert, bart in zip(bert_f1_result, bart_results):
        f_Score = bert + tau*bart[0]
        fp.write(f"{bert:3f} {bart[0]:3f} {f_Score:3f}\n")