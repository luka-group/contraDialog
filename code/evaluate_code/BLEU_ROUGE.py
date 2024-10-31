import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# def calculate_bleu(reference, candidate):
#     reference = [reference.split()]  # NLTK expects a list of reference tokens
#     candidate = candidate.split()
#     return sentence_bleu(reference, candidate)

def calculate_bleu(reference, candidate):
    reference = [reference.split()]  # NLTK expects a list of reference tokens
    candidate = candidate.split()
    smoothing_function = SmoothingFunction().method1  # Choose smoothing method
    weights = (0.25, 0.25, 0.25, 0.25)  # BLEU-4 weights for 4-grams
    return sentence_bleu(reference, candidate, weights=weights, smoothing_function=smoothing_function)


def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]['rouge-l']['f']

# 读取JSON文件
with open('label_explanation/finetuned_v2_llama_label_explanation.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 初始化计数器
total_bleu = 0
total_rouge = 0

# 遍历每个字典
for entry in data:
    label = entry['label']
    explanation = entry['explanation']

    # 计算BLEU-4值
    bleu_score_label = calculate_bleu(label, explanation)
    total_bleu += bleu_score_label

    # 计算ROUGE-L值
    rouge_score_label = calculate_rouge(label, explanation)
    total_rouge += rouge_score_label

# 计算平均值
avg_bleu = total_bleu / len(data)
avg_rouge = total_rouge / len(data)

# 打印结果
print(f'Average BLEU-4: {avg_bleu}')
print(f'Average ROUGE-L: {avg_rouge}')
