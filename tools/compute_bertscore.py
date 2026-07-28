# compute_bertscore.py

import json
import sys
import jieba
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def compute_metrics(preds, refs):
    assert len(preds) == len(refs)

    rougeL_scores = []

    scorer = rouge_scorer.RougeScorer(
        ['rougeL'],
        use_stemmer=False
    )

    for pred, ref in zip(preds, refs):
        pred_seg = " ".join(jieba.cut(pred))
        ref_seg = " ".join(jieba.cut(ref))

        rouge_scores = scorer.score(ref_seg, pred_seg)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

    # ✅ BERTScore（逐样本）
    P, R, F1 = bert_score(preds, refs, lang="zh", device="cpu")

    return {
        "ROUGE-L": rougeL_scores,        # ✅ list
        "BERTScore-F1": F1.tolist(),     # ✅ list
    }


if __name__ == "__main__":
    """
    输入格式：
    python compute_bertscore.py input.json

    input.json:
    {
        "preds": [...],
        "refs": [...]
    }
    """

    input_file = sys.argv[1]

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data["preds"]
    refs = data["refs"]

    result = compute_metrics(preds, refs)

    # ✅ 关键：打印 JSON（GRPO 里读取）
    print(json.dumps(result, ensure_ascii=False))