import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score


# =====================================================
# Utility: Load JSONL
# =====================================================
def load_any(path):
    text = open(path, "r").read().strip()

    # Case 1: JSON list
    if text.startswith("["):
        return json.loads(text)

    # Case 2: JSONL
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items



# =====================================================
# Span metrics
# =====================================================
def span_to_set(span_list):
    res = set()
    for sp in span_list:
        
        # 🚨 修正逻辑开始 🚨
        if len(sp) == 1:
            # 如果 Bug Span 只有一个元素，例如 [5]，则将其视为 [5, 5]
            start_line = sp[0]
            s, e = start_line, start_line
        elif len(sp) == 2:
            # 标准格式 [s, e]
            s, e = sp
        else:
            # 忽略其他不合法的格式，例如 [5, 9, 10]
            # print(f"Warning: Skipping illegal span format: {sp}") # 可选：用于调试
            continue
        # 🚨 修正逻辑结束 🚨
        
        for i in range(s, e + 1):
            res.add(i)
    return res


def span_iou(pred, gold):
    P = span_to_set(pred)
    G = span_to_set(gold)
    if not P and not G:
        return 1.0
    if not P or not G:
        return 0.0
    return len(P & G) / len(P | G)


def span_recall(pred, gold):
    P = span_to_set(pred)
    G = span_to_set(gold)
    if not G:
        return 1.0
    if not P:
        return 0.0
    return len(P & G) / len(G)


# =====================================================
# NLP metrics: summary / planner
# =====================================================
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rougeL(a, b):
    return rouge.score(a, b)["rougeL"].fmeasure


def bleu(a, b):
    try:
        return sentence_bleu([b.split()], a.split(), smoothing_function=smooth)
    except:
        return 0.0


# =====================================================
# Main Evaluation
# =====================================================
def evaluate(gold_path, pred_path):
    gold = load_any(gold_path)
    pred = load_any(pred_path)


    gold_map = {x["question_title"]: x for x in gold}

    # Results
    span_ious, span_recalls = [], []
    summary_rouge, summary_bleu, summary_bert = [], [], []
    planner_rouge, planner_bleu, planner_bert = [], [], []

    for p in tqdm(pred, disable=True):
        title = p["question_title"]
        if title not in gold_map:
            continue
        g = gold_map[title]

        # 1) ---------------- Span ----------------
        span_pred = p.get("bug_span", [])
        span_gold = g.get("bug_span", [])
        span_ious.append(span_iou(span_pred, span_gold))
        span_recalls.append(span_recall(span_pred, span_gold))

        # 2) ---------------- Summary ----------------
        summ_pred = p.get("bug_summary", "").strip()
        summ_gold = g.get("bug_summary", "").strip()
        summary_rouge.append(rougeL(summ_pred, summ_gold))
        summary_bleu.append(bleu(summ_pred, summ_gold))

        # BERTScore
        try:
            _, _, F = bert_score([summ_pred], [summ_gold], lang="en", rescale_with_baseline=True)
            summary_bert.append(F.item())
        except:
            summary_bert.append(0)

        # 3) ---------------- Planner ----------------
        raw_plan = p.get("planner_text", "")
        if isinstance(raw_plan, list):
            # 如果是列表，安全地取出第一个元素作为字符串
            raw_plan = str(raw_plan[0]) if raw_plan else ""
        else:
            # 确保任何非列表都被视为字符串
            raw_plan = str(raw_plan) 
        plan_pred = raw_plan.strip()
        plan_gold = g.get("planner_text", "").strip()
        planner_rouge.append(rougeL(plan_pred, plan_gold))
        planner_bleu.append(bleu(plan_pred, plan_gold))

        try:
            _, _, Fp = bert_score([plan_pred], [plan_gold], lang="en", rescale_with_baseline=True)
            planner_bert.append(Fp.item())
        except:
            planner_bert.append(0)

    # ================================
    # Summary
    # ================================
    summary = {
        "Localizer_IoU": sum(span_ious) / len(span_ious),
        "Localizer_Recall": sum(span_recalls) / len(span_recalls),

        "Summary_RougeL": sum(summary_rouge) / len(summary_rouge),
        "Summary_BLEU": sum(summary_bleu) / len(summary_bleu),
        "Summary_BERTScore": sum(summary_bert) / len(summary_bert),

        "Planner_RougeL": sum(planner_rouge) / len(planner_rouge),
        "Planner_BLEU": sum(planner_bleu) / len(planner_bleu),
        "Planner_BERTScore": sum(planner_bert) / len(planner_bert),
    }

    print("\n===== Intrinsic Evaluation =====")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    evaluate(
        gold_path="/home/ubuntu/finetune_e2e_new/data/test_intrinsic.json",
        pred_path="/home/ubuntu/finetune_e2e_new/data/preds_intrinsic.json"
    )
