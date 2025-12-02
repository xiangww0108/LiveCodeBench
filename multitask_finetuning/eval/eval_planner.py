import json
from collections import defaultdict
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="You should probably TRAIN")
warnings.filterwarnings("ignore", message="resume_download")


# ---------------------------
# Load JSON
# ---------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------
# Bug span exact + overlap metrics
# ---------------------------
def bug_span_exact(pred, gold):
    return pred == gold


def bug_span_overlap(pred, gold):
    if not pred or not gold:
        return False
    for ps in pred:
        for gs in gold:
            # if ranges overlap
            if not (ps[1] < gs[0] or ps[0] > gs[1]):
                return True
    return False


def safe_bleu(pred, gold):
    try:
        return sentence_bleu([gold.split()], pred.split(),
                             smoothing_function=SmoothingFunction().method1)
    except:
        return 0.0


# ============================================================
#                    MAIN EVALUATION
# ============================================================
def main():
    pred_file = "data/Qwen3-TrainTest-data/multitask-step-by-step/finetuning_result_new.json"
    gold_file = "data/Qwen3-TrainTest-data/multitask-step-by-step/test-planner.json"
    
    pred_data = load_json(pred_file)
    gold_data = load_json(gold_file)

    # Map gold items by question_title
    gold_map = {g["question_title"]: g for g in gold_data}

    rouge_scorer_fn = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    results = []
    # localizer_exact_cnt = 0
    # localizer_overlap_cnt = 0

    planner_bert_list = []
    planner_rougeL_list = []
    planner_bleu_list = []

    # total_localizer = 0
    total_planner = 0

    for pred in pred_data:

        title = pred["question_title"]

        if title not in gold_map:
            print("WARNING: title not found in gold:", title)
            continue

        gold = gold_map[title]

        # # =======================================
        # #             LOCALIZER
        # # =======================================
        # gold_span = gold.get("label_bug_span", None)
        # pred_span = pred.get("bug_span", None)

        # localizer_exact = None
        # localizer_overlap = None

        # if gold_span is not None and isinstance(gold_span, list):
        #     total_localizer += 1
        #     localizer_exact = bug_span_exact(pred_span, gold_span)
        #     localizer_overlap = bug_span_overlap(pred_span, gold_span)

        #     if localizer_exact:
        #         localizer_exact_cnt += 1
        #     if localizer_overlap:
        #         localizer_overlap_cnt += 1

        # =======================================
        #              PLANNER
        # =======================================
        gold_plan = gold.get("planner_text", None)
        
        pred_plan = pred.get("planner_output", "")

        bert_f1 = None
        rougeL = None
        bleu = None

        if gold_plan:
            total_planner += 1

            # BERTScore
            try:
                P, R, F1 = bert_score([pred_plan], [gold_plan], lang="en", rescale_with_baseline=True)
                bert_f1 = F1.item()
                planner_bert_list.append(bert_f1)
            except:
                bert_f1 = 0.0

            # ROUGE-L
            try:
                score = rouge_scorer_fn.score(pred_plan, gold_plan)
                rougeL = score["rougeL"].fmeasure
                planner_rougeL_list.append(rougeL)
            except:
                rougeL = 0.0

            # BLEU
            bleu = safe_bleu(pred_plan, gold_plan)
            planner_bleu_list.append(bleu)

        results.append({
            "question_title": title,
            # "bug_span_pred": pred_span,
            # "bug_span_gold": gold_span,
            # "localizer_exact": localizer_exact,
            # "localizer_overlap": localizer_overlap,
            "planner_bert": bert_f1,
            "planner_rougeL": rougeL,
            "planner_bleu": bleu
        })

    # ==========================
    #   SUMMARY METRICS
    # ==========================
    summary = {
        "total_examples": len(pred_data),
        # "localizer_total": total_localizer,
        # "localizer_exact_acc": localizer_exact_cnt / total_localizer if total_localizer else None,
        # "localizer_overlap_acc": localizer_overlap_cnt / total_localizer if total_localizer else None,
        "planner_total": total_planner,
        "planner_bert_avg": sum(planner_bert_list) / len(planner_bert_list) if planner_bert_list else None,
        "planner_rougeL_avg": sum(planner_rougeL_list) / len(planner_rougeL_list) if planner_rougeL_list else None,
        "planner_bleu_avg": sum(planner_bleu_list) / len(planner_bleu_list) if planner_bleu_list else None
    }

    output = {
        "summary": summary,
        "details": results
    }

    with open("data/output/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("=== Evaluation Complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
