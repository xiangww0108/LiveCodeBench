import json
import argparse


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_spans(span):
    """
    Normalize span format into list of [start, end].
    Acceptable formats:
      - [7, 9]
      - [[7, 9], [12, 14]]
      - {"bug_span": [...]}
    """
    if isinstance(span, dict) and "bug_span" in span:
        span = span["bug_span"]

    # Single span [s,e]
    if isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span):
        return [span]

    # Multi span [[s,e], [s,e]]
    return span


def span_list_to_set(spans):
    """
    Convert a list of spans like [[s,e], [a,b]] into a set of line numbers.
    """
    lines = set()
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, list) and len(span) == 2:
                s, e = min(span), max(span)
                lines.update(range(s, e + 1))
    return lines


def compute_iou_and_recall(pred_spans, true_spans):
    """
    EXACT SAME METRIC as your original localizer version:
        IoU = |pred ∩ gt| / |pred ∪ gt|
        Recall = |pred ∩ gt| / |gt|
    """
    pred_lines = span_list_to_set(pred_spans)
    true_lines = span_list_to_set(true_spans)

    # Avoid division by zero
    if len(true_lines) == 0:
        return 0.0, 0.0

    intersection = len(pred_lines & true_lines)
    union = len(pred_lines | true_lines)

    iou = intersection / union if union > 0 else 0.0
    recall = intersection / len(true_lines)

    return iou, recall


def evaluate(golden_file, pred_file):
    golden = load_json(golden_file)
    preds = load_json(pred_file)

    # Build GT index: title → bug_span
    golden_map = {
        item["question_title"]: normalize_spans(item["bug_span"])
        for item in golden
    }

    total = 0
    ious = []
    recalls = []

    for p in preds:
        title = p["question_title"]
        if title not in golden_map:
            continue

        total += 1

        gt_spans = golden_map[title]
        pred_spans = normalize_spans(p["bug_span"])

        iou, recall = compute_iou_and_recall(pred_spans, gt_spans)
        ious.append(iou)
        recalls.append(recall)

    print("\n===== Localizer JSON Evaluation =====")
    print(f"Matched Samples: {total}")
    print(f"Mean IoU:    {sum(ious)/total:.4f}")
    print(f"Mean Recall: {sum(recalls)/total:.4f}")

    return {
        "total": total,
        "mean_iou": sum(ious)/total if total else 0,
        "mean_recall": sum(recalls)/total if total else 0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=str, default="data/Qwen3-TrainTest-data/multitask-step-by-step/test-localizer.json")
    parser.add_argument("--pred", type=str, default="data/Qwen3-TrainTest-data/multitask-step-by-step/finetuning_result_new.json")
    args = parser.parse_args()

    evaluate(args.golden, args.pred)
