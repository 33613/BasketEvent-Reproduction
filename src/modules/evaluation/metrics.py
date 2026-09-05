"""两篇篮球论文的核心指标；所有分数取值为 0～1。

序列指标对应 BasketballBench 附录 C.1.3。人物—事件集合指标和时间
Hit 对应 BasketEvent 第 5.2 节。数据集、身份表示和排名策略必须另行声明，
相同公式不代表复现了作者的测试集或官方评测程序。
"""

from collections import Counter
from math import isfinite

from src.modules.event_recognition.labels import LABEL_MAP

EVENTS = tuple(event for event in LABEL_MAP if event != "blank")
ALIASES = {name.casefold(): name for name in LABEL_MAP}
ALIASES.update({"assist": "ast", "background": "blank"})


def event_name(value):
    """统一本项目的十类名称；未知类别不能悄悄变成背景。"""
    key = str(value).strip().casefold()
    if key not in ALIASES:
        raise ValueError(f"未知事件类别：{value!r}")
    return ALIASES[key]


def ratio(numerator, denominator):
    """论文约定零分母时取零。"""
    return numerator / denominator if denominator else 0.0


def prf(matches, predicted, reference):
    """从一对一匹配计数计算精确率、召回率和 F1。"""
    return {"matches": matches, "predicted": predicted, "reference": reference,
            "precision": ratio(matches, predicted), "recall": ratio(matches, reference),
            "f1": ratio(2 * matches, predicted + reference)}


def same_actor(left, right):
    """缺失身份永远不与 GT 匹配；不把两个 anonymous 当成同一个人。"""
    return bool(left.get("actor")) and left.get("actor") == right.get("actor")


def lcs_pairs(predicted, reference, with_actor=False):
    """返回保持顺序的一对一最长公共子序列索引。

    Type 对齐发生长度平局时先跳过预测项，不使用身份正确与否打破平局，
    避免为了提高 PA 偷看身份答案。论文未给出平局实现，此处明确固定规则。
    """
    def equal(i, j):
        return (predicted[i]["event"] == reference[j]["event"]
                and (not with_actor or same_actor(predicted[i], reference[j])))

    n, m = len(predicted), len(reference)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = (1 + table[i + 1][j + 1] if equal(i, j)
                           else max(table[i + 1][j], table[i][j + 1]))
    result, i, j = [], 0, 0
    while i < n and j < m:
        if equal(i, j):
            result.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return result


def sequence_metrics(samples):
    """跨视频累计 LCS 后算 micro-F1；不先算每视频 F1 再平均。"""
    type_matches = full_matches = actor_matches = n_pred = n_ref = 0
    details = []
    for sample in samples:
        pred, ref = sample["predictions"], sample["references"]
        pairs = lcs_pairs(pred, ref)
        full = lcs_pairs(pred, ref, with_actor=True)
        actors = sum(same_actor(pred[i], ref[j]) for i, j in pairs)
        type_matches += len(pairs)
        full_matches += len(full)
        actor_matches += actors
        n_pred += len(pred)
        n_ref += len(ref)
        details.append({"sample_id": sample["sample_id"], "predicted": len(pred),
                        "reference": len(ref), "type_pairs": pairs, "full_pairs": full})
    return {"event_type": prf(type_matches, n_pred, n_ref),
            "full_event": prf(full_matches, n_pred, n_ref),
            "participant_accuracy": ratio(actor_matches, type_matches),
            "participant_accuracy_denominator": type_matches, "samples": details}


def player_pair_metrics(samples):
    """人物—事件集合 Macro-F1，以及按每视频全局排名计算的 Macro-Recall@K。

    一个视频里同一身份的同类事件按集合去重（与序列任务不同）。身份缺失的
    预测各算一个无法匹配的候选。ranked_pairs 必须包含全体轨迹的所有非背景
    类别概率，而不只是最终 argmax；否则不输出 Recall@K。
    """
    counts = {name: Counter() for name in EVENTS}
    ranking_available = all("ranked_pairs" in sample for sample in samples)
    for sample in samples:
        def keys(records, side="prediction"):
            return {(row.get("actor") or f"__unknown_{side}_{i}", row["event"])
                    for i, row in enumerate(records) if row["event"] != "blank"}
        ref, pred = keys(sample["references"], "reference"), keys(sample["predictions"])
        for name in EVENTS:
            counts[name]["reference"] += sum(event == name for _, event in ref)
            counts[name]["predicted"] += sum(event == name for _, event in pred)
            counts[name]["matches"] += sum(event == name for _, event in ref & pred)
        if ranking_available:
            ranked = sorted(sample["ranked_pairs"], key=lambda row: -row["confidence"])
            for k in (1, 3, 5):
                found = ref & keys(ranked[:k])
                for name in EVENTS:
                    counts[name][f"recall_matches_{k}"] += sum(e == name for _, e in found)
    per_class = {name: prf(c["matches"], c["predicted"], c["reference"])
                 for name, c in counts.items()}
    result = {"per_class": per_class,
              "macro_f1_fixed_10": sum(c["f1"] for c in per_class.values()) / len(EVENTS),
              "macro_f1_gt_supported": ratio(sum(c["f1"] for c in per_class.values() if c["reference"]),
                                             sum(bool(c["reference"]) for c in per_class.values())),
              "unsupported_classes": [n for n, c in per_class.items() if not c["reference"]],
              "recall_at_k": None}
    if ranking_available:
        result["recall_at_k"] = {
            str(k): sum(ratio(c[f"recall_matches_{k}"], c["reference"])
                        for c in counts.values()) / len(EVENTS) for k in (1, 3, 5)}
    return result


def interval_overlap(predicted, reference, modified=False):
    """普通时间 IoU 用并集；论文 mIoU 用较短区间，二者不能混称。"""
    for interval in (predicted, reference):
        if (len(interval) != 2 or not all(isfinite(float(x)) for x in interval)
                or interval[0] < 0 or interval[1] <= interval[0]):
            raise ValueError(f"无效时间区间：{interval}")
    intersection = max(0, min(predicted[1], reference[1]) - max(predicted[0], reference[0]))
    lengths = (predicted[1] - predicted[0], reference[1] - reference[0])
    denominator = min(lengths) if modified else sum(lengths) - intersection
    return ratio(intersection, denominator)


def temporal_hits(targets):
    """每个手工指定目标算一次 Hit，漏检算零，阈值严格大于而非大于等于。

    prediction 必须是该目标轨迹的最高 gate 片段，不能用最高分类概率、
    扩展素材边界或多个候选里对 GT 最有利的片段代替。
    """
    rows = []
    for target in targets:
        ref, pred = target["reference"], target.get("prediction")
        overlap = interval_overlap(pred["interval"], ref["interval"], True) if pred else 0
        iou = interval_overlap(pred["interval"], ref["interval"]) if pred else 0
        correct = bool(pred and pred["event"] == ref["event"])
        rows.append({"miou": overlap, "iou": iou, "class_correct": correct})
    return {"target_count": len(rows),
            **{f"hit_at_{threshold}": ratio(sum(r["class_correct"] and r["miou"] > threshold
                                                for r in rows), len(rows))
               for threshold in (0.3, 0.5)}, "targets": rows}
