"""GraphRAG-Bench metric formulas with complete classification validation."""
import numpy as np


def require_reason(row):
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError("A nonempty judgment reason is required")


def validate_labels(payload, key, ids, label):
    rows = payload.get(key)
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError("Missing or extra classifications")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in ids or row["id"] in seen:
            raise ValueError("Unknown or duplicate classification ID")
        if type(row.get(label)) is not int or row[label] not in (0, 1):
            raise ValueError("Classification must be integer 0 or 1")
        require_reason(row)
        seen.add(row["id"])
    if seen != set(ids):
        raise ValueError("Classification IDs must exactly cover the input")


def context_relevancy(question, contexts, judge, task):
    if not any(c.strip() for c in contexts) or "\n".join(contexts).strip() == question.strip():
        return 0.0, {"reason": "empty_context"}

    def validate(value):
        if type(value.get("score")) is not int or value["score"] not in (0, 1, 2):
            raise ValueError("Relevance score must be 0, 1 or 2")
        require_reason(value)

    ratings = [judge.call(
        f"{task}/context_relevancy/{i}",
        '仅根据问题和整组上下文判断是否含有回答所需信息。无关=0，部分包含=1，完整包含=2。返回 {"score":0,"reason":"理由"}。不得使用外部知识。',
        {"question": question, "contexts": contexts}, validate,
    ) for i in (1, 2)]
    return sum(row["score"] for row in ratings) / 4, {"ratings": ratings}


def evidence_recall(question, contexts, evidence, judge, task):
    if not evidence or any(not isinstance(e, str) or not e.strip() for e in evidence):
        raise ValueError("Nonempty gold evidence is required")
    if not any(c.strip() for c in contexts):
        return 0.0, {"reason": "empty_context", "matched": 0, "total": len(evidence)}
    entries = [{"id": f"E{i}", "text": text} for i, text in enumerate(evidence, 1)]
    ids = [item["id"] for item in entries]
    result = judge.call(
        task + "/evidence_recall",
        '逐条判断标准证据是否能从上下文直接得到支持。完整支持=1，否则=0。只参考上下文，不使用外部知识。每个证据 ID 必须且只能出现一次。返回 {"classifications":[{"id":"E1","attributed":1,"reason":"理由"}]}。',
        {"question": question, "contexts": contexts, "evidence": entries},
        lambda p: validate_labels(p, "classifications", ids, "attributed"),
    )
    return sum(row["attributed"] for row in result["classifications"]) / len(entries), result


def extract_statements(question, text, judge, task):
    def validate(value):
        rows = value.get("statements")
        if not isinstance(rows, list) or not rows or any(not isinstance(s, str) or not s.strip() for s in rows):
            raise ValueError("Nonempty atomic statements are required")
        if len(set(rows)) != len(rows):
            raise ValueError("Repeated statements")
    return judge.call(
        task, '把给定回答拆分为可独立核验、无代词歧义的原子事实陈述。保留主体、条件、数字、时限、否定及法规引用，不增加原文没有的事实。去除纯标题，重复事实只保留一次。返回 {"statements":["事实1","事实2"]}。',
        {"question": question, "answer": text}, validate,
    )["statements"]


def answer_correctness(question, answer, reference, judge, embedding, task):
    if not answer.strip():
        return 0.0, {"reason": "empty_answer", "factuality_f1": 0.0, "semantic_similarity": 0.0}
    if not reference.strip():
        raise ValueError("Gold answer is empty")
    actual = extract_statements(question, answer, judge, task + "/answer_statements")
    expected = extract_statements(question, reference, judge, task + "/reference_statements")
    a_rows = [{"id": f"A{i}", "text": s} for i, s in enumerate(actual, 1)]
    g_rows = [{"id": f"G{i}", "text": s} for i, s in enumerate(expected, 1)]

    def validate(value):
        validate_labels(value, "answer_classifications", [r["id"] for r in a_rows], "supported")
        validate_labels(value, "reference_classifications", [r["id"] for r in g_rows], "covered")

    result = judge.call(
        task + "/correctness",
        '比较生成答案和标准答案的原子事实。逐条标记生成事实是否受标准答案直接支持（supported:1/0），逐条标记标准事实是否被生成答案正确覆盖（covered:1/0）。不得使用外部知识；注意否定、主体、适用条件、时限和金额。每个 ID 必须且只能出现一次。返回 {"answer_classifications":[{"id":"A1","supported":1,"reason":"理由"}],"reference_classifications":[{"id":"G1","covered":1,"reason":"理由"}]}。',
        {"question": question, "answer_statements": a_rows, "reference_statements": g_rows}, validate,
    )
    tp = sum(r["supported"] for r in result["answer_classifications"])
    fp = len(actual) - tp
    fn = sum(1 - r["covered"] for r in result["reference_classifications"])
    factuality = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    # Fail rather than silently truncate the semantic-similarity input.
    for text in (answer, reference):
        size = len(embedding.model.tokenizer(text, truncation=False)["input_ids"])
        if size > embedding.model.max_seq_length:
            raise ValueError(f"Evaluation text has {size} tokens; increase evaluation embedding_max_seq_length")
    vectors = embedding.batch_encode([answer, reference])
    cosine = float(np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
    similarity = (float(np.clip(cosine, -1, 1)) + 1) / 2
    return 0.75 * factuality + 0.25 * similarity, {
        "factuality_f1": factuality, "semantic_similarity": similarity,
        "tp": tp, "fp": fp, "fn": fn,
        "answer_statements": a_rows, "reference_statements": g_rows, **result,
    }
