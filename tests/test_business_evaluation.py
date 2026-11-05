import unittest
from types import SimpleNamespace
import numpy as np
from business_qa import answer_question, final_answer
from business import split_chunks
from evaluation.metrics import context_relevancy, evidence_recall, answer_correctness, validate_labels
from evaluation.run import summarize, validate_input
from evaluation.tokenization import ChineseRougeTokenizer, rouge_l


class CharacterEncoder:
    def __call__(self, text, add_special_tokens=True, **kwargs):
        value = {"input_ids": list(range(len(text) + (2 if add_special_tokens else 0)))}
        if kwargs.get("return_offsets_mapping"):
            value["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return value


class Judge:
    def call(self, task, instruction, payload, validator):
        if "answer_classifications" in instruction:
            value = {"answer_classifications": [{"id": "A1", "supported": 1, "reason": "matched"}], "reference_classifications": [{"id": "G1", "covered": 1, "reason": "matched"}, {"id": "G2", "covered": 0, "reason": "missing"}]}
        elif task.endswith("reference_statements"):
            value = {"statements": ["一项事实", "另一项事实"]}
        elif task.endswith("answer_statements"):
            value = {"statements": ["一项事实"]}
        elif "evidence" in payload:
            value = {"classifications": [{"id": row["id"], "attributed": int(i == 0), "reason": "fixture"} for i, row in enumerate(payload["evidence"])]}
        else:
            value = {"score": 2, "reason": "fixture"}
        validator(value)
        return value


class BusinessEvaluationTests(unittest.TestCase):
    def test_chinese_identity(self):
        self.assertEqual(rouge_l("不得兼任。", "不得兼任。"), 1)

    def test_known_lcs(self):
        self.assertAlmostEqual(rouge_l("银行报告", "银行"), 2 / 3)

    def test_numeric_tokens_and_latin(self):
        tokens = ChineseRougeTokenizer().tokenize("资本１０％，BGE-M3，１５日")
        self.assertIn("10%", tokens)
        self.assertIn("15", tokens)
        self.assertIn("bge-m3", tokens)

    def test_different_deadlines_are_different(self):
        self.assertLess(rouge_l("5日内报告", "15日内报告"), 1)

    def test_empty_rouge(self):
        self.assertEqual(rouge_l("", "标准答案"), 0)

    def test_retrieval_metrics_do_not_need_generated_answer(self):
        self.assertEqual(context_relevancy("问题", ["资料"], Judge(), "x")[0], 1)
        self.assertEqual(evidence_recall("问题", ["资料"], ["证据一", "证据二"], Judge(), "x")[0], 0.5)

    def test_empty_retrieval(self):
        self.assertEqual(context_relevancy("问题", [], None, "x")[0], 0)
        self.assertEqual(evidence_recall("问题", [], ["证据"], None, "x")[0], 0)

    def test_missing_classifications_rejected(self):
        with self.assertRaises(ValueError):
            validate_labels({"rows": [{"id": "E1", "hit": 1, "reason": "ok"}]}, "rows", ["E1", "E2"], "hit")

    def test_duplicate_classifications_rejected(self):
        row = {"id": "E1", "hit": 1, "reason": "ok"}
        with self.assertRaises(ValueError):
            validate_labels({"rows": [row, row]}, "rows", ["E1", "E2"], "hit")

    def test_correctness_formula(self):
        embedding = SimpleNamespace(model=SimpleNamespace(tokenizer=CharacterEncoder(), max_seq_length=2048), batch_encode=lambda text: np.array([[1, 0], [1, 0]], dtype=np.float32))
        score, detail = answer_correctness("问题", "答案", "标准答案", Judge(), embedding, "x")
        self.assertAlmostEqual(score, 0.75)
        self.assertEqual(detail["fn"], 1)

    def test_correctness_empty_answer(self):
        self.assertEqual(answer_correctness("问题", "", "标准答案", None, None, "x")[0], 0)

    def test_failed_scores_keep_denominator_visible(self):
        rows = [{"question_type": "Fact Retrieval", "generated_answer": "答案", "metrics": {"rouge_l": {"score": 1, "status": "ok"}}}, {"question_type": "Fact Retrieval", "generated_answer": "", "metrics": {"rouge_l": {"score": None, "status": "error"}}}]
        output = summarize(rows)["overall"]["metrics"]["rouge_l"]
        self.assertEqual((output["expected_n"], output["valid_n"], output["failed_n"]), (2, 1, 1))

    def test_partial_run_requires_explicit_flag(self):
        row = {"id": "fixture", "question": "问题", "question_type": "Fact Retrieval", "ground_truth": "答案", "generated_answer": "答案", "context": ["资料"], "evidence": ["资料"]}
        validate_input([row], True)
        with self.assertRaises(ValueError):
            validate_input([row], False)

    def test_actual_context_is_preserved_on_generation_error(self):
        def fail(**kwargs):
            raise RuntimeError("fixture")
        candidate = SimpleNamespace(docs=["资料一", "资料二"], doc_metadata=[{}, {}], doc_scores=np.array([1, 0.5]))
        rag = SimpleNamespace(retrieve=lambda q: [candidate], embedding_model=SimpleNamespace(model=SimpleNamespace(tokenizer=CharacterEncoder())), qa_llm=SimpleNamespace(infer=fail))
        result = answer_question(rag, "问题", top_k=1)
        self.assertEqual(result["context"], ["资料一"])
        self.assertEqual(result["retrieved_candidates"], ["资料一", "资料二"])
        self.assertEqual(result["status"], "generation_error")

    def test_final_answer_removes_tagged_reasoning(self):
        self.assertEqual(final_answer("<think>内部内容</think>结论：不可以。"), "结论：不可以。")
        with self.assertRaises(ValueError):
            final_answer("<think>未完成")

    def test_chunks_fit_and_cover_source(self):
        text = "甲乙丙丁戊己庚辛壬癸" * 5
        pieces = list(split_chunks(text, "标题", CharacterEncoder(), 20, 2))
        self.assertTrue(all(len(CharacterEncoder()(p[0])["input_ids"]) <= 20 for p in pieces))
        covered = set()
        for _, start, end in pieces:
            covered.update(range(start, end))
        self.assertEqual(covered, set(range(len(text))))


if __name__ == "__main__":
    unittest.main()
