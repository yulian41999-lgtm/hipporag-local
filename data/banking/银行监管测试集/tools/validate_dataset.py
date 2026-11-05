import json
import hashlib
from collections import Counter
from pathlib import Path
from inspect_corpus import ROOT, OUT, load_doc
from export_eval_input import merge_predictions

def validate():
    rows = json.loads((OUT / 'banking_qa_15.json').read_text(encoding='utf-8'))
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(len(rows) == 15, 'Question count must be 15.')
    check(len(set(q['id'] for q in rows)) == 15, 'Question IDs must be unique.')
    check(len(set(q['question'] for q in rows)) == 15, 'Question text must be unique.')
    types = Counter(q['question_type'] for q in rows)
    check(types == {'Fact Retrieval': 10, 'Complex Reasoning': 5}, 'Question type count mismatch.')
    doc_uses = Counter()
    total_evidence = 0
    for q in rows:
        ident = q['id']
        required = ['id', 'source', 'question', 'answer', 'question_type', 'evidence',
                    'evidence_relations', 'evidence_sources', 'answer_key_points',
                    'reasoning_steps', 'applicability', 'cross_document_dependencies', 'topic']
        check(all(k in q for k in required), ident + ': missing required field.')
        check(all(q[k].strip() for k in ['question', 'answer', 'evidence', 'evidence_relations']), ident + ': empty content.')
        check(q['answer'].startswith('结论\n\n') and '\n\n一、' in q['answer'] and '\n\n二、' in q['answer']
              and '\n\n总结\n\n' in q['answer'], ident + ': sample answer structure missing.')
        check('根据《' in q['answer'] and '本地标示现行有效' in q['answer'], ident + ': answer lacks explicit legal basis.')
        check(q['question_style'] and q['style_reference_examples'], ident + ': sample style mapping missing.')
        check(not any(hint in q['question'] for hint in ['根据本地', '请分别判断', '结合该办法', '根据《']),
              ident + ': question contains unnecessary source or task hints.')
        check('context' not in q and 'generated_answer' not in q, ident + ': system output leaked into gold.')
        doc_ids = set(s['doc_id'] for s in q['evidence_sources'])
        doc_uses.update(doc_ids)
        evidence_ids = set(s['evidence_id'] for s in q['evidence_sources'])
        check(len(evidence_ids) == len(q['evidence_sources']), ident + ': repeated evidence IDs.')
        check(doc_ids == set(q['applicability']['version_doc_ids']), ident + ': applicability versions mismatch.')
        check(q['evidence'] == '\n\n'.join(s['quote'] for s in q['evidence_sources']), ident + ': evidence text mismatch.')
        check(len(q['answer_key_points']) >= 2, ident + ': insufficient scoring points.')
        for s in q['evidence_sources']:
            total_evidence += 1
            d = load_doc(ROOT / s['source_file'])
            check(d['validity'] == '现行有效' == s['local_validity'], ident + ': source not locally valid.')
            check(d['sha256'] == s['source_sha256'], ident + ': source hash mismatch.')
            check(d['publication_date'] and d['effective_date'] and '*' not in d['publication_date'], ident + ': unclear date.')
            check(d['doc_id'] and d['title'] and d['publication_date'] and d['effective_date'] and d['sha256'],
                  ident + ': missing source version identification.')
            normalized = (OUT / s['normalized_file']).read_text(encoding='utf-8').splitlines()
            start, end = s['normalized_line_start'], s['normalized_line_end']
            check(1 <= start <= end <= len(normalized), ident + ': invalid evidence range.')
            check(s['quote'] == '\n'.join(normalized[start-1:end]), ident + ': normalized excerpt mismatch.')
            check(s['quote'] == '\n'.join(d['lines'][start-1:end]), ident + ': source re-extraction mismatch.')
            check('\ufffd' not in s['quote'] and '\x00' not in s['quote'], ident + ': damaged evidence text.')
            expected_quote = s['quote']
            if ident == 'Compliance-F005':
                expected_quote = expected_quote[expected_quote.index('离任审计结论显示'):]
            check(expected_quote in q['answer'], ident + ': answer quotation is missing or altered.')
        if q['question_type'] == 'Fact Retrieval':
            check(len(doc_ids) == 1 and len(q['evidence_sources']) == 1, ident + ': fact question must use one evidence passage.')
            check(not q['reasoning_steps'], ident + ': fact question contains reasoning annotation.')
        else:
            check(len(doc_ids) >= 2, ident + ': reasoning needs multiple documents.')
            check(len(q['reasoning_steps']) >= 3, ident + ': missing reasoning steps.')
            check(set(x['doc_id'] for x in q['cross_document_dependencies']) == doc_ids, ident + ': document dependency coverage mismatch.')
            used = set()
            for step in q['reasoning_steps']:
                check(set(step['evidence_ids']) <= evidence_ids, ident + ': unknown reasoning evidence.')
                used.update(step['evidence_ids'])
            check(used == evidence_ids, ident + ': cited evidence unused in reasoning.')
            final_ids = set(q['reasoning_steps'][-1]['evidence_ids'])
            check(len(set(s['doc_id'] for s in q['evidence_sources'] if s['evidence_id'] in final_ids)) >= 2,
                  ident + ': final inference does not combine documents.')
    check(len(doc_uses) >= 10, 'At least 10 distinct source documents required.')
    questions = json.loads((OUT / 'questions.json').read_text(encoding='utf-8'))
    check(questions == [{'id': q['id'], 'question': q['question']} for q in rows],
          'Inference questions differ from gold data or contain extra fields.')
    audit = json.loads((OUT / 'content_audit.json').read_text(encoding='utf-8'))
    check(audit['dataset_sha256'] == hashlib.sha256((OUT / 'banking_qa_15.json').read_bytes()).hexdigest(),
          'Gold data differs from the audited dataset.')
    manifest = json.loads((OUT / 'source_manifest.json').read_text(encoding='utf-8'))
    check({d['doc_id'] for d in manifest} == set(doc_uses), 'Source manifest does not match gold data.')
    check(max(doc_uses.values()) <= 3, 'A source document is used by more than 3 questions.')
    facts = [q for q in rows if q['question_type'] == 'Fact Retrieval']
    check(len(set(q['evidence_sources'][0]['doc_id'] for q in facts)) == 10,
          'Fact questions must use 10 distinct source documents.')
    # Isolated in-memory fixtures exercise adapter alignment and failure paths.
    # These fixtures are never exported or treated as actual model results.
    fixture = [{'id': q['id'], 'generated_answer': 'ADAPTER_FIXTURE_ONLY',
                'context': ['RETRIEVAL_FIXTURE_ONLY']} for q in reversed(rows)]
    merged = merge_predictions(rows, fixture)
    check(all(m['ground_truth'] == q['answer'] and m['generated_answer'] == 'ADAPTER_FIXTURE_ONLY'
              and m['context'] == ['RETRIEVAL_FIXTURE_ONLY'] and isinstance(m['evidence'], list)
              for m, q in zip(merged, rows)), 'Adapter alignment or gold/result separation failed.')
    rejection_checks = {}
    for name, invalid in [('missing_id', fixture[:-1]), ('duplicate_id', fixture + fixture[:1]),
                          ('wrong_context_type', [dict(fixture[0], context='invalid')] + fixture[1:])]:
        try:
            merge_predictions(rows, invalid)
        except ValueError:
            rejection_checks[name] = True
        else:
            rejection_checks[name] = False
            errors.append('Adapter failed to reject ' + name)
    report = {'passed': not errors, 'question_count': len(rows), 'question_types': dict(types),
              'distinct_document_count': len(doc_uses), 'evidence_passage_count': total_evidence,
              'cross_document_reasoning_count': sum(q['question_type'] == 'Complex Reasoning' and len(q['applicability']['version_doc_ids']) >= 2 for q in rows),
              'maximum_questions_per_document': max(doc_uses.values()), 'document_question_counts': dict(sorted(doc_uses.items())),
              'all_sources_locally_marked_valid': all(s['local_validity'] == '现行有效' for q in rows for s in q['evidence_sources']),
              'answer_style': '结论＋分项法规依据及原文＋适用分析＋总结',
              'inference_question_count': len(questions),
              'audited_dataset_hash_match': audit['dataset_sha256'] == hashlib.sha256((OUT / 'banking_qa_15.json').read_bytes()).hexdigest(),
              'question_character_range': [min(len(q['question']) for q in rows), max(len(q['question']) for q in rows)],
              'answer_character_range': [min(len(q['answer']) for q in rows), max(len(q['answer']) for q in rows)],
              'adapter_rejection_checks': rejection_checks, 'errors': errors,
              'review_scope': '本脚本核验结构、来源版本、原文一致性、推理输入及评测适配，并确认题库与content_audit.json所记录哈希一致；历史逐题语义与独立HTML解析记录见该文件。通过不代表重新完成法律语义复审或全部外部现行效力确认。'}
    (OUT / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    validate()
