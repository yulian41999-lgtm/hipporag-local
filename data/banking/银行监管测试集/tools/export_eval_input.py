"""Join gold data with actual system predictions; never substitute gold evidence for retrieval."""
import argparse
import json
from pathlib import Path

def merge_predictions(gold, predictions):
    expected = {q['id'] for q in gold}
    by_id = {}
    for p in predictions:
        ident = p['id']
        if ident in by_id:
            raise ValueError('Duplicate prediction ID: ' + ident)
        if not isinstance(p.get('generated_answer'), str):
            raise ValueError('generated_answer must be a string: ' + ident)
        context = p.get('context')
        if not isinstance(context, list) or not all(isinstance(c, str) for c in context):
            raise ValueError('context must be a list of actual retrieved text strings: ' + ident)
        by_id[ident] = p
    if set(by_id) != expected:
        raise ValueError('Prediction IDs mismatch. Missing: %s; unknown: %s' %
                         (sorted(expected - set(by_id)), sorted(set(by_id) - expected)))
    return [{'id': q['id'], 'source': q['source'], 'question': q['question'],
             'question_type': q['question_type'], 'ground_truth': q['answer'],
             'generated_answer': by_id[q['id']]['generated_answer'],
             'context': by_id[q['id']]['context'],
             'evidence': [s['quote'] for s in q['evidence_sources']],
             'evidence_relations': q['evidence_relations']} for q in gold]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, default=Path(__file__).resolve().parents[1] / 'banking_qa_15.json')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.gold.resolve(), args.predictions.resolve()}:
        parser.error('Output must not overwrite gold data or system predictions.')
    rows = merge_predictions(json.loads(args.gold.read_text(encoding='utf-8-sig')),
                             json.loads(args.predictions.read_text(encoding='utf-8-sig')))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Exported %s records.' % len(rows))
