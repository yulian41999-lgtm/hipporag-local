"""Read the current project's saved graph and print status without modifying it."""
from pathlib import Path
import json
import igraph

ROOT = Path(__file__).resolve().parents[1]
setup = json.loads((ROOT / 'outputs/current_1700_recovery.json').read_text())
output = Path(setup['target']).resolve()
output.relative_to(ROOT)
manifest = json.loads((output / 'banking/index/business_corpus_manifest.json').read_text())
paths = list((output / 'banking/index').rglob('graph.pickle'))
if len(paths) != 1:
    raise RuntimeError('Expected exactly one saved graph; found %d' % len(paths))
graph_path = paths[0]
g = igraph.Graph.Read_Pickle(str(graph_path))
sizes = g.connected_components(mode='weak').sizes()
report = {
    'index_state': manifest['state'],
    'chunks': manifest['chunks'],
    'nodes': g.vcount(),
    'edges': g.ecount(),
    'directed': g.is_directed(),
    'connected_components': len(sizes),
    'largest_component_nodes': max(sizes, default=0),
    'isolated_nodes': sum(d == 0 for d in g.degree()),
    'self_loops': sum(g.is_loop()),
    'parallel_edges': sum(g.is_multiple()),
    'graph_file_mib': round(graph_path.stat().st_size / 1024 ** 2, 2),
    'graph_path': str(graph_path),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
