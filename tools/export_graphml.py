"""Export the current saved graph for Gephi, keeping complex attributes as JSON."""
from pathlib import Path
from datetime import datetime
import json,hashlib,igraph,math
ROOT=Path(__file__).resolve().parents[1]
setup=json.loads((ROOT/'outputs/current_1700_recovery.json').read_text())
output=Path(setup['target']).resolve(); output.relative_to(ROOT)
source=next((output/'banking/index').rglob('graph.pickle'))
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
g=igraph.Graph.Read_Pickle(str(source))
destination=ROOT/'outputs/graph_visualization'/('banking-1700-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.graphml')
assert not destination.exists()
destination.parent.mkdir(parents=True,exist_ok=True)
serialized={'graph':[],'vertex':[],'edge':[]}
scalar=(str,int,float,bool)
def xml_valid(text):
 return all(ord(c) in (9,10,13) or 0x20<=ord(c)<=0xD7FF or 0xE000<=ord(c)<=0xFFFD or 0x10000<=ord(c)<=0x10FFFF for c in text)
def display(text):
 return ''.join(c if xml_valid(c) else ('\\u%04x' % ord(c)) for c in text)
g.vs['label']=[display((v['content'] or '空实体')) if v['hash_id'].startswith('entity-') else '文本块-'+v['hash_id'][-8:] for v in g.vs]
for scope,sequence in [('vertex',g.vs),('edge',g.es)]:
 for attr in sequence.attributes():
  values=sequence[attr]
  if any(not isinstance(v,scalar) or (isinstance(v,str) and not xml_valid(v)) for v in values):
   sequence[attr]=[json.dumps(v,ensure_ascii=True,separators=(',',':')) for v in values]
   serialized[scope].append(attr)
for attr in g.attributes():
 if not isinstance(g[attr],scalar) or (isinstance(g[attr],str) and not xml_valid(g[attr])):
  g[attr]=json.dumps(g[attr],ensure_ascii=True,separators=(',',':'))
  serialized['graph'].append(attr)
g.write_graphml(str(destination))
loaded=igraph.Graph.Read_GraphML(str(destination))
assert loaded.vcount()==g.vcount() and loaded.ecount()==g.ecount()
assert loaded.is_directed()==g.is_directed()
assert loaded.get_edgelist()==g.get_edgelist()
for attr in g.vs.attributes(): assert loaded.vs[attr]==g.vs[attr], ('vertex attribute changed',attr)
numeric_errors={}
for attr in g.es.attributes():
 a=loaded.es[attr]; b=g.es[attr]
 if all(isinstance(x,(float,int)) for x in b):
  assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in zip(a,b)), ('numeric edge attribute changed',attr)
  numeric_errors[attr]=max((abs(x-y) for x,y in zip(a,b)),default=0)
 else:
  assert a==b, ('edge attribute changed',attr)
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
report={'format':'GraphML','source':str(source),'destination':str(destination),'nodes':g.vcount(),'edges':g.ecount(),'directed':g.is_directed(),'bytes':destination.stat().st_size,'json_encoded_attributes':serialized,'round_trip_verified':True,'numeric_max_absolute_error':numeric_errors,'source_unchanged':True}
destination.with_suffix('.export.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
