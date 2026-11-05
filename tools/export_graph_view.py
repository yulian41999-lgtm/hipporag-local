"""Export bounded, real graph neighborhoods for a read-only visualization."""
from pathlib import Path
from datetime import datetime
import json,igraph,hashlib
ROOT=Path('/home/ailab/data/HippoRAG/hipporag_local')
r=Path(json.loads((ROOT/'outputs/current_1700_recovery.json').read_text())['target'])
g=igraph.Graph.Read_Pickle(str(next(r.rglob('graph.pickle'))))
corpus=json.loads((r/'banking/corpus.json').read_text())
chunks={'chunk-'+hashlib.md5(x['text'].encode()).hexdigest():x for x in corpus}
archive=json.loads(next(r.rglob('openie_results_ner_*.json')).read_text())
by_chunk={x['idx']:x for x in archive['docs']}
out={'created_at':datetime.now().astimezone().isoformat(),'documents':1700,'nodes':g.vcount(),'edges':g.ecount(),'views':[]}
for seed in ['小额担保贷款','资本充足率','存款准备金','金融资产管理公司']:
 center=next(v.index for v in g.vs if v['content']==seed)
 incident=[g.es[e] for e in g.incident(center)]
 def neighbor(e): return e.target if e.source==center else e.source
 ranked=sorted(incident,key=lambda e:(e['edge_kind']!='fact',-e['weight'],g.vs[neighbor(e)]['content']))
 entity_edges=[e for e in ranked if g.vs[neighbor(e)]['hash_id'].startswith('entity-')][:11]
 passage_edges=[e for e in ranked if g.vs[neighbor(e)]['hash_id'].startswith('chunk-')][:3]
 selected={center,*[neighbor(e) for e in entity_edges+passage_edges]}
 nodes=[]
 for i in sorted(selected):
  v=g.vs[i]; key=v['hash_id']; is_chunk=key.startswith('chunk-'); chunk=chunks.get(key)
  label=(chunk['source_id'] if chunk else key[:14]) if is_chunk else (v['content'] or '空实体')
  detail={'id':i,'label':label,'type':'passage' if is_chunk else 'entity','degree':g.degree(i),'center':i==center}
  if chunk:
   detail['document_title']=chunk.get('metadata',{}).get('title','')
   detail['source_id']=chunk['source_id']; detail['excerpt']=chunk['text'][:600]
  nodes.append(detail)
 edges=[]
 for e in g.es:
  if e.source not in selected or e.target not in selected: continue
  item={'source':e.source,'target':e.target,'kind':e['edge_kind'],'weight':round(float(e['weight']),3),'triples':[]}
  if e['edge_kind']=='fact':
   pair={g.vs[e.source]['content'].strip().lower(),g.vs[e.target]['content'].strip().lower()}
   seen=set()
   for chunk_key in (e['fact_source_counts'] or {}):
    for t in by_chunk.get(chunk_key,{}).get('extracted_triples',[]):
     if len(t)==3 and {str(t[0]).strip().lower(),str(t[2]).strip().lower()}==pair:
      triple=tuple(map(str,t))
      if triple not in seen: item['triples'].append(list(triple)); seen.add(triple)
     if len(item['triples'])>=3: break
    if len(item['triples'])>=3: break
   keys=list((e['fact_source_counts'] or {}).keys())
   item['source_documents']=list(dict.fromkeys(chunks[k]['source_id'] for k in keys if k in chunks))[:5]
  edges.append(item)
 out['views'].append({'seed':seed,'center':center,'full_degree':g.degree(center),'selected_neighbors':len(selected)-1,'total_neighbors':len(g.neighbors(center)),'nodes':nodes,'edges':edges})
dest=ROOT/'outputs/graph_visualization'; dest.mkdir(exist_ok=True)
p=dest/'neighborhoods.json'; p.write_text(json.dumps(out,ensure_ascii=False,separators=(',',':')))
print('EXPORT',p,'BYTES',p.stat().st_size,'VIEWS',[(v['seed'],len(v['nodes']),len(v['edges'])) for v in out['views']])
