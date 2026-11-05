from pathlib import Path
import runpy,json,gzip,datetime,hashlib
root=Path('/home/ailab/data/HippoRAG/hipporag_local')
m=runpy.run_path(str(root/'tools/graph_viewer/server.py'))
g=m['GRAPH'];chunks=m['CHUNKS'];docs=m['DOCS'];edge_rows=[]
for e in g.es:
 row={'source':e.source,'target':e.target,'kind':e['edge_kind'],'weight':float(e['weight']),'triples':[]}
 if e['edge_kind']=='fact':
  pair={g.vs[e.source]['content'].strip().lower(),g.vs[e.target]['content'].strip().lower()};seen=set()
  for ck in (e['fact_source_counts'] or {}):
   for t in docs.get(ck,{}).get('extracted_triples',[]):
    if len(t)==3 and {str(t[0]).strip().lower(),str(t[2]).strip().lower()}==pair:
     triple=tuple(map(str,t))
     if triple not in seen:row['triples'].append(list(triple));seen.add(triple)
    if len(row['triples'])>=6:break
   if len(row['triples'])>=6:break
  row['documents']=list(dict.fromkeys(chunks[k]['source_id'] for k in (e['fact_source_counts'] or {}) if k in chunks))[:8]
 edge_rows.append(row)
nodes=[m['node'](i) for i in range(g.vcount())]
result={'exported_at':datetime.datetime.now().astimezone().isoformat(),'stats':{'documents':1700,'chunks':len(chunks),'nodes':g.vcount(),'edges':g.ecount(),'entities':len(m['ENTITIES']),'state':'ready','default_id':m['LABEL_TO_ID'].get('小额担保贷款',0)},'nodes':nodes,'edges':edge_rows}
p=root/'reports/offline_20261102/graph-snapshot.json.gz'
with gzip.open(p,'wt',encoding='utf-8',compresslevel=6) as f:json.dump(result,f,ensure_ascii=False,separators=(',',':'))
print('EXPORT_COMPLETE',p,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
