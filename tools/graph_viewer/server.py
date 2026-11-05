"""Read-only graph explorer; loopback binding, no model calls or graph writes."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
import json,hashlib,igraph,time,threading
ROOT=Path(__file__).resolve().parents[2]
SETUP=json.loads((ROOT/'outputs/current_1700_recovery.json').read_text())
OUTPUT=Path(SETUP['target']).resolve(); OUTPUT.relative_to(ROOT)
GRAPH_PATH=next((OUTPUT/'banking/index').rglob('graph.pickle'))
GRAPH=igraph.Graph.Read_Pickle(str(GRAPH_PATH))
CORPUS=json.loads((OUTPUT/'banking/corpus.json').read_text())
CHUNKS={'chunk-'+hashlib.md5(c['text'].encode()).hexdigest():c for c in CORPUS}
ARCHIVE=json.loads(next((OUTPUT/'banking/index').rglob('openie_results_ner_*.json')).read_text())
DOCS={d['idx']:d for d in ARCHIVE['docs']}
DEGREES=GRAPH.degree()
ENTITIES=[(v.index,v['content']) for v in GRAPH.vs if v['hash_id'].startswith('entity-') and v['content']]
LABEL_TO_ID={name:i for i,name in ENTITIES}
CACHE={}; LOCK=threading.Lock()
def node(i,center=None):
 v=GRAPH.vs[i]; key=v['hash_id']; passage=key.startswith('chunk-'); chunk=CHUNKS.get(key)
 item={'id':i,'label':chunk['source_id'] if passage and chunk else (v['content'] or '空实体'),'type':'passage' if passage else 'entity','degree':DEGREES[i],'center':i==center}
 if chunk:
  item.update(source_id=chunk['source_id'],title=chunk.get('metadata',{}).get('title',''),excerpt=chunk['text'])
 return item
def neighborhood(center,limit):
 key=(center,limit)
 with LOCK:
  if key in CACHE: return CACHE[key]
 incident=[GRAPH.es[i] for i in GRAPH.incident(center)]
 def other(e): return e.target if e.source==center else e.source
 ranked=sorted(incident,key=lambda e:(e['edge_kind']!='fact',-float(e['weight']),GRAPH.vs[other(e)]['content']))
 entity=[other(e) for e in ranked if GRAPH.vs[other(e)]['hash_id'].startswith('entity-')]
 passage=[other(e) for e in ranked if GRAPH.vs[other(e)]['hash_id'].startswith('chunk-')]
 chosen=list(dict.fromkeys(entity[:max(1,limit-3)]+passage[:3]))
 if len(chosen)<limit:
  for i in entity+passage:
   if i not in chosen and len(chosen)<limit: chosen.append(i)
 chosen=chosen[:limit]; selected={center,*chosen}; edges=[]
 for i in selected:
  for ei in GRAPH.incident(i):
   e=GRAPH.es[ei]
   if e.source not in selected or e.target not in selected or e.source!=i: continue
   item={'source':e.source,'target':e.target,'kind':e['edge_kind'],'weight':float(e['weight']),'triples':[]}
   if e['edge_kind']=='fact':
    pair={GRAPH.vs[e.source]['content'].strip().lower(),GRAPH.vs[e.target]['content'].strip().lower()}
    seen=set()
    for ck in (e['fact_source_counts'] or {}):
     for t in DOCS.get(ck,{}).get('extracted_triples',[]):
      if len(t)==3 and {str(t[0]).strip().lower(),str(t[2]).strip().lower()}==pair:
       triple=tuple(map(str,t))
       if triple not in seen: item['triples'].append(list(triple)); seen.add(triple)
      if len(item['triples'])>=6: break
     if len(item['triples'])>=6: break
    item['documents']=list(dict.fromkeys(CHUNKS[k]['source_id'] for k in (e['fact_source_counts'] or {}) if k in CHUNKS))[:8]
   edges.append(item)
 result={'center':center,'seed':node(center)['label'],'total_neighbors':len(GRAPH.neighbors(center)),'nodes':[node(i,center) for i in [center]+chosen],'edges':edges}
 with LOCK:
  if len(CACHE)>=100: CACHE.clear()
  CACHE[key]=result
 return result
class Handler(BaseHTTPRequestHandler):
 def log_message(self,format,*args): pass
 def send(self,data,status=200,kind='application/json; charset=utf-8'):
  if isinstance(data,(dict,list)): data=json.dumps(data,ensure_ascii=False).encode()
  self.send_response(status); self.send_header('Content-Type',kind)
  self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
  self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Referrer-Policy','no-referrer')
  self.end_headers(); self.wfile.write(data)
 def do_GET(self):
  try:
   u=urlsplit(self.path); p=parse_qs(u.query)
   if u.path=='/':
    return self.send((Path(__file__).parent/'index.html').read_bytes(),kind='text/html; charset=utf-8')
   if u.path=='/api/stats':
    return self.send({'documents':1700,'chunks':5337,'nodes':GRAPH.vcount(),'edges':GRAPH.ecount(),'entities':len(ENTITIES),'state':'ready','default_id':LABEL_TO_ID.get('小额担保贷款',0)})
   if u.path=='/api/search':
    q=p.get('q',[''])[0].strip().casefold()[:120]
    matches=[(i,name) for i,name in ENTITIES if q in name.casefold()]
    matches.sort(key=lambda x:(not x[1].casefold().startswith(q),-DEGREES[x[0]],len(x[1])))
    return self.send({'matches':[{'id':i,'label':name,'degree':DEGREES[i]} for i,name in matches[:30]],'total':len(matches)})
   if u.path=='/api/neighborhood':
    i=int(p.get('id',['0'])[0]); limit=max(4,min(30,int(p.get('limit',['16'])[0])))
    if not 0<=i<GRAPH.vcount(): return self.send({'error':'节点不存在'},404)
    return self.send(neighborhood(i,limit))
   return self.send({'error':'未找到'},404)
  except (ValueError,TypeError):
   return self.send({'error':'参数无效'},400)
  except (BrokenPipeError,ConnectionResetError): pass
  except Exception as exc:
   print('REQUEST_ERROR',type(exc).__name__,flush=True)
   return self.send({'error':'读取失败，请检查服务日志'},500)
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',8790),Handler)
 print('GRAPH_VIEWER_READY http://127.0.0.1:8790/ nodes=%d edges=%d' % (GRAPH.vcount(),GRAPH.ecount()),flush=True)
 server.serve_forever()
