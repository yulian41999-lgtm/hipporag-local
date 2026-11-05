from pathlib import Path
import sys,unittest,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from settings import load_settings
load_settings()
from evaluation.repair import ReusingJudge
class Log:
 def __init__(self):self.values=[]
 def open(self,*args,**kwargs):return self
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def write(self,value):self.values.append(json.loads(value))
class Delegate:
 def __init__(self):self.calls=[]
 def call(self,*args):self.calls.append(args);return {'new':True}
class Tests(unittest.TestCase):
 def setup_judge(self):
  log=Log();delegate=Delegate()
  original=[{'task':'Q/answer_statements','input':{'question':'原题','answer':'原回答'},'raw_response':'{"statements":["原事实"]}','attempt':0,'status':'ok'}]
  return ReusingJudge(original,delegate,log),delegate,log
 def test_exact_payload_reuses_without_model(self):
  judge,delegate,log=self.setup_judge()
  result=judge.call('Q/answer_statements','same prompt',{'answer':'原回答','question':'原题'},lambda p:self.assertEqual(p['statements'],['原事实']))
  self.assertEqual(result['statements'],['原事实']);self.assertEqual(delegate.calls,[]);self.assertEqual(judge.reused,1);self.assertEqual(log.values[0]['status'],'reused')
 def test_changed_payload_never_reuses(self):
  judge,delegate,log=self.setup_judge()
  judge.call('Q/answer_statements','same prompt',{'question':'原题','answer':'不同回答'},lambda p:None)
  self.assertEqual(len(delegate.calls),1);self.assertEqual(judge.reused,0)
 def test_cached_result_is_revalidated(self):
  judge,delegate,log=self.setup_judge()
  def reject(value):raise ValueError('invalid cached result')
  with self.assertRaises(ValueError):judge.call('Q/answer_statements','same prompt',{'question':'原题','answer':'原回答'},reject)
  self.assertEqual(delegate.calls,[]);self.assertEqual(judge.reused,0)
 def test_failed_attempt_not_reused(self):
  delegate=Delegate();judge=ReusingJudge([{'task':'Q','input':{},'raw_response':'bad','attempt':0,'status':'error'}],delegate,Log())
  judge.call('Q','same',{},lambda p:None);self.assertEqual(len(delegate.calls),1)
if __name__=='__main__':unittest.main()
