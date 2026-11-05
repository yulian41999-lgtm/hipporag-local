import sys,json,unittest,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from settings import load_settings
load_settings()
from hipporag.information_extraction.openie_openai import OpenIE
class Fake:
 def __init__(self,replies):
  self.replies=iter(replies); self.calls=[]; self.global_config=types.SimpleNamespace(response_format=None)
 def infer(self,**kwargs):
  self.calls.append(kwargs); value=next(self.replies)
  if isinstance(value,Exception): raise value
  response,hit=value
  return response,{'prompt_tokens':10,'completion_tokens':5,'finish_reason':'stop'},hit
GOOD='{"triples":[["甲银行","位于","北京"]]}'
class Tests(unittest.TestCase):
 def run_case(self,replies):
  fake=Fake(replies); result=OpenIE(fake).triple_extraction('test','甲银行位于北京',['甲银行','北京']); return fake,result
 def test_good_request_unchanged(self):
  fake,r=self.run_case([(GOOD,True)])
  self.assertEqual(len(fake.calls),1); self.assertEqual(set(fake.calls[0]),{'messages','max_new_tokens'})
  self.assertNotIn('error',r.metadata); self.assertTrue(r.metadata['cache_hit'])
  self.assertEqual(r.metadata['billable_prompt_tokens'],0)
 def test_strict_retry_and_accounting(self):
  fake,r=self.run_case([('说明文字',True),(GOOD,False)])
  self.assertEqual(len(fake.calls),2); self.assertEqual(fake.calls[1]['response_format'],{'type':'json_object'})
  self.assertNotEqual(fake.calls[0]['messages'],fake.calls[1]['messages'])
  self.assertEqual(r.metadata['prompt_tokens'],20); self.assertEqual(r.metadata['billable_prompt_tokens'],10)
  self.assertEqual(r.metadata['billable_completion_tokens'],5); self.assertFalse(r.metadata['cache_hit'])
  self.assertEqual(len(r.triples),1)
 def test_retry_limit(self):
  fake,r=self.run_case([('bad',True),('bad',False),('bad',False)])
  self.assertEqual(len(fake.calls),3); self.assertIn('error',r.metadata)
 def test_malformed_strict_triple_retried(self):
  fake,r=self.run_case([('bad',True),('{"triples":[["missing object","relation"]]}',False),(GOOD,False)])
  self.assertEqual(len(fake.calls),3); self.assertNotIn('error',r.metadata)
 def test_empty_explicit_array_is_valid(self):
  fake,r=self.run_case([('bad',True),('{"triples":[]}',False)])
  self.assertNotIn('error',r.metadata); self.assertEqual(r.triples,[])
 def test_recovery_cache_reused(self):
  fake,r=self.run_case([('bad',True),(GOOD,True)])
  self.assertTrue(r.metadata['cache_hit']); self.assertEqual(r.metadata['billable_prompt_tokens'],0)
 def test_network_failure_is_not_silently_accepted(self):
  fake,r=self.run_case([RuntimeError('network unavailable')])
  self.assertIn('error',r.metadata); self.assertEqual(len(fake.calls),1)
if __name__=='__main__': unittest.main()
