"""One authorized restart of the verified banking index task, preserving caches."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time
from datetime import datetime

ROOT = Path('/home/ailab/data/HippoRAG/hipporag_local').resolve()
PID = 2755556
SOURCE = ROOT / 'outputs/server_native_full_c4_20261030-105942'
STAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
TARGET = ROOT / ('outputs/server_native_full_c4_e64_f32_' + STAMP)
BACKUP = ROOT / ('outputs/restart_backups/embedding_batches_' + STAMP)
os.chdir(ROOT)
proc = Path('/proc') / str(PID)
assert (proc / 'cwd').resolve() == ROOT, 'PID working directory changed'
cmd = (proc / 'cmdline').read_bytes().split(b'\0')
assert cmd[:4] == [b'python', b'-u', b'business.py', b'index'], 'PID command changed'
env = dict(item.split('=', 1) for item in (proc / 'environ').read_bytes().decode().split('\0') if '=' in item)
assert env.get('HIPPO_OUTPUT_DIR') == 'outputs/server_native_full_c4_20261030-105942', 'PID output changed'
assert not TARGET.exists() and not BACKUP.exists()
assert 'HIPPO_OPENIE_MAX_WORKERS' in (ROOT / 'business.py').read_text()
ast.parse((ROOT / 'business.py').read_text())
assert (ROOT / '.venv/bin/python').exists()
databases = list((SOURCE / 'banking/index/llm_cache').glob('*.sqlite'))
assert len(databases) == 1
source_db = databases[0]
corpus = SOURCE / 'banking/corpus.json'
manifest = SOURCE / 'banking/corpus.manifest.json'
assert corpus.is_file() and manifest.is_file()
corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
report = {'authorized_old_pid': PID, 'source_output': str(SOURCE), 'target_output': str(TARGET), 'openie_max_workers': 4, 'entity_batch_size': 64, 'fact_batch_size': 32, 'corpus_sha256': corpus_hash}
BACKUP.mkdir(parents=True)

def snapshot(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source_db.resolve().as_uri() + '?mode=ro', uri=True) as src:
        assert src.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        with sqlite3.connect(destination) as dst:
            src.backup(dst)
            assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            return dst.execute('SELECT COUNT(*) FROM cache').fetchone()[0]

report['prestop_cache_rows'] = snapshot(BACKUP / 'before_stop' / source_db.name)
(BACKUP / 'restart_report.json').write_text(json.dumps(report, indent=2))
print('PRESTOP_BACKUP_OK', report['prestop_cache_rows'], flush=True)
os.kill(PID, signal.SIGTERM)
for _ in range(100):
    if not proc.exists() or (proc / 'stat').read_text().split(') ', 1)[1].startswith('Z'):
        break
    time.sleep(0.1)
else:
    raise RuntimeError('Old task still running; no replacement started')
print('AUTHORIZED_OLD_PROCESS_EXITED', PID, flush=True)
report['final_cache_rows'] = snapshot(BACKUP / 'after_stop' / source_db.name)
assert report['final_cache_rows'] >= report['prestop_cache_rows']
new_db = TARGET / 'banking/index/llm_cache' / source_db.name
new_db.parent.mkdir(parents=True)
shutil.copy2(BACKUP / 'after_stop' / source_db.name, new_db)
with sqlite3.connect(new_db.resolve().as_uri() + '?mode=ro', uri=True) as conn:
    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert conn.execute('SELECT COUNT(*) FROM cache').fetchone()[0] == report['final_cache_rows']
for name in ('corpus.json', 'corpus.manifest.json'):
    shutil.copy2(SOURCE / 'banking' / name, TARGET / 'banking' / name)
assert hashlib.sha256((TARGET / 'banking/corpus.json').read_bytes()).hexdigest() == corpus_hash
shutil.copy2(ROOT / 'business.py', BACKUP / 'business.py')
shutil.copy2(ROOT / 'embedding_batches.py', BACKUP / 'embedding_batches.py')
shutil.copy2(ROOT / 'settings.py', BACKUP / 'settings.py')
shutil.copy2(ROOT / 'config.json', BACKUP / 'config.json')
shutil.copy2(ROOT / 'business_config.json', BACKUP / 'business_config.json')
shutil.copy2(manifest, BACKUP / 'corpus.manifest.json')
env.update(HIPPO_OPENIE_MAX_WORKERS='4', HIPPO_ENTITY_BATCH_SIZE='64', HIPPO_FACT_BATCH_SIZE='32', HIPPO_OUTPUT_DIR=str(TARGET.relative_to(ROOT)))
with (TARGET / 'index.log').open('xb') as log:
    log.write(('Restart with OpenIE=4, entity_batch_size=64, fact_batch_size=32; cache_rows=' + str(report['final_cache_rows']) + '\n').encode())
    log.flush()
    child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', 'business.py', 'index'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
report['new_pid'] = child.pid
report['started_at'] = datetime.now().astimezone().isoformat()
for path in (BACKUP / 'restart_report.json', TARGET / 'concurrency_restart.json'):
    path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
time.sleep(2)
if child.poll() is not None:
    raise RuntimeError('Replacement exited early; inspect new index.log')
print('NEW_PROCESS_RUNNING', child.pid, flush=True)
