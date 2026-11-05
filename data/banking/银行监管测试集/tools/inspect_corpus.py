import hashlib
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '银行监管测试集'

def load_doc(path):
    raw = path.read_text(encoding='utf-8-sig')
    def meta(key):
        m = re.search(r'id="' + key + r'"[^>]*>(.*?)</span>', raw, re.S)
        return html.unescape(re.sub('<[^>]+>', '', m.group(1))).strip() if m else ''
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', raw, re.S)
    title = html.unescape(re.sub('<[^>]+>', '', title_match.group(1))).strip() if title_match else path.stem.split('__', 1)[1]
    text = re.sub(r'<!--.*?-->|<script\b[^>]*>.*?</script>|<style\b[^>]*>.*?</style>', '', raw, flags=re.S | re.I)
    text = re.sub(r'</(?:p|div|tr|h[1-6])\s*>|<br\s*/?>', '\n', text, flags=re.I)
    text = html.unescape(re.sub('<[^>]+>', '', text)).replace('\x00', '\ufffd')
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return {'doc_id': path.name.split('__')[0], 'title': title,
            'path': path.relative_to(ROOT).as_posix(),
            'validity': meta('cchcnvaliditystatusid'),
            'publication_date': meta('cchcnpromulgatingdate'),
            'effective_date': meta('cchcneffectivedate'),
            'document_number': meta('cchcndocumentnumber'),
            'agency': meta('cchcnpromulgatingagencyid'),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'lines': lines}

