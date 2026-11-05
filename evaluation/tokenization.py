"""Deterministic Chinese-character and Latin/numeric tokenization for ROUGE."""
import re
import unicodedata

TOKENIZER_VERSION = "zh-char-alnum-v1"
TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0003134f]"
    r"|[a-z][a-z0-9]*(?:[.'-][a-z0-9]+)*|[0-9]+(?:\.[0-9]+)*(?:%|‰)?"
)


class ChineseRougeTokenizer:
    def tokenize(self, text):
        return TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())


def rouge_l(answer, reference):
    """ROUGE-L F1 using the standard LCS formula and the explicit tokenizer."""
    tokenizer = ChineseRougeTokenizer()
    predicted, expected = tokenizer.tokenize(answer), tokenizer.tokenize(reference)
    if not predicted or not expected:
        return 0.0
    # One-row dynamic programming, O(n*m) time and O(min(n,m)) memory.
    if len(expected) > len(predicted):
        predicted, expected = expected, predicted
    previous = [0] * (len(expected) + 1)
    for token in predicted:
        current = [0]
        for j, other in enumerate(expected, 1):
            current.append(previous[j - 1] + 1 if token == other else max(current[-1], previous[j]))
        previous = current
    return 2.0 * previous[-1] / (len(predicted) + len(expected))
