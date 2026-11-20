"""Records what Hugging Face tokenizers produce for a corpus, to check the Go tokenizers against.

usage: golden.py TOKENIZER_JSON OUT_JSON

For every string: the normalised text, the ids without special tokens and the ids with them. For
pairs of strings: ids and segment ids under several truncation limits.
"""
import json
import random
import sys

from tokenizers import Tokenizer

src, out = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(src)
random.seed(7)

fixed = [
    "Hello world", "straße", "Müller", "Übergrößenträger", "Größe", "naïve café", "Å", "ÄÖÜ äöü ß",
    "Donaudampfschifffahrtsgesellschaftskapitän", "Das ist ein Test.", "Wie geht es dir?",
    "東京", "北京市", "こんにちは", "안녕하세요", "Привет мир", "مرحبا", "שלום", "สวัสดี", "नमस्ते",
    "😀", "👨‍👩‍👧", "🇩🇪", "a😀b", "https://example.com/a?b=c#d", "john.doe@example.com",
    "ＡＢＣ１２３", "ｈｅｌｌｏ", "ﬁﬂﬃ", "㎏ ℃ ㈱ ㌔", "①②③ ⑴", "ℌℍ ℕ ℝ", "𝐀𝐁𝐂 𝟏𝟐𝟑", "x²+y³=z¹⁰", "½ ¼ ¾",
    "é ä ô", "ñ", "Å", "À́", "ḳ́",
    "한", "한글", "क्ष", "नि", "ก็", "ปู่", "กำ",
    "a​b", "a‌b", "a‍b", "﻿bom", "a b", "a　b", "a b", "a b",
    "tab\there", "line\nbreak", "cr\r\nlf", "a\x00b", "a\x07b", "\x0b\x0c", "two  spaces", "  lead", "trail  ",
    "   ", "", " ", "\t", "a  b   c",
    "<s>", "</s>", "<pad>", "<unk>", "<mask>", "Das <mask> ist </s> ein <s> Test",
    "▁already", "a▁b", "\U00020000", "ꙮ", "ᚠᛇᚻ", "ǅ", "İstanbul", "ı", "ǆ", "ŉ", "ſ", "ʼn",
    "—", "…", "“quoted”", "it's", "don’t", "C++", "a/b", "Dr.", "U.S.A.", "COVID-19", "GPT-4o",
    "αβγ", "Ελληνικά", "ñandú", "Ὀδυσσεύς", "ǰ", "ΐ", "ﷺ", "ﻻ", "ﭏ", "ㅤ", "ᅟ",
]
fixed += [
    "I'm sure you're right, it's DON'T they'll we've She'd", "12345 6789 3.14159 1,000,000 0042", "x = 3 * (y + 4)\n\tif x > 0:\n\t\treturn x",
    "def f(a, b):\n    return a + b\n\n\nprint(f(1, 2))", "{\"name\": \"Müller\", \"age\": 42, \"tags\": [\"a\", \"b\"]}",
    "# Title\n\n* item one\n* item two\n\n```python\nprint('hi')\n```", "a\n\n\nb", "a \n b", "a  \n  b", "   leading", "trailing   ", "\n\n", " \n",
    "<|im_start|>user\nHallo<|im_end|>\n<|im_start|>assistant\n", "<|endoftext|>", "text<|im_end|>more", "Wie viele Beine hat eine Spinne?",
    "Ünïcödé ñ ß € £ ¥ © ® ™", "日本語のテキスト、中文文本，한국어 텍스트", "🙂🙃 emoji 👍🏽 and flags 🇩🇪🇫🇷", "\u200b\u00a0 odd\u2003spaces\u3000here",
    "https://example.com/path?x=1&y=2#frag", "user@example.com +49 170 1234567", "snake_case camelCase PascalCase kebab-case", "e.g. i.e. etc.; (a) [b] {c} <d>",
]
cases = list(fixed)
ranges = [(0x20, 0x24F), (0x300, 0x36F), (0x370, 0x6FF), (0x900, 0xDFF), (0xE00, 0xE7F), (0x1100, 0x11FF),
          (0x2000, 0x206F), (0x2100, 0x214F), (0x2460, 0x24FF), (0x3040, 0x30FF), (0x3300, 0x33FF),
          (0x4E00, 0x9FFF), (0xAC00, 0xAC80), (0xFB00, 0xFB4F), (0xFE30, 0xFE6F), (0xFF00, 0xFFEF),
          (0x1D400, 0x1D7FF), (0x1F300, 0x1F64F)]
for _ in range(1200):
    n = random.randint(1, 6)
    chars = []
    for _ in range(n):
        lo, hi = random.choice(ranges)
        c = chr(random.randint(lo, hi))
        if 0xD800 <= ord(c) <= 0xDFFF:
            c = "a"
        chars.append(c)
    cases.append("".join(chars))
words = ["der", "die", "das", "und", "ist", "nicht", "ein", "Haus", "Straße", "über", "schön", "Mädchen", "Fußball",
         "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "naïve", "über-cool", "e-mail", "3.14", "50%"]
for _ in range(400):
    cases.append(" ".join(random.choice(words) for _ in range(random.randint(1, 30))))

alphabet = [" ", " ", " ", "\t", "\n", "\n", "\r", "\u00a0", "\u2003", "a", "b", "Z", "1", "2", ".", ",", "!", "'", "s", "\u00e9", "\u4e2d"]
for _ in range(2000):
    cases.append("".join(random.choice(alphabet) for _ in range(random.randint(1, 14))))

records = []
for s in cases:
    enc = tok.encode(s, add_special_tokens=False)
    records.append({
        "s": s,
        "norm": tok.normalizer.normalize_str(s) if tok.normalizer is not None else s,
        "ids": enc.ids,
        "special": tok.encode(s, add_special_tokens=True).ids,
        "decoded": tok.decode(enc.ids, skip_special_tokens=False),
    })

pairs = []
for _ in range(300):
    a = " ".join(random.choice(words) for _ in range(random.randint(1, 40)))
    b = " ".join(random.choice(words) for _ in range(random.randint(1, 80)))
    pairs.append((a, b))
pairs += [("", "nur ein Dokument"), ("nur eine Frage", ""), ("a", "b"), ("Müller", "Straße")]
pair_records = []
for limit in (0, 8, 16, 32, 512):
    if limit:
        tok.enable_truncation(max_length=limit, strategy="longest_first")
    else:
        tok.no_truncation()
    for a, b in pairs:
        enc = tok.encode(a, b)
        pair_records.append({"a": a, "b": b, "limit": limit, "ids": enc.ids, "types": enc.type_ids})
    for s in cases[:200]:
        enc = tok.encode(s)
        pair_records.append({"a": s, "b": None, "limit": limit, "ids": enc.ids, "types": enc.type_ids})

json.dump({"cases": records, "pairs": pair_records}, open(out, "w"), ensure_ascii=False)
print(len(records), "strings;", len(pair_records), "encodings with truncation limits")
