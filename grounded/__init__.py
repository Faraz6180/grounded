"""grounded: free, offline reliability checks for RAG answers (no API keys, no dependencies)."""
import re, math, json, sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Optional

__version__ = "0.1.0"
__all__ = ["check", "evaluate", "Result", "Issue"]

STOP = set("a an the and or but if of to in on at by for with as is are was were be been being it its this that these those from into than then so such can could should would will may might do does did has have had not no yes you your we our they their he she his her me my which who whom what when where why how also about over under between per via there here very more most some any".split())
MONTHS = {m: i for i, m in enumerate("january february march april may june july august september october november december".split())}
NUM = re.compile(r"(?<!\w)[$€£]?\d[\d,]*(?:\.\d+)?%?")
CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SUPPORTED, WEAK = 0.55, 0.35

def _stem(w):
    for s in ("ingly", "edly", "ing", "ed", "es", "s", "ly"):
        if len(w) > len(s) + 3 and w.endswith(s):
            return w[: -len(s)]
    return w

def _toks(t):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 1 and w not in STOP]

def _num(x):
    x = re.sub(r"[$€£%,]", "", x)
    return x[:-2] if x.endswith(".0") else x

def _sentences(text):
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        s = s.strip()
        if not s:
            continue
        if out and re.fullmatch(r"(\s*\[\d+\]\s*)+\.?", s):
            out[-1] += " " + s
        else:
            out.append(s)
    return out

def _chunks(ctx):
    if isinstance(ctx, (list, tuple)):
        return {i + 1: str(c) for i, c in enumerate(ctx)}
    p = re.split(r"(?m)^\s*\[(\d+)\]\s*", ctx)
    if len(p) > 2:
        return {int(p[i]): p[i + 1].strip() for i in range(1, len(p) - 1, 2)}
    return {i + 1: x.strip() for i, x in enumerate(x for x in ctx.split("\n") if x.strip())}

@dataclass
class Issue:
    kind: str      # ungrounded | weak | citation | numeric | date | entity
    severity: str  # high | medium
    sentence: str
    detail: str

@dataclass
class Result:
    score: float
    confidence: str            # green | yellow | red
    explanation: str
    citation_accuracy: Optional[float]
    issues: List[Issue] = field(default_factory=list)
    sentences: List[dict] = field(default_factory=list)
    def to_dict(self): return asdict(self)

def check(question: str, context, answer: str) -> Result:
    ch = _chunks(context)
    ct = {k: set(_toks(v)) for k, v in ch.items()}
    n = max(len(ch), 1)
    df = Counter(t for s in ct.values() for t in s)
    w = lambda t: 1 + math.log((n + 1) / (df[t] + 1))
    cov = lambda st, k: sum(w(t) for t in st if t in ct[k]) / sum(w(t) for t in st)
    alltext = " ".join(ch.values())
    ref = (alltext + " " + question).lower()
    known = {_num(x) for x in NUM.findall(alltext + " " + question)}
    sents, issues, ok, tot = [], [], 0, 0
    for s in _sentences(answer):
        cites = [int(c) for g in CITE.findall(s) for c in re.split(r"\s*,\s*", g)]
        body = CITE.sub("", s).strip()
        st = set(_toks(body))
        if len(st) < 3 or body.endswith("?"):
            sents.append(dict(text=s, support=None, status="skip", flags=[])); continue
        sc = {k: cov(st, k) for k in ch}
        best = max(sc, key=sc.get) if sc else None
        sup = sc[best] if best else 0.0
        status = "supported" if sup >= SUPPORTED else "weak" if sup >= WEAK else "ungrounded"
        flags = []
        if status != "supported":
            issues.append(Issue("ungrounded" if status == "ungrounded" else "weak",
                "high" if status == "ungrounded" else "medium", body,
                f"Best match in context is {sup:.0%}" if best else "No context provided"))
        for c in cites:
            tot += 1
            if c not in ch:
                issues.append(Issue("citation", "high", body, f"Cites [{c}] but the context has no chunk [{c}]"))
            elif sc[c] >= WEAK:
                ok += 1
            elif sup >= SUPPORTED:
                issues.append(Issue("citation", "high", body, f"Citation drift: cites [{c}] but the support is in [{best}]"))
            else:
                issues.append(Issue("citation", "high", body, f"[{c}] does not support this claim"))
        for x in NUM.findall(body):
            if _num(x) not in known:
                flags.append(x); issues.append(Issue("numeric", "high", body, f"{x} does not appear in the context"))
        for i, e in enumerate(re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*", body)):
            low = e.lower()
            if body.startswith(e) and " " not in e and low not in MONTHS:
                continue
            if any(p not in ref for p in low.split()):
                flags.append(e); issues.append(Issue("date" if low in MONTHS else "entity", "medium", body, f"{e} does not appear in the context"))
        sents.append(dict(text=s, support=round(sup, 3), status=status, flags=flags))
    scored = [x for x in sents if x["support"] is not None]
    if not scored:
        return Result(0.0, "yellow", "No checkable claims found in the answer.", None, issues, sents)
    score = round(100 * sum(min(1, x["support"] / SUPPORTED) for x in scored) / len(scored), 1)
    bad = sum(x["status"] == "ungrounded" for x in scored)
    high = sum(i.severity == "high" for i in issues)
    cacc = round(100 * ok / tot, 1) if tot else None
    if score < 50 or bad / len(scored) >= 0.34 or high >= 3:
        conf, why = "red", f"{bad} of {len(scored)} claims are unsupported and {high} serious issue(s) were found. Do not show this answer to users."
    elif score < 75 or issues:
        conf, why = "yellow", f"Mostly supported, but {len(issues)} issue(s) need a look before you trust this answer."
    else:
        conf, why = "green", "Every claim traces back to the retrieved context and no suspicious facts were found."
    return Result(score, conf, why, cacc, issues, sents)

def evaluate(examples: List[dict]) -> dict:
    rs = [check(e["question"], e["context"], e["answer"]) for e in examples]
    n = max(len(rs), 1)
    return dict(n=len(rs), avg_score=round(sum(r.score for r in rs) / n, 1),
                red_pct=round(100 * sum(r.confidence == "red" for r in rs) / n, 1),
                by_confidence=dict(Counter(r.confidence for r in rs)),
                common_issues=Counter(i.kind for r in rs for i in r.issues).most_common(), results=rs)

if __name__ == "__main__":  # python -m grounded examples.json
    data = json.load(open(sys.argv[1]))
    rep = evaluate(data)
    print(f"{rep['n']} examples | avg groundedness {rep['avg_score']}% | red {rep['red_pct']}% | issues {rep['common_issues']}")
