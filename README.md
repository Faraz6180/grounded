# Grounded...........
Free, offline reliability checks for RAG systems and simple agents. No API keys, no dependencies.

    pip install 
```python
from grounded import check
r = check(question="When was the tower built?",
          context="[1] The tower was completed in 1889.",
          answer="The tower was completed in 1901 [1].")
print(r.score, r.confidence)
for i in r.issues: print(i.kind, i.detail)
```
Batch: `python -m grounded examples.json` (list of `{question, context, answer}`).

Detects ungrounded claims, citation drift, numeric/date/entity hallucination, and low overall support using lexical evidence matching. It is a fast first line of defence, not a semantic judge: paraphrase-heavy answers may score lower than they deserve.
