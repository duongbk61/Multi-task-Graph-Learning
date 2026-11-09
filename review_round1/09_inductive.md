# [A9] Inductive vs Transductive (standard config: rule-free + fix)

Inductive protocol: at TRAIN time the test nodes and ALL edges incident to them are removed
from the graph (encoder never sees test nodes / cannot sample them as neighbours); at TEST time
the full graph is restored and test nodes are forwarded once. Transductive = standard full-graph.

| Setting | Ponzi Macro-F1 | Ponzi PR-AUC | Phish Macro-F1 | Phish PR-AUC |
|---|---|---|---|---|
| Transductive | 81.24±1.69 (n=15) | 68.93±2.39 | 97.21±0.69 | 99.59±0.09 |
| Inductive | 81.74±0.91 (n=5) | 69.22±2.58 | 97.14±0.58 | 99.64±0.08 |

## Conclusion (positive, defensible)
Inductive performance ≈ transductive — **no degradation** when test nodes are entirely unseen during
training (Ponzi even +0.5). The model detects fraud on **new/unseen accounts**, not by transductive
memorisation. PR-AUC on Phish stays 99.6 in the inductive setting. This directly answers the reviewer's
inductive request with a favourable result and is a genuine strength of the approach.

(Edges removed e.g. Ponzi EOA-trans-CA 3,337,998 → 3,294,306; the test-incident edges are dropped.)
