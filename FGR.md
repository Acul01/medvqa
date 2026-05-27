# Additional Extension: Multimodal Rationale Faithfulness Evaluation

## Motivation

A major limitation of current reasoning-based Medical VQA systems is that generated reasoning traces may be:
- linguistically plausible,
- internally coherent,
- and predictive of the correct answer,
while still being weakly grounded in the underlying medical image.

This creates an important distinction between:

- textual reasoning consistency,
and
- visually grounded reasoning faithfulness.

To explicitly evaluate this distinction, the proposed framework introduces an additional multimodal rationale faithfulness evaluation stage.

The central idea is:

> A faithful explanation should not only support the correct answer linguistically, but should also depend on the clinically relevant visual evidence.

---

# Rationale Faithfulness Evaluation Settings

Given:
- image \(x\),
- question \(q\),
- generated reasoning trace \(r\),

the framework evaluates the generated rationale under multiple visual evidence conditions.

---

## Setting A — Explanation Only

The model receives:
- question,
- reasoning trace,

but no image.

\[
model(q, r)
\rightarrow y
\]

This evaluates:

- linguistic self-consistency,
- reasoning plausibility,
- answer predictiveness of the textual rationale.

Interpretation:
- high performance here indicates that the reasoning itself strongly implies the answer,
- but does not prove visual grounding.

---

## Setting B — Explanation + Full Image

The model receives:
- full image,
- question,
- reasoning trace.

\[
model(x, q, r)
\rightarrow y
\]

This evaluates:
- multimodal consistency between reasoning and image evidence.

This serves as the reference condition.

---

## Setting C — Explanation + Relevant Region Only

The model receives:
- only the high-importance visual region,
- question,
- reasoning trace.

\[
model(x_{relevant}, q, r)
\rightarrow y
\]

The relevant region is determined using the perturbation-based importance scores from Stage 2.

This evaluates:
- visual evidence sufficiency,
- grounding sufficiency of the rationale.

Interpretation:
- if the explanation remains correct and confident,
  the selected region likely contains sufficient visual evidence.

---

## Setting D — Explanation + Irrelevant Region Only

The model receives:
- only low-importance regions,
- question,
- reasoning trace.

\[
model(x_{irrelevant}, q, r)
\rightarrow y
\]

This evaluates:
- shortcut reliance,
- robustness against visually irrelevant evidence,
- language-prior dominance.

Interpretation:
- if the model still produces highly confident correct answers,
  the reasoning may be weakly grounded and dominated by textual priors.

---

## Setting E — Explanation + Relevant Region Masked

The model receives:
- the original image,
- but with the relevant region removed.

\[
model(x_{\setminus relevant}, q, r)
\rightarrow y
\]

This evaluates:
- visual necessity of the selected evidence region.

Expected behavior:
- confidence should decrease,
- entropy should increase,
- reasoning should become unstable,
- or the answer should change.

---

# Proposed Metrics

## Language Consistency

Measures whether the explanation alone predicts the answer.

\[
S_{lang}
=
correct\_rate(q,r)
\]

---

## Visual Sufficiency

Measures whether the relevant region alone preserves the correct answer.

\[
S_{suff}
=
correct\_rate(x_{relevant}, q, r)
\]

---

## Visual Necessity

Measures whether removing the relevant region damages answer generation.

\[
S_{nec}
=
correct\_rate(x,q,r)
-
correct\_rate(x_{\setminus relevant}, q, r)
\]

Possible extensions:
- confidence drop,
- entropy increase,
- token probability shift.

---

## Shortcut Risk Score

Measures whether irrelevant regions are sufficient for stable answer generation.

\[
S_{shortcut}
=
correct\_rate(x_{irrelevant}, q, r)
\]

High values indicate:
- weak grounding,
- shortcut learning,
- or language-prior dominance.

---

# Combined Rationale Faithfulness Score

The final rationale faithfulness score combines:
- visual sufficiency,
- visual necessity,
- and shortcut robustness.

\[
S_{faithful}
=
S_{suff}
+
S_{nec}
-
S_{shortcut}
\]

Optional uncertainty-aware variant:

\[
S_{faithful}
=
correct\_rate_{relevant}
\cdot
confidence_{relevant}
\cdot
(1-entropy_{relevant})
-
correct\_rate_{irrelevant}
\cdot
confidence_{irrelevant}
\]

---

# Interpretation of Outcomes

| Observation | Interpretation |
|---|---|
| Explanation-only performs well | Reasoning is linguistically predictive |
| Relevant-region-only performs well | Explanation is visually grounded |
| Irrelevant-region-only performs well | Shortcut or language-prior reliance |
| Relevant-region-masked causes instability | Evidence region is necessary |
| Relevant-region-masked has little effect | Weak causal grounding |
| Confidence remains high after evidence removal | Potential overconfidence / miscalibration |

---

# Relation to Faithful Explainability

The proposed evaluation moves beyond:
- post-hoc attention visualization,
- GradCAM plausibility,
- or purely textual reasoning evaluation.

Instead, it evaluates whether:
- generated reasoning traces,
- answers,
- and uncertainty estimates

behaviorally depend on clinically relevant visual evidence.

This provides a substantially stronger approximation of faithful multimodal reasoning.

