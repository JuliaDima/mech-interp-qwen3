Concept Geometry via Contrastive Residual-Stream Analysis
============================================================

Attribution graphs, documented in :doc:`attribution_graph` and :doc:`methodology`,
reproduce the gradient-based circuit-discovery methodology of Lindsey et al. (2025) for a
handful of hand-picked prompts: two-digit addition and multilingual antonym completion.
That approach is guided by a prior hypothesis about the circuit under study, and its cost
grows quickly with the number of prompts, layers, and token positions considered. This
page documents a complementary method, implemented under
``experiments/concept_localization/``, that instead starts from nothing but a contrastive
definition of a concept and asks, empirically, where in the network the concept first
becomes distinguishable from chance, whether it is carried by a single stable direction or
several as it propagates through depth, and which individual transcoder features align
with it. It is the mathematical and statistical framework behind the "Concept emergence"
mode of the visualiser (:doc:`concept_representation` covers how to read the resulting
plots; this page covers how the quantities they show are defined and estimated). Three
arithmetic concepts — carry detection, GCD divisibility, and residue-class membership —
are analysed in full depth to study how this geometry changes with problem structure, and
are used throughout as running examples.

Contrastive concept datasets
------------------------------

Each concept is defined by a matched pair of prompts :math:`(x^+, x^-)` that differ in
exactly one property: the positive prompt contains the target concept, and the negative
prompt removes it while preserving syntax, length, numeric scale, and answer format as
closely as possible. Holding everything but the concept fixed prevents the mean
residual-stream difference between the two prompt sets from being dominated by incidental
token changes rather than by the concept itself. A given prompt template is retained for
analysis only once Qwen3-4B reaches at least 10% first-token accuracy on it, which
verifies that the model minimally respects the expected output format before its internal
representations are examined.

In code, a concept dataset is a small, self-contained Python module under
``experiments/concept_localization/concept_datasets/``. Every module follows the same
shape: a ``TEMPLATES`` dictionary mapping surface-form names (``T0``, ``T1``, ``T2``, …) to
one or more phrasings of the same underlying predicate, and a ``generate_<concept>_pairs()``
function that enumerates or samples matched instances and returns a list of
``ConceptPair`` objects,

.. code-block:: python

   @dataclass
   class ConceptPair:
       prompt_pos: str
       prompt_neg: str
       label_pos: str = ""
       label_neg: str = ""
       template: str = "T0"
       meta: dict = field(default_factory=dict)
       predict_pos: str = ""
       predict_neg: str = ""

Keeping several templates per concept lets the pipeline check that a discovered direction
reflects the concept itself rather than one particular surface form: the same predicate is
asked as a direct calculation (``T0``), a yes/no question (``T1``), and a third rephrasing
(``T2``), and a genuine concept direction should generalise across all of them. The
``meta`` dictionary carries whatever per-pair bookkeeping the concept needs (operand
values, digit positions, entity words) for later stratification of results, for instance by
digit pair or by residue class.

The library currently spans eighteen such datasets, grouped by the kind of predicate they
isolate:

* **Arithmetic and number-theoretic** — ``carry`` (units-digit carry in three-digit
  addition), ``gcd`` (divisibility of :math:`a` by 7), ``residue_class`` (membership of
  :math:`a` in residue class 0 mod 7), ``prime`` (primality of a three-digit integer),
  ``perfect_square`` (whether :math:`n` is a perfect square), ``decimal_termination``
  (whether :math:`1/n` has a terminating decimal expansion, i.e. :math:`n = 2^a5^b`), and
  ``geometric_series`` (whether a ratio :math:`p/q` gives a convergent series, :math:`|r| <
  1`).
* **Logical and relational** — ``syllogism`` (valid Barbara syllogism vs. an undistributed
  middle term), ``negation_scope`` (truth of a negated comparison, "m is not greater than
  n"), ``transitive_ordering`` (transitive chaining of an ordering relation), and
  ``balanced_parentheses`` (whether a sequence of space-separated parentheses is balanced).
* **Physical-reasoning analogues expressed in natural language** — ``conservation``
  (whether a bounce height is physically consistent with energy conservation),
  ``momentum_conservation`` (whether post-collision velocities conserve momentum for unit
  masses), ``doppler_shift`` (whether an approaching source raises or lowers observed
  pitch), ``wave_interference`` (constructive vs. destructive interference from a path
  difference), ``dot_product_sign`` (whether two vectors form an acute or obtuse angle),
  and ``triangle_inequality`` (whether three side lengths can form a triangle).
* **Causal framing** — ``causal_direction`` (whether a stated cause-effect order matches
  the physically plausible direction, e.g. "does fire lead to ash?" against its reversal).

Every one of these datasets can be run through the same pipeline (delta extraction,
transcoder projection, causal validation) without any concept-specific code elsewhere in
the pipeline; only the contrastive predicate and its surface templates differ. The three
concepts examined in full geometric depth here are summarised in the table below, each
isolating a single binary predicate at fixed numeric scale.

.. list-table:: Contrastive prompt sets for the three concepts studied in depth. Accuracy
   and mean predicted probability :math:`\bar p` are measured on the first answer token.
   :widths: 12 30 18 18 8 8 8
   :header-rows: 1

   * - Concept
     - Template
     - Positive
     - Negative
     - :math:`N`
     - Acc.
     - :math:`\bar p`
   * - Carry
     - ``what is {a}+{b}? Answer:``
     - :math:`a_0 + b_0 \geq 10`
     - :math:`a_0 + b_0 < 10`
     - 99
     - 100%
     - 0.92
   * - GCD
     - ``the gcd of {a} and 7 is:``
     - :math:`\gcd(a,7) = 7`
     - :math:`\gcd(a,7) = 1`
     - 100
     - 86%
     - 0.50
   * - Residue class
     - ``calc: {a}%7=``
     - :math:`a \equiv 1 \pmod 7`
     - :math:`a \not\equiv 1 \pmod 7`
     - 100
     - 12%
     - 0.12

The residue-class template's low first-token accuracy (12%) is itself informative: it
already hints, before any residual-stream analysis, that this seven-way classification is
harder for the model to resolve at the first output token than the two binary tasks above
it — a pattern the geometric analysis below confirms from a different angle.

Residual-stream concept delta
--------------------------------

Let :math:`h_{l,t}(x) \in \mathbb{R}^d` be the residual-stream vector at layer :math:`l`
and token position :math:`t` for prompt :math:`x`. Rather than committing to a single token
position in advance, the pipeline scans a set of candidate anchor positions, since the
model processes different aspects of a computation at different tokens. For a fixed anchor
:math:`t^*`, the layer-wise concept delta is

.. math::

   \delta_l = \frac{1}{N}\sum_{i=1}^N h_{l,t^*}(x_i^+) - \frac{1}{N}\sum_{i=1}^N h_{l,t^*}(x_i^-)
            = \mathbb{E}[h_{l,t^*}(x^+)] - \mathbb{E}[h_{l,t^*}(x^-)].

Under the linear representation hypothesis (Park et al., 2023), :math:`\delta_l`
approximates a concept axis whenever the positive and negative prompts differ in only one
property and the concept is linearly separable at that layer. The trajectory
:math:`(\delta_l)_{l=0}^{L-1}` traces how this contrast changes as it passes through the
network's 36 layers.

Layer and anchor metrics
----------------------------

The most direct measure of concept strength at layer :math:`l` is the delta norm,
:math:`D_l = \lVert \delta_l \rVert_2`. Because residual-stream norms tend to grow with
depth regardless of any particular concept, :math:`D_l` is divided by the mean activation
norm at the same layer,

.. math::

   \mu_l^{\mathrm{act}} = \mathbb{E}_x\!\left[\lVert h_{l,t^*}(x)\rVert_2\right],
   \qquad
   D_l^{\mathrm{act}} = \frac{D_l}{\mu_l^{\mathrm{act}} + \varepsilon}.

To compare trajectories across anchors that differ in absolute scale, each is further
divided by its own maximum,

.. math::

   \tilde D_l^{\mathrm{act}} = \frac{D_l^{\mathrm{act}}}{\max_j D_j^{\mathrm{act}} + \varepsilon} \in [0, 1].

High :math:`\tilde D_l^{\mathrm{act}}` marks a layer where the concept delta is large
relative to the residual stream's own scale at that depth; different anchors within the
same concept can show this concentrated at a single layer or spread broadly across many.

A separate question from *how large* the delta is at each layer is *whether it points in
the same direction* from one layer to the next. The inter-layer cosine similarity matrix

.. math::

   C_{l,m} = \frac{\delta_l \cdot \delta_m}{\lVert \delta_l \rVert_2 \lVert \delta_m \rVert_2 + \varepsilon}

answers this directly: a block of high similarity indicates one stable concept axis
carried forward largely unchanged, a two-block structure with a sharp boundary indicates a
single representational rotation at a specific layer, and a diffuse diagonal indicates
gradual, continuous rotation with no privileged direction. Panickssery et al. (2024)
report a diffuse-diagonal pattern for contrastive steering vectors in Llama 2; the
residue-class heatmap in this codebase shows a comparably diffuse pattern, while carry and
GCD instead show sharper block structure, discussed further below.

Null permutation test
-------------------------

A contrastive scan can produce an apparent direction even when the positive/negative
labelling carries no real concept information, simply because any two disjoint sets of
prompts differ somewhat in their mean activations. To estimate this background, the
positive/negative labels are repeatedly permuted at random and the delta trajectory is
recomputed for each permutation :math:`\pi`, giving :math:`\delta_l^\pi` and, for whichever
scalar statistic :math:`q_l` is under test (typically :math:`D_l^{\mathrm{act}}`), a null
distribution

.. math::

   \{q_l(\delta^\pi) : \pi = 1, \ldots, P\}.

An observed statistic clearly above this null range indicates concept-specific signal
beyond what generic prompt-set variation produces on its own; a statistic within or below
the null range indicates that the particular binary contrast used does not, at that
anchor, produce a discriminative direction — which is not the same as saying the model
performs no relevant computation at that position, only that this contrast does not
recover it linearly there.

Causal validation
---------------------

Correlational evidence — a large, stable, above-null delta — does not by itself establish
that the residual stream at that layer and position causally determines the model's
output. Two complementary causal tests address this.

**Activation patching.** For each pair :math:`(x_i^+, x_i^-)` and layer :math:`l`, the
positive residual vector at the anchor is spliced into the negative prompt's forward pass,
:math:`h_{l,t^*}(x_i^-) \leftarrow h_{l,t^*}(x_i^+)`. Writing :math:`m(x) = \ell_{\mathrm{pos}}(x)
- \ell_{\mathrm{neg}}(x)` for the answer logit margin at the final prediction position, the
activation patching score is

.. math::

   S_l^{\mathrm{patch}} = \frac{1}{N}\sum_{i=1}^N \left[m(\tilde x_{i,[l]}^-) - m(x_i^-)\right],

where :math:`\tilde x_{i,[l]}^-` denotes the negative prompt with its layer-:math:`l`
anchor activation replaced by the positive prompt's. A positive score means the patched
state moves the model's output toward the positive answer, and is direct evidence that the
residual stream at that layer and position is *sufficient* to flip the model's behaviour.

**Gradient-dot-delta.** As a cheaper first-order approximation to patching (avoiding a
full forward pass per pair and layer), each layer is instead scored by the inner product of
the output-margin gradient with the mean concept delta,

.. math::

   S_l^{\mathrm{grad}} = \frac{1}{N}\sum_{i=1}^N \nabla_{h_{l,t^*}} m(x_i^-) \cdot \delta_l.

This follows from a first-order Taylor expansion of the patched margin around the
unpatched negative activation,

.. math::

   m(\tilde x_{i,[l]}^-) \approx m(x_i^-) + \nabla_{h_{l,t^*}} m(x_i^-) \cdot \left(h_{l,t^*}(x_i^+) - h_{l,t^*}(x_i^-)\right),

so that when :math:`S_l^{\mathrm{grad}}` and :math:`S_l^{\mathrm{patch}}` agree closely, the
model's output is well approximated as locally linear along the concept direction at that
layer; when they diverge, either higher-order effects dominate or the mean delta is a poor
stand-in for the individual pairwise differences it averages over.

Anchor selection
-------------------

Scanning every token position with the full causal validation above is unnecessary, since
most positions in a prompt carry no concept-relevant signal at all. Anchors are instead
selected in two stages: first, all candidate positions are ranked by the mean inter-layer
cosine similarity of :math:`\delta_l` (Equation for :math:`C_{l,m}` above) and the top six
are retained; second, these six are re-ranked by the combined score

.. math::

   S = \tilde D^{\mathrm{act}} + \max(0,\, D^{\mathrm{act}} - \mathrm{null}) + S^{\mathrm{patch}},

which rewards a position where the delta is large relative to the residual stream's scale,
clearly exceeds its own permutation null, and is causally sufficient under activation
patching. The six anchors ranked by this score are what the visualiser's anchor stepper
walks through for a given concept and template.

Feature-direction interventions
-----------------------------------

Each transcoder feature has a decoder direction it writes into the residual stream, and the
pipeline identifies which features are most aligned with a concept's delta direction (see
:doc:`concept_representation` for how those alignment scores are shown in the visualiser).
Ranking by alignment alone is correlational; the pipeline also tests projected features
causally by modifying the raw MLP output directly along their decoder directions, rather
than replacing the whole MLP output with its transcoder reconstruction (which measurably
degrades first-token accuracy on its own and would confound the test).

**Ablation** removes a feature's contribution from the MLP output it feeds into,

.. math::

   y_l(x) \leftarrow y_l(x) - (1 - \alpha)\, z_{l,f}(x)\, w_{l,f}^{\mathrm{dec}},

with :math:`\alpha = 0` giving full ablation and :math:`\alpha = 1` leaving the model
unchanged. **Injection** instead adds a feature's decoder direction at a fixed magnitude
regardless of whether the feature activates naturally on that prompt,

.. math::

   y_l(x) \leftarrow y_l(x) + \delta_f\, w_{l,f}^{\mathrm{dec}},

where :math:`\delta_f` is set to the feature's mean natural activation on positive prompts,
:math:`\bar a_f^+ = \mathbb{E}[z_{l,f}(x^+)]`, or a fixed scale chosen for the experiment.

For carry, joint ablation of the ten most-aligned features at the ones\ :sub:`b` anchor
reduces accuracy from 100% to 47.8% on the 23 prompts that activate them, while ablating
any single one of the ten has no measurable effect — evidence of functional redundancy
among these features rather than a single load-bearing unit. The resulting prediction
shifts fall between :math:`-8` and :math:`-11`, close to the units digit of the second
operand, consistent with these features encoding part of the digit-level value entering
the running sum. Injecting four decoder directions at fixed magnitude :math:`\delta = 5.0`
into 198 carry-negative prompts reduces accuracy from 100% to 22.2%, shifting 154
predictions by :math:`+7` or :math:`+8` — as though a carry were present despite
:math:`a_0 + b_0 < 10`.

GCD and residue-class geometry
----------------------------------

The same framework applied to GCD divisibility and to residue-class membership shows how
the geometry changes with problem structure rather than staying fixed across tasks.

For **GCD divisibility by seven**, several anchors exceed the permutation-null band,
confirming a genuine binary component to the task, and the inter-layer cosine matrix shows
a sharp block transition at layer 10 for the ones\ :sub:`a` anchor. This is also where
projected features begin exhibiting prime and co-prime patterns with respect to seven —
notably, at a position reached *before* the model has read the operator token (the digit
7) in the prompt. This ordering indicates that Qwen3-4B already carries feature detectors
for divisibility by seven for arbitrary integers, independent of this specific prompt's
operator, and that the contrastive design of this experiment is what surfaces them. The
effect is loosely analogous to predictive preactivation in human language comprehension,
where contextual expectations can probabilistically activate representations of likely
upcoming input before that input is actually encountered (Kuperberg and Jaeger, 2016) — here
the anticipated content is computationally specific (a divisibility relation) rather than a
lexical identity.

For **residue-class membership modulo seven** (:math:`a \equiv 1 \pmod 7` against its
complement), no anchor's delta trajectory consistently exceeds the permutation-null band.
This is the clearest geometric signature of the three concepts studied: unlike carry and
GCD, residue-class membership does not resolve into one dominant binary direction under
this method, plausibly because the underlying task is a seven-way classification collapsed
post hoc into a single class versus its complement, rather than a task the model
represents as binary in the first place. Consistent with this, many of the most strongly
projected features at these anchors turn out to be divisibility-by-seven detectors — the
same detectors surfaced by the GCD analysis — rather than detectors specific to residue
class 1, further tying the two arithmetic concepts to a shared underlying representation of
divisibility by seven rather than two independent concept directions.

.. figure:: _static/images/gcd_concept_emergence.gif
   :alt: Anchor-by-anchor emergence of the GCD divisibility direction
   :align: center

   The GCD run stepped anchor by anchor: the delta trajectory and permutation null (left),
   the transcoder features most aligned with the divisibility direction at that anchor and
   their per-digit activation profiles (middle), and the inter-layer cosine matrix showing
   where the block transition at layer 10 appears and how stable the direction is
   afterward (right). Produced by the same pipeline and equations described above; see
   :doc:`concept_representation` for how to read each panel.

Further reading
-------------------

* :doc:`concept_representation` — how to read the visualiser's three synchronised panels
  for a loaded concept run, and how to load a run for any of the eighteen concept datasets.
* :doc:`attribution_graph` and :doc:`methodology` — the attribution-graph reproduction of
  Lindsey et al. (2025), used elsewhere in this project for hypothesis-driven circuit
  discovery on individual prompts rather than the hypothesis-light, dataset-level method
  described on this page.
* `experiments/concept_localization/concept_datasets/
  <https://github.com/JuliaDima/mech-interp-qwen3/tree/main/experiments/concept_localization/concept_datasets>`_
  — the full, growing set of concept dataset modules.
