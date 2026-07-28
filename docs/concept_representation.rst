Concept Representation Viewer
==============================

This page documents the "Concept emergence" mode of the ``viz`` app (the Concept Atlas),
which visualizes a complete run of the concept localization pipeline
(``experiments/concept_localization/``) for one concept and prompt template. Delta
extraction (``extract_deltas_generic.py``) computes the contrastive residual-stream
difference and its permutation null at every layer, and analysis
(``analyze.py``) projects that difference onto transcoder encoder directions to find the
features most aligned with it; this page covers what the resulting three plots mean and
how to read them together, not how they are computed.

What the page shows
---------------------

Each concept (for example, carry vs. no-carry in addition, or GCD = 7 vs. GCD ≠ 7) is a
binary distinction with matched positive/negative prompts that differ only in whether the
concept holds. For a chosen **anchor** — a specific token position and rank in the
prompt — the pipeline contrasts the positive and negative prompts and computes the mean
residual-stream difference at every layer,

.. math::

   \Delta_\ell = \mathrm{mean}(h_\ell^{\mathrm{pos}}) - \mathrm{mean}(h_\ell^{\mathrm{neg}})

together with a permutation null (the same statistic computed after randomly shuffling
which prompts are labeled positive/negative) and the transcoder-feature projections of
:math:`\Delta_\ell` at every layer. The page loads every anchor directory for a run and
lets you step through anchor positions, optionally animating the sequence, while three
panels update in sync for the selected anchor.

Read together, the three panels answer three separate questions: where does the concept
first become statistically distinguishable from chance (the trajectory panel), does it
stay encoded along one fixed direction as it propagates through the network or does the
network rotate it onto a different subspace (the cosine heatmap), and which individual
sparse features are actually carrying it at each layer (the feature-alignment panel).
None of these three questions is answered by the others: a concept can emerge early but
still rotate through several unrelated directions before the final layer, or stay on a
single stable direction while the specific transcoder features supporting it change from
layer to layer.

Transcoder feature alignment
-------------------------------

Each point is one transcoder feature, plotted at its layer (x-axis) and its combined
alignment score with :math:`\Delta_\ell` (y-axis, in :math:`[-1, 1]`). A positive score
(red) means the feature's decoder direction points the same way as the concept
difference at that layer — it supports the concept; a negative score (blue) means it
opposes it. The combined score blends decoder cosine similarity, encoder cosine
similarity, and the feature's mean activation on positive vs. negative prompts, all of
which are shown in the per-feature tooltip and in the inspector panel opened by clicking
a point. Lines drawn between features across layers mark pairs that co-activate with a
consistent sign across the underlying anchors; line thickness and opacity scale with how
often that co-activation holds (the support rate) and how consistent its sign is.

Clicking a feature opens its activation plot below the three panels — either a 1D bar
plot of positive/negative activation across a modular input sweep, or a 2D heatmap over
two operands — reusing the same top-k figures produced by the sweep pipeline rather than
re-deriving them.

Delta trajectory and permutation null
----------------------------------------

This panel tracks the norm of :math:`\Delta_\ell` across layers: a raw/peak-normalized
curve (solid) and an activation-normalized curve (dashed), plotted against a shaded band
showing the permutation null's mean plus one standard deviation. The null is the same
statistic computed under a label-shuffled version of the same prompts, so it reflects the
noise floor you would see even if the concept had no effect on the residual stream. A
layer where the trajectory rises clearly above the shaded band is a layer where the
concept's effect on the residual stream is distinguishable from that noise floor —
informally, the layer at which the concept **emerges**. The activation-normalized curve
additionally controls for the residual stream's overall growth in norm across depth, so a
late-layer rise in the raw curve alone is not mistaken for emergence.

Inter-layer direction similarity
-----------------------------------

This heatmap shows the cosine similarity between :math:`\Delta_\ell` and
:math:`\Delta_{\ell'}` for every pair of layers :math:`\ell, \ell'`. A cell near +1 means
the concept direction at those two layers is essentially the same direction, i.e. once
the concept appears it is carried forward largely unchanged. Values well below 1 —
especially blocks of low or negative similarity between early and late layers — indicate
that the direction **rotates**: the network re-encodes the same binary distinction along
a different subspace as it propagates, rather than simply amplifying a fixed direction.
This matters for interpretability and for steering: a feature or direction found at one
layer should not be assumed to transfer to another layer unless this heatmap shows high
similarity between them.

Concepts
----------

Every concept dataset lives under
``experiments/concept_localization/concept_datasets/`` and defines a matched
positive/negative predicate directly in code; the four bundled in the viz app's preset
buttons are:

.. list-table::
   :widths: 15 35 30 20
   :header-rows: 1

   * - Concept
     - Predicate
     - Example
     - Source
   * - ``carry``
     - pos: :math:`a_0+b_0 \ge 10`; neg: :math:`a_0+b_0 < 10` (ones digits :math:`a_0,
       b_0`, isolated so no other column carries)
     - ``127+236=363`` (7+6=13, carry) vs. ``123+236=359`` (3+6=9, no carry)
     - ``carry_dataset.py``
   * - ``gcd``
     - pos: :math:`a \equiv 0 \pmod 7` (gcd(a,7)=7); neg: :math:`a \bmod 7 \in
       \{1,\ldots,6\}` (gcd(a,7)=1)
     - ``gcd(70,7)=7`` vs. ``gcd(71,7)=1``
     - ``gcd_dataset.py``
   * - ``residue_class``
     - pos: :math:`a \equiv 0 \pmod 7`; neg: :math:`a \equiv r \pmod 7,\ r \in
       \{1,\ldots,6\}`
     - ``70 % 7 = 0`` vs. ``74 % 7 = 4``
     - ``residue_class_dataset.py``
   * - ``causal_direction``
     - pos: prompt states the true (cause → effect) order; neg: the same prompt with
       the two entities swapped
     - "Does fire lead to ash?" (yes) vs. "Does ash lead to fire?" (no)
     - ``causal_direction_dataset.py``

The remaining 13 concept datasets (physics, logic, and language predicates such as
``prime``, ``perfect_square``, ``syllogism``, or ``momentum_conservation``) follow the
same pattern and can be loaded by pointing the run-path field at their
``runs/concept_localization/<concept>/<concept>_T0`` output directory once a sweep has
been run for them. Each dataset module's own docstring and ``TEMPLATES`` dict is the
authoritative source for its exact predicate — the table above summarizes rather than
replaces it.

Loading a run
---------------

A run is loaded by path (or via the four preset buttons for the bundled example
concepts), and the loader discovers every ``anchor_rank*_pos*`` directory beneath it,
combining each anchor's ``results.json``, ``deltas.pt`` trajectories and cosine matrix,
permutation null, and feature projections into one consolidated view. An anchor is kept
even if some of these artifacts are missing; missing data is reported in the UI rather
than silently dropped. The visualiser's source lives in its own repository; see
`concept-run-visualization.md <https://gitlab.developers.cam.ac.uk/eid23/mechinterp-viz/-/blob/main/docs/concept-run-visualization.md>`_
there for the artifact discovery rules and the browser/server boundary (a static
deployment requires running ``export_concept_run.py`` ahead of time to produce a
single JSON file).
