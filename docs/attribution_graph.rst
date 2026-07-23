Attribution Graph Viewer
========================

This page documents the interactive attribution graph viewer (the "Attribution graph"
mode of the ``viz`` app) and the computation it visualizes. For the underlying gradient
methodology, see :doc:`methodology`; for the exact JSON schema the viewer consumes, see
``viz/docs/visualization-json-format.md`` in the repository.

What an attribution graph is
-----------------------------

An attribution graph is a directed graph over one forward pass of the model on a single
prompt. Every node is either:

* a **token** node — one of the prompt's input tokens;
* a **feature** node — a transcoder (sparse autoencoder) feature active at a given layer
  and token position;
* an **error** node — the residual between an MLP layer's true output and its transcoder
  reconstruction, kept so that attribution mass which the transcoder does not explain is
  not silently dropped;
* a **logit** node — one of the salient output tokens (the smallest set of tokens whose
  probabilities cover most of the model's output distribution at the final position).

Every edge carries an ``attribution_score``: the gradient-based contribution of the
source node to the target node, computed with the linearized backward pass described in
:doc:`methodology` (attention detached, RMSNorm linearized, gradients taken from a
demeaned unembedding direction so that only the relative preference for the target logit
is measured). Reconstructive patching makes feature-to-feature edges possible by routing
the backward pass through the transcoder bottleneck, so a downstream feature's activation
can be attributed to the upstream features that fed it.

Because a raw graph over every feature and layer would contain thousands of nodes, the
graphs loaded by the viewer have already been pruned: nodes below a significance
threshold (as a fraction of total attribution) are removed, and only the edges needed to
explain the surviving nodes are kept.

Reading the layout
-------------------

The viewer arranges nodes on a fixed grid rather than a force-directed layout, so that
position is always meaningful:

* **Columns are layers.** The leftmost column holds the prompt's token nodes; each
  subsequent column is one transformer layer; the rightmost column holds the logit
  nodes for the salient output tokens.
* **Rows are token positions.** Each row corresponds to one position in the prompt, so a
  feature's row shows which token it fired on.
* **Color encodes node type and strength.** Token nodes are green. Feature nodes (and the
  aggregate "bucket" nodes used when many features share a layer/position cell) are
  shaded blue-to-navy by how many underlying features they aggregate. Error nodes are
  red. Logit nodes are shaded green-to-dark-green by predicted probability, so the
  model's top guess is easy to spot at a glance.
* **Edges** are thin grey lines by default; the top edges by ``|attribution_score|`` are
  the ones actually drawn, since a full graph's edge count is unusable at a glance.

Clicking a node opens the **Inspector** panel, which shows:

* the node's layer, token position, token/logit string, activation, and total
  attribution;
* its **incoming** and **outgoing** edges, sorted by descending absolute attribution
  score, colored green for a positive (supporting) contribution and red for a negative
  (suppressing) one. Clicking any listed neighbor re-centers the inspector on it, which
  is the fastest way to walk a circuit edge by edge.

Loading a graph
----------------

The app scans a ``graphs/`` directory for any JSON file containing both ``nodes`` and
``edges`` arrays, and lists all of them in a selector; a JSON file can also be uploaded
directly through the UI. An optional sibling ``metadata.json`` (or ``graph_metadata.json``
/ ``run_metadata.json``) supplies the prompt text, expected answer, model/transcoder
identifiers, the node/edge pruning thresholds used, and before/after node and edge counts,
all of which are rendered alongside the graph. See ``viz/docs/visualization-json-format.md``
for the full key-by-key schema and the validation checklist to follow before publishing a
new graph JSON file.
