"""Distillation package.

Intentionally left empty — the layer mapping, combined loss, and training loop
are implemented interactively in a tutorial session so the process is understood
step by step. Planned modules:

  layer_map.py  teacher->student uniform-stride map (overridable)
  loss.py       (1-alpha)*CE + alpha*T^2*KL + beta*masked_MSE(hidden)
  trainer.py    frozen teacher + student loop over MRPC
"""
