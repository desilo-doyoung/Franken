"""Distillation package.

  layer_map.py  teacher->student uniform-stride map (overridable)
  loss.py       (1-alpha)*CE + alpha*T^2*KL + beta*masked_MSE(hidden)
  trainer.py    frozen teacher + student loop over MRPC
"""
