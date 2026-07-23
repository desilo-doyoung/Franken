"""Distillation package (model- and task-agnostic).

layer_map.py  teacher->student uniform-stride map (overridable)
loss.py       masked_mse_loss — the generic hidden-state MSE helper tasks share
trainer.py    frozen teacher + student loop, driven by a ModelBackend + Task
"""
