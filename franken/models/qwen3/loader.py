def init_student_from_teacher(student, teacher_state_dict, layer_map):
    new_state = {}

    for key, tensor in teacher_state_dict.items():
        if key.startswith("layers."):
            t = int(key.split("layers.")[1].split(".")[0])  # teacher block index
            if t in layer_map:
                i = layer_map.index(t)  # student slot for teacher block t
                new_key = key.replace(f"layers.{t}.", f"layers.{i}.", 1)
                new_state[new_key] = tensor
        else:
            new_state[key] = tensor  # embed_tokens / norm: verbatim

    student.load_state_dict(new_state, strict=False)

    return student
