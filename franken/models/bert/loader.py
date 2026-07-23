def init_student_from_teacher(student, teacher_state_dict, layer_map):
    new_state = {}

    for key, tensor in teacher_state_dict.items():
        if ".encoder.layer." in key:
            # 1. parse the teacher block index right after ".encoder.layer."
            #    e.g. "bert.encoder.layer.3.attention.self.query.weight" -> 3
            t = int(key.split(".encoder.layer.")[1].split(".")[0])
            # 2. is this teacher block one the student wants? find its student slot.
            #    layer_map[i] == t  means teacher block t seeds student block i.
            if t in layer_map:
                i = layer_map.index(t)
                # 3. rewrite ".encoder.layer.{t}." -> ".encoder.layer.{i}."
                new_key = key.replace(f".encoder.layer.{t}.", f".encoder.layer.{i}.")
                new_state[new_key] = tensor
        else:
            # non-layer key (embeddings, pooler, classifier): copy verbatim
            new_state[key] = tensor

    student.load_state_dict(new_state, strict=False)

    return student
