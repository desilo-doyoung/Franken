def resolve_layer_map(num_teacher, num_student, override=None) -> list[int]:
    """Which teacher block seeds each student block: ``layer_map[i] == t``. Default is a uniform
    stride over the teacher's depth."""
    if override is not None:
        if len(override) != num_student:
            raise ValueError("Length of override must match num_student.")
        return override

    if num_student <= 0:
        raise ValueError("num_student must be a positive integer.")
    if num_teacher < num_student:
        raise ValueError("num_teacher must be greater than or equal to num_student.")

    stride = num_teacher / num_student
    return [round((i + 1) * stride) - 1 for i in range(num_student)]  # -1: 1-based -> 0-based
