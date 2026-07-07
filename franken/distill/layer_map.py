
def uniform_stride(num_teacher, num_student) -> list[int]:
    """
    Returns a list of indices that represent a uniform stride from the teacher to the student.

    Args:
        num_teacher (int): The number of teacher samples.
        num_student (int): The number of student samples.

    Returns:
        list[int]: A list of indices representing the uniform stride.
    """
    if num_student <= 0:
        raise ValueError("num_student must be a positive integer.")

    if num_teacher < num_student:
        raise ValueError("num_teacher must be greater than or equal to num_student.")

    stride = num_teacher / num_student
    indices = [round((i+1) * stride) for i in range(num_student)]

    return indices


def resolve_layer_map(num_teacher, num_student, override=None) -> list[int]:
    """
    Resolves the layer mapping from teacher to student layers.

    Args:
        num_teacher (int): The number of teacher layers.
        num_student (int): The number of student layers.
        override (list[int], optional): An optional list of indices to override the default mapping.

    Returns:
        list[int]: A list of indices representing the layer mapping.
    """
    if override is not None:
        if len(override) != num_student:
            raise ValueError("Length of override must match num_student.")
        return override

    return uniform_stride(num_teacher, num_student)
