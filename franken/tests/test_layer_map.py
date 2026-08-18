import pytest

from franken.distill.layer_map import resolve_layer_map


def test_uniform_stride_takes_the_last_block_of_each_group():
    depth19 = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27]
    assert resolve_layer_map(12, 6) == [1, 3, 5, 7, 9, 11]
    assert resolve_layer_map(28, 19) == depth19


def test_equal_depths_is_the_identity():
    assert resolve_layer_map(28, 28) == list(range(28))


def test_final_student_block_always_maps_to_the_final_teacher_block():
    # The embedding is pooled from the last block, so a map that stops short changes what is
    # being distilled.
    for student in range(1, 29):
        assert resolve_layer_map(28, student)[-1] == 27


def test_map_is_strictly_increasing():
    m = resolve_layer_map(28, 19)
    assert all(b > a for a, b in zip(m, m[1:], strict=False))


def test_override_passes_through():
    assert resolve_layer_map(28, 3, override=[0, 13, 27]) == [0, 13, 27]


def test_override_length_must_match_student_depth():
    with pytest.raises(ValueError):
        resolve_layer_map(28, 3, override=[0, 27])


@pytest.mark.parametrize(("teacher", "student"), [(28, 0), (28, -1), (6, 12)])
def test_impossible_depths_raise(teacher, student):
    with pytest.raises(ValueError):
        resolve_layer_map(teacher, student)
