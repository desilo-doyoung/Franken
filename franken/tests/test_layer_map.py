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


def test_stride_is_still_the_default():
    # Every shipped config says `hidden_layer_map: null`, so the default resolving differently
    # would silently restate what every recorded result was trained on.
    for teacher, student in ((12, 6), (28, 19), (16, 12), (16, 8)):
        assert resolve_layer_map(teacher, student) == resolve_layer_map(
            teacher, student, mode="stride"
        )


@pytest.mark.parametrize(("teacher", "student"), [(16, 15), (16, 12), (16, 8), (28, 19), (12, 6)])
def test_interior_window_spares_both_ends(teacher, student):
    m = resolve_layer_map(teacher, student, mode="interior_window")
    assert len(m) == student
    assert m[0] == 0 and m[-1] == teacher - 1
    assert 1 in m  # block 1 is second-costliest by leave-one-out; stride drops it at 16->12


@pytest.mark.parametrize(("teacher", "student"), [(16, 15), (16, 12), (16, 8), (16, 2), (16, 1)])
def test_interior_window_drops_one_contiguous_run(teacher, student):
    m = resolve_layer_map(teacher, student, mode="interior_window")
    dropped = sorted(set(range(teacher)) - set(m))
    assert len(dropped) == teacher - student
    if dropped:
        assert dropped == list(range(dropped[0], dropped[-1] + 1))


@pytest.mark.parametrize("student", list(range(1, 17)))
def test_interior_window_stays_valid_when_ends_cannot_be_spared(student):
    # At student depth 1-2 there is no room for three protected blocks; the map must still be a
    # strictly increasing list of the right length ending at the final teacher block.
    m = resolve_layer_map(16, student, mode="interior_window")
    assert len(m) == student and m[-1] == 15
    assert all(b > a for a, b in zip(m, m[1:], strict=False))


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        resolve_layer_map(16, 12, mode="middle_out")


def test_override_beats_mode():
    assert resolve_layer_map(16, 2, override=[3, 9], mode="interior_window") == [3, 9]
