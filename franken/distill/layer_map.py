# Where the interior window sits, as a fraction of teacher depth. qwen3's trained optimum centred
# near 0.46 and its peak was broad, so this is a midpoint, not a tuned value.
WINDOW_CENTER = 0.45

MODES = ("stride", "interior_window")


def _stride(num_teacher: int, num_student: int) -> list[int]:
    stride = num_teacher / num_student
    return [round((i + 1) * stride) - 1 for i in range(num_student)]  # -1: 1-based -> 0-based


def _interior_window(num_teacher: int, num_student: int) -> list[int]:
    """Drop one contiguous window from the interior, sparing blocks 0, 1 and the last.

    Sparing the bottom is the only layer-choice finding that replicated across models: on llama a
    leave-one-out puts block 0 39x above the cheapest block, and qwen3 measured STS-B -19.3% when
    the bottom went. `stride` drops block 0 outright at several depths (12->8, 16->10 and below).

    Scanned untrained against `stride`, an ends-protected spread, and greedy leave-one-out at every
    depth 15..8: this is the only rule that never collapses, and at 16->8 the only one whose init
    still beats a uniform predictor. ⚠️ Untrained ranking -- qwen3 found contiguity washed out once
    trained, so treat the margin over an ends-protected spread as unconfirmed.
    """
    n = num_teacher - num_student
    # Clamped so the window still fits when the student is too shallow to spare three blocks.
    start = min(max(round(WINDOW_CENTER * num_teacher - n / 2), 2), max(num_teacher - 1 - n, 0))
    return [i for i in range(num_teacher) if not start <= i < start + n]


def resolve_layer_map(num_teacher, num_student, override=None, mode: str = "stride") -> list[int]:
    """Which teacher block seeds each student block: ``layer_map[i] == t``.

    ``override`` wins; otherwise ``mode`` picks the rule. Default stays ``stride`` so every existing
    config resolves exactly as before -- ``hidden_layer_map: null`` has always meant this mode.
    """
    if override is not None:
        if len(override) != num_student:
            raise ValueError("Length of override must match num_student.")
        return override

    if num_student <= 0:
        raise ValueError("num_student must be a positive integer.")
    if num_teacher < num_student:
        raise ValueError("num_teacher must be greater than or equal to num_student.")
    if mode not in MODES:
        raise ValueError(f"Unknown layer-map mode {mode!r}; use {' | '.join(MODES)}")

    return (
        _stride(num_teacher, num_student)
        if mode == "stride"
        else _interior_window(num_teacher, num_student)
    )
