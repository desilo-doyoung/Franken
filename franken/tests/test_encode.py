import torch

from franken.encode import embed_batches


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def forward(self, model, inputs):
        self.calls += 1
        return {"output": model(inputs["input_ids"])}


class FakeTask:
    def model_inputs(self, batch):
        return {"input_ids": batch["input_ids"]}


def batches(*rows):
    return [{"input_ids": torch.tensor(r, dtype=torch.float32)} for r in rows]


def test_rows_come_back_in_batch_order():
    (out,) = embed_batches(
        FakeBackend(),
        FakeTask(),
        batches([[1.0], [2.0]], [[3.0]]),
        torch.device("cpu"),
        lambda x: x,
    )
    assert out.flatten().tolist() == [1.0, 2.0, 3.0]


def test_several_models_share_one_pass():
    # The point of the varargs: a second pass would have to reproduce the batching to keep the
    # student's rows aligned with the teacher's.
    backend = FakeBackend()
    a, b = embed_batches(
        backend,
        FakeTask(),
        batches([[1.0]], [[2.0]]),
        torch.device("cpu"),
        lambda x: x,
        lambda x: x * 10,
    )
    assert a.flatten().tolist() == [1.0, 2.0]
    assert b.flatten().tolist() == [10.0, 20.0]
    assert backend.calls == 4  # 2 batches x 2 models, i.e. one pass


def test_output_is_fp32_on_cpu():
    (out,) = embed_batches(
        FakeBackend(),
        FakeTask(),
        batches([[1.0]]),
        torch.device("cpu"),
        lambda x: x.to(torch.float16),
    )
    assert out.dtype is torch.float32 and out.device.type == "cpu"


def test_ctx_is_entered_per_batch():
    entered = []

    class Ctx:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *a):
            return False

    embed_batches(
        FakeBackend(),
        FakeTask(),
        batches([[1.0]], [[2.0]], [[3.0]]),
        torch.device("cpu"),
        lambda x: x,
        ctx=Ctx,
    )
    assert len(entered) == 3
