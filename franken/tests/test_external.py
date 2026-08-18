from franken.data.embed_corpus import WEB_SEARCH, Pool
from franken.data.external import EXTERNAL, Benchmark


def test_the_five_scored_benchmarks_are_declared():
    assert set(EXTERNAL) == {"nfcorpus", "scifact", "fiqa", "xpqa_cmn", "code_apps"}


def test_every_benchmark_instructs_its_queries():
    # The instruction asymmetry is the model's contract; an uninstructed query measures
    # symmetric similarity instead of retrieval.
    for name, b in EXTERNAL.items():
        assert callable(b.load), name
        assert b.instruct, name


def test_task_specific_strings_are_the_exception():
    # Only where a sweep measured a task string beating web by more than the ~0.005 floor.
    tailored = {n for n, b in EXTERNAL.items() if b.instruct != WEB_SEARCH}
    assert tailored == {"nfcorpus", "code_apps"}


def test_pool_passes_the_instruction_to_the_loader():
    seen = []
    b = Benchmark(lambda task: seen.append(task) or Pool(), "an instruction")
    b.pool()
    assert seen == ["an instruction"]
