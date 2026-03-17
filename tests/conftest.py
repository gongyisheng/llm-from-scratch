import atexit
import os

import pytest
import torch


# init distributed when running under torchrun
if "RANK" in os.environ:
    from parallel.comm import init_process_group, destroy_process_group
    init_process_group()
    atexit.register(destroy_process_group)


def pytest_addoption(parser):
    parser.addoption(
        "--model", action="append", default=[],
        help="Model name(s) to test (e.g. --model Qwen3-0.6B --model Qwen3-4B)",
    )
    parser.addoption(
        "--device", default=None,
        help="Device to run tests on (e.g. cpu, cuda, cuda:0)",
    )


def pytest_generate_tests(metafunc):
    if "model_name" in metafunc.fixturenames:
        models = metafunc.config.getoption("model")
        if not models:
            pytest.skip("No --model specified and no DEFAULT_MODELS in module")
        metafunc.parametrize("model_name", models)


@pytest.fixture
def device(request):
    cli = request.config.getoption("device")
    if cli is not None:
        return cli
    return "cuda" if torch.cuda.is_available() else "cpu"
