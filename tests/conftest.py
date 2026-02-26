import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model", action="append", default=[],
        help="Model name(s) to test (e.g. --model Qwen3-0.6B --model Qwen3-4B)",
    )
    parser.addoption(
        "--device", default=None,
        help="Device to run tests on (e.g. cpu, cuda, cuda:0). Defaults to module's DEFAULT_DEVICE.",
    )


def pytest_generate_tests(metafunc):
    if "model_name" in metafunc.fixturenames:
        models = metafunc.config.getoption("model")
        if not models:
            models = getattr(metafunc.module, "DEFAULT_MODELS", None)
        if not models:
            pytest.skip("No --model specified and no DEFAULT_MODELS in module")
        metafunc.parametrize("model_name", models)


@pytest.fixture
def device(request):
    cli = request.config.getoption("device")
    if cli is not None:
        return cli
    return getattr(request.module, "DEFAULT_DEVICE", "cuda")
