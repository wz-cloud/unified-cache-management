"""CPU-only checks for external Mamba align configuration and isolation."""

import importlib.util
import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import model_validator
from pydantic.dataclasses import dataclass


path = Path(__file__).resolve().parents[1] / (
    "ucm/integration/vllm/patch/v0230/vllm/external_mamba.py"
)
spec = importlib.util.spec_from_file_location("external_mamba_patch", path)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def config(connector="UCMConnector", mode="align", pc=False):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=pc,
            mamba_cache_mode=mode,
            mamba_block_size=None,
            block_size=128,
        ),
        kv_transfer_config=SimpleNamespace(kv_connector=connector),
    )


def test_model_config_and_exception_restore():
    class Model:
        @classmethod
        def verify_and_update_config(cls, cfg):
            cache = cfg.cache_config
            if not cache.enable_prefix_caching:
                cache.mamba_cache_mode = "none"
            cache.mamba_block_size = (
                cache.block_size if cache.enable_prefix_caching else 4096
            )
            if getattr(cfg, "fail", False):
                raise ValueError("invalid chunked prefill")

    patch.patch_model_config(Model)
    patch.patch_model_config(Model)
    cfg = config()
    Model.verify_and_update_config(cfg)
    assert cfg.cache_config.mamba_cache_mode == "align"
    assert cfg.cache_config.mamba_block_size == 128
    assert not cfg.cache_config.enable_prefix_caching
    cfg.fail = True
    with pytest.raises(ValueError):
        Model.verify_and_update_config(cfg)
    assert not cfg.cache_config.enable_prefix_caching
    other = config(connector="OtherConnector")
    Model.verify_and_update_config(other)
    assert other.cache_config.mamba_cache_mode == "none"


def test_compiled_pydantic_validator_preserves_other_cases():
    @dataclass
    class Config:
        cache_config: object
        kv_transfer_config: object

        @model_validator(mode="after")
        def validate_mamba_block_size(self):
            if not self.cache_config.enable_prefix_caching:
                raise ValueError("PC required")
            return self

    patch.patch_block_size_validator(Config)
    patch.patch_block_size_validator(Config)
    cfg = config()
    result = Config(**vars(cfg))
    assert not result.cache_config.enable_prefix_caching
    for cfg in (config(mode="none"), config(connector="OtherConnector")):
        with pytest.raises(ValueError, match="PC required"):
            Config(**vars(cfg))
    Config(**vars(config(pc=True)))


def test_runner_allocates_multiple_blocks_and_restores_on_error():
    class Runner:
        def may_reinitialize_input_batch(self, fail=False):
            assert not self.vllm_config.cache_config.enable_prefix_caching
            blocks = 32 if self.cache_config.enable_prefix_caching else 1
            if fail:
                raise RuntimeError("allocation failed")
            return blocks

    patch.patch_runner(Runner)
    patch.patch_runner(Runner)
    runner = Runner()
    runner.vllm_config = config()
    runner.cache_config = runner.vllm_config.cache_config
    original = runner.cache_config
    assert runner.may_reinitialize_input_batch() == 32
    with pytest.raises(RuntimeError):
        runner.may_reinitialize_input_batch(fail=True)
    assert runner.cache_config is original
    assert not original.enable_prefix_caching


@pytest.mark.parametrize("keyword", [False, True])
def test_preprocess_uses_private_config(keyword):
    cache = config().cache_config

    def preprocess(a, b, cache_config):
        assert cache_config.enable_prefix_caching
        assert not cache.enable_prefix_caching
        assert cache_config.mamba_cache_mode == "align"
        raise RuntimeError("copy failed")

    mod = SimpleNamespace(preprocess_mamba=preprocess)
    patch.patch_preprocess(mod)
    patch.patch_preprocess(mod)
    with pytest.raises(RuntimeError, match="copy failed"):
        if keyword:
            mod.preprocess_mamba(None, None, cache_config=cache)
        else:
            mod.preprocess_mamba(None, None, cache)
    assert not cache.enable_prefix_caching


def test_multi_connector_scope():
    cfg = config(connector="MultiConnector")
    cfg.kv_transfer_config.kv_connector_extra_config = {
        "connectors": [{"kv_connector": "UCMConnectorV1"}]
    }
    assert patch.external_align(cfg)
    cfg.kv_transfer_config.kv_connector_extra_config = {
        "connectors": [{"kv_connector": "OtherConnector"}]
    }
    assert not patch.external_align(cfg)


def test_v0230_original_model_config():
    """Exercise the actual tagged method when a sibling vLLM checkout exists."""
    checkout = path.parents[7] / "vllm"
    result = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "show",
            "v0.23.0:vllm/model_executor/models/config.py",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        pytest.skip("Requires sibling vLLM checkout with v0.23.0 tag")
    tree = ast.parse(result.stdout)
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MambaModelConfig"
    )
    model.bases = []
    logger = SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
    namespace = {"logger": logger}
    exec(
        compile(ast.Module(body=[model], type_ignores=[]), "v0.23.0", "exec"), namespace
    )
    cls = namespace["MambaModelConfig"]
    patch.patch_model_config(cls)
    cfg = config()
    cfg.model_config = SimpleNamespace(max_model_len=4096)
    cfg.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    cls.verify_and_update_config(cfg)
    assert cfg.cache_config.mamba_cache_mode == "align"
    assert cfg.cache_config.mamba_block_size == 128
    assert not cfg.cache_config.enable_prefix_caching
    cfg.scheduler_config.enable_chunked_prefill = False
    with pytest.raises(AssertionError, match="Chunked prefill"):
        cls.verify_and_update_config(cfg)
    assert not cfg.cache_config.enable_prefix_caching
