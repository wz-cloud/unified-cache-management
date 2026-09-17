from wrapt import when_imported

from ucm.integration.vllm.patch.common import external_mamba


@when_imported("vllm.model_executor.models.config")
def patch_model_config(mod):
    external_mamba.patch_model_config(getattr(mod, "MambaModelConfig", None))


@when_imported("vllm.config.vllm")
def patch_config(mod):
    external_mamba.patch_block_size_validator(getattr(mod, "VllmConfig", None))


@when_imported("vllm.v1.worker.gpu_model_runner")
def patch_runner(mod):
    external_mamba.patch_runner(getattr(mod, "GPUModelRunner", None))


@when_imported("vllm.v1.worker.mamba_utils")
def patch_preprocess(mod):
    external_mamba.patch_preprocess(mod)


@when_imported("vllm_ascend.patch.platform.patch_mamba_config")
def patch_ascend_config(mod):
    # Ascend replaces the hybrid class's method and recomputes block sizes.
    from vllm.model_executor.models import config

    external_mamba.patch_model_config(
        getattr(config, "HybridAttentionMambaModelConfig", None)
    )


@when_imported("vllm_ascend.worker.model_runner_v1")
def patch_ascend_runner(mod):
    external_mamba.patch_runner(getattr(mod, "NPUModelRunner", None))
