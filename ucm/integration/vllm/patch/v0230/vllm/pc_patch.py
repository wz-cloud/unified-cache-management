from ucm.integration.vllm.patch.utils import when_imported
from ucm.integration.vllm.patch.v0230.vllm import external_mamba


@when_imported("vllm.model_executor.models.config")
def patch_model_config(mod):
    external_mamba.patch_model_config(mod.MambaModelConfig)


@when_imported("vllm.config.vllm")
def patch_config(mod):
    external_mamba.patch_block_size_validator(mod.VllmConfig)


@when_imported("vllm.v1.worker.gpu_model_runner")
def patch_runner(mod):
    external_mamba.patch_runner(mod.GPUModelRunner)


@when_imported("vllm.v1.worker.mamba_utils")
def patch_preprocess(mod):
    external_mamba.patch_preprocess(mod)


@when_imported("vllm_ascend.patch.platform.patch_mamba_config")
def patch_ascend_config(mod):
    # Ascend replaces the hybrid class's method and recomputes block sizes.
    from vllm.model_executor.models.config import HybridAttentionMambaModelConfig

    external_mamba.patch_model_config(HybridAttentionMambaModelConfig)


@when_imported("vllm_ascend.worker.model_runner_v1")
def patch_ascend_runner(mod):
    external_mamba.patch_runner(mod.NPUModelRunner)
