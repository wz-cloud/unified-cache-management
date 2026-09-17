"""vLLM 0.23: Mamba align state management without local prefix reuse."""

from copy import copy
from functools import wraps
from types import FunctionType


def uses_ucm(transfer):
    if transfer is None:
        return False
    if isinstance(transfer, dict):
        name = transfer.get("kv_connector", "") or ""
        extra = transfer.get("kv_connector_extra_config", {}) or {}
    else:
        name = getattr(transfer, "kv_connector", "") or ""
        extra = getattr(transfer, "kv_connector_extra_config", {}) or {}
    return "UCM" in name or (
        name == "MultiConnector"
        and any(uses_ucm(item) for item in extra.get("connectors", []))
    )


def external_align(config):
    transfer = getattr(config, "kv_transfer_config", None)
    cache = config.cache_config
    return (
        uses_ucm(transfer)
        and not cache.enable_prefix_caching
        and cache.mamba_cache_mode == "align"
    )


def patch_model_config(cls):
    if "_ucm_external_align_original" in vars(cls):
        return
    original = cls.verify_and_update_config.__func__
    cls._ucm_external_align_original = staticmethod(original)

    @classmethod
    @wraps(original)
    def verify(model_cls, config):
        if not external_align(config):
            return original(model_cls, config)
        cache = config.cache_config
        # Run upstream align initialization, including block size and chunked
        # prefill validation. Restore local lookup policy even on failure.
        cache.enable_prefix_caching = True
        try:
            return original(model_cls, config)
        finally:
            cache.enable_prefix_caching = False

    cls.verify_and_update_config = verify


def patch_block_size_validator(cls):
    current = cls.validate_mamba_block_size
    if getattr(current, "_ucm_external_align", False):
        return
    # Pydantic's compiled schema retains this function object. Replace its
    # code in place so both direct calls and schema validation see the patch.
    original = FunctionType(
        current.__code__,
        current.__globals__,
        current.__name__,
        current.__defaults__,
        current.__closure__,
    )
    original.__kwdefaults__ = current.__kwdefaults__

    def validate(self, _original=original, _external_align=external_align):
        if _external_align(self):
            return self
        return _original(self)

    current.__code__ = validate.__code__
    current.__defaults__ = validate.__defaults__
    current.__kwdefaults__ = validate.__kwdefaults__
    current._ucm_external_align = True


def patch_runner(cls):
    original = cls.may_reinitialize_input_batch
    if getattr(original, "_ucm_external_align", False):
        return

    @wraps(original)
    def initialize(self, *args, **kwargs):
        if not external_align(self.vllm_config):
            return original(self, *args, **kwargs)
        cache = self.cache_config
        # Only the initialization routine sees this private copy. The engine
        # and scheduler retain enable_prefix_caching=False.
        self.cache_config = copy(cache)
        self.cache_config.enable_prefix_caching = True
        try:
            return original(self, *args, **kwargs)
        finally:
            self.cache_config = cache

    initialize._ucm_external_align = True
    cls.may_reinitialize_input_batch = initialize


def patch_preprocess(mod):
    original = mod.preprocess_mamba
    if getattr(original, "_ucm_external_align", False):
        return

    @wraps(original)
    def preprocess(*args, **kwargs):
        cache = kwargs.get("cache_config")
        if cache is None and len(args) > 2:
            cache = args[2]
        if cache is not None and (
            not cache.enable_prefix_caching and cache.mamba_cache_mode == "align"
        ):
            # Upstream only reads the PC flag for its assertion. Do not mutate
            # the shared scheduler/worker configuration during a forward pass.
            cache = copy(cache)
            cache.enable_prefix_caching = True
            if len(args) > 2:
                args = (*args[:2], cache, *args[3:])
            else:
                kwargs["cache_config"] = cache
        return original(*args, **kwargs)

    preprocess._ucm_external_align = True
    mod.preprocess_mamba = preprocess
