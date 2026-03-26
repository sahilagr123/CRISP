"""vLLM engine wrapper — Ray actors for fast inference with vLLM.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI) with
project-specific paths updated. Provides:

- ``BaseLLMRayActor`` — base class handling GPU visibility, env setup,
  full-determinism, and vLLM version checks.
- ``LLMRayActor`` — ``@ray.remote`` actor wrapping ``vllm.LLM``.
- ``create_vllm_engines`` — factory that creates engines in placement groups.
- ``batch_vllm_engine_call`` — fans out a method call to all engines.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class BaseLLMRayActor:
    """Base class for vLLM Ray actors.

    Handles GPU visibility setup, environment configuration,
    full-determinism flags, and vLLM version compatibility checks.
    """

    def _configure_env(self, full_determinism: bool = False) -> None:
        """Set environment variables for deterministic execution."""
        if full_determinism:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def _check_vllm_version(self) -> str:
        """Return the installed vLLM version string."""
        import vllm

        return vllm.__version__

    def _setup_gpu_visibility(self, cuda_visible_devices: Optional[str] = None) -> None:
        """Restrict CUDA visible devices for this worker."""
        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


class LLMRayActor(BaseLLMRayActor):
    """Ray actor wrapping ``vllm.LLM`` for fast inference.

    This class is decorated with ``@ray.remote`` at import time when Ray is
    available.  When Ray is not installed the class can still be imported
    (for testing with mocks).
    """

    def __init__(
        self,
        pretrain: str,
        seed: int = 0,
        full_determinism: bool = False,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
        max_model_len: int = 4096,
        gpu_memory_utilization: Optional[float] = None,
        tensor_parallel_size: int = 1,
        vllm_enable_sleep: bool = False,
        **kwargs: Any,
    ) -> None:
        import torch
        import vllm

        self._configure_env(full_determinism)

        engine_kwargs: dict[str, Any] = dict(
            model=pretrain,
            seed=seed,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            worker_cls="auto",
            worker_extension_cls="crisp.infra.vllm_worker_wrap.WorkerWrap",
            **kwargs,
        )
        if gpu_memory_utilization is not None:
            engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        if vllm_enable_sleep:
            engine_kwargs["enable_sleep_mode"] = True

        self.llm = vllm.LLM(**engine_kwargs)

        # Discover collective_rpc path for weight sync
        if hasattr(self.llm, "collective_rpc"):
            self._rpc_path = "LLM"
        elif hasattr(self.llm.llm_engine, "collective_rpc"):
            self._rpc_path = "LLMEngine"
        elif hasattr(self.llm.llm_engine, "model_executor"):
            self._rpc_path = "model_executor"
        else:
            self._rpc_path = "unknown"
        logger.info("vLLM collective_rpc path: %s (vllm %s)", self._rpc_path, vllm.__version__)

    # ------------------------------------------------------------------
    # Methods exposed as Ray remote calls
    # ------------------------------------------------------------------

    def _collective_rpc(self, method: str, args: tuple = (), kwargs: dict | None = None) -> Any:
        """Call collective_rpc on vLLM workers, supporting both V0 and V1 engines.

        V0: self.llm.llm_engine.model_executor.collective_rpc(...)
        V1: self.llm.collective_rpc(...) or self.llm.llm_engine.collective_rpc(...)
        """
        if self._rpc_path == "LLM":
            return self.llm.collective_rpc(method, args=args, kwargs=kwargs)
        if self._rpc_path == "LLMEngine":
            return self.llm.llm_engine.collective_rpc(method, args=args, kwargs=kwargs)
        if self._rpc_path == "model_executor":
            return self.llm.llm_engine.model_executor.collective_rpc(
                method, args=args, kwargs=kwargs,
            )
        # Last resort: try all paths with diagnostics
        for path_name, obj in [
            ("LLM", self.llm),
            ("LLMEngine", self.llm.llm_engine),
        ]:
            if hasattr(obj, "collective_rpc"):
                logger.info("Found collective_rpc on %s (late discovery)", path_name)
                return obj.collective_rpc(method, args=args, kwargs=kwargs)
        # Dump available attributes for debugging
        engine_attrs = [a for a in dir(self.llm.llm_engine) if not a.startswith("__")]
        raise AttributeError(
            f"Cannot find collective_rpc. LLMEngine attrs: {engine_attrs}"
        )

    def init_process_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        use_ray: bool = False,
    ) -> None:
        """Initialise torch distributed process group on each worker."""
        self._collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name),
            kwargs={"backend": backend, "use_ray": use_ray},
        )

    def update_weight(
        self,
        name: str,
        dtype: Any,
        shape: tuple,
        empty_cache: bool = False,
    ) -> None:
        """Broadcast-receive a single weight tensor from DeepSpeed rank-0."""
        self._collective_rpc(
            "update_weight",
            args=(name, dtype, shape),
            kwargs={"empty_cache": empty_cache},
        )

    def update_weight_cuda_ipc(
        self,
        name: str,
        dtype: Any,
        shape: tuple,
        ipc_handles: Any = None,
        empty_cache: bool = False,
    ) -> None:
        """Receive weight via CUDA IPC handles (same-node only)."""
        self._collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape),
            kwargs={"ipc_handles": ipc_handles, "empty_cache": empty_cache},
        )

    def reset_prefix_cache(self) -> None:
        """Reset the KV prefix cache."""
        self._collective_rpc("reset_prefix_cache")

    def sleep(self) -> None:
        """Put the engine to sleep (free GPU memory)."""
        self.llm.sleep()

    def wake_up(self) -> None:
        """Wake the engine from sleep."""
        self.llm.wake_up()

    def update_weight_direct(self, name: str, weight) -> None:
        """Update a single weight tensor directly (no process group needed).

        Used for single-GPU weight sync from DeepSpeed to vLLM.
        """
        self._collective_rpc("load_weight_direct", args=(name, weight))

    def load_weights_from_file(self, path: str) -> None:
        """Load all weights from a saved state dict file.

        Used for single-GPU weight sync: caller saves state dict to a temp
        file, then calls this to load it into the vLLM model via collective_rpc.
        Avoids tensor serialization through vLLM V1's msgspec IPC.
        """
        self._collective_rpc("load_weights_from_file", args=(path,))

    def generate(self, sampling_params: Any, prompt_token_ids: list) -> list:
        """Generate completions for a batch of prompts.

        Args:
            sampling_params: vLLM SamplingParams object.
            prompt_token_ids: List of token-id lists, one per prompt.

        Returns:
            List of RequestOutput objects from vLLM.
        """
        from vllm import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=tids) for tids in prompt_token_ids]
        return self.llm.generate(prompts, sampling_params)


# ---------------------------------------------------------------------------
# Apply @ray.remote decorator only when Ray is available
# ---------------------------------------------------------------------------
try:
    import ray

    LLMRayActor = ray.remote(LLMRayActor)  # type: ignore[misc]
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg: Any = None,
    gpu_memory_utilization: Optional[float] = None,
    vllm_enable_sleep: bool = False,
    llm_actor_cls: Any = LLMRayActor,
    **kwargs: Any,
) -> List[Any]:
    """Create *num_engines* vLLM Ray actors with GPU placement groups.

    Parameters
    ----------
    num_engines:
        Number of independent vLLM engine actors to create.
    tensor_parallel_size:
        Number of GPUs per engine (for tensor parallelism).
    pretrain:
        HuggingFace model name or path.
    seed:
        Random seed for reproducibility.
    full_determinism:
        Whether to enable full deterministic mode.
    enable_prefix_caching:
        Enable vLLM KV-cache prefix sharing.
    enforce_eager:
        Disable CUDA graphs (useful for debugging).
    max_model_len:
        Maximum sequence length for the model.
    shared_pg:
        Optional pre-existing Ray placement group.
    gpu_memory_utilization:
        Fraction of GPU memory to use (0-1).
    vllm_enable_sleep:
        Enable vLLM sleep mode for memory saving.
    llm_actor_cls:
        The Ray actor class to instantiate (default ``LLMRayActor``).

    Returns
    -------
    list
        A list of Ray actor handles.
    """
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    engines: list[Any] = []
    for i in range(num_engines):
        # Create a placement group per engine unless one is shared
        if shared_pg is None:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles, strategy="PACK")
            ray.get(pg.ready())
        else:
            pg = shared_pg

        # When sharing a placement group, use fractional GPU claim
        # so multiple engines can coexist on the same GPU.
        gpu_claim = 0.4 if shared_pg is not None else tensor_parallel_size

        engine = llm_actor_cls.options(
            num_cpus=1,
            num_gpus=gpu_claim,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
            ),
        ).remote(
            pretrain=pretrain,
            seed=seed,
            full_determinism=full_determinism,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            vllm_enable_sleep=vllm_enable_sleep,
            **kwargs,
        )
        engines.append(engine)

    return engines


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def batch_vllm_engine_call(
    engines: List[Any],
    method_name: str,
    *args: Any,
    rank_0_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Fan out a method call to all engines and collect results.

    Parameters
    ----------
    engines:
        List of Ray actor handles.
    method_name:
        Name of the method to call on each engine.
    rank_0_only:
        If ``True`` and torch.distributed is initialised, only rank-0
        actually issues the calls.  Other ranks return ``None``.

    Returns
    -------
    list or None
        Results from ``ray.get`` or ``None`` if skipped.
    """
    import torch

    if rank_0_only and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None

    import ray

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        ref = method.remote(*args, **kwargs)
        refs.append(ref)

    return ray.get(refs)
