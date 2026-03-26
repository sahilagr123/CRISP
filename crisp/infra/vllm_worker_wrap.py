"""vLLM WorkerWrap — extension injected into each vLLM worker process.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI) with
PPO-specific code removed. Receives weight broadcasts from DeepSpeed
training processes and applies them to the vLLM model.
"""


class WorkerWrap:
    """Extension injected into each vLLM worker process."""

    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        use_ray=False,
    ):
        """Init torch process group for model weights update."""
        import torch
        from crisp.infra.utils import init_process_group

        assert torch.distributed.is_initialized()
        assert group_name != ""
        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(
                world_size=world_size,
                rank=rank,
                backend=backend,
                group_name=group_name,
            )
            self._model_update_group = group_name
        else:
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )

    def load_weight_direct(self, name, weight):
        """Load a single weight tensor directly (no broadcast)."""
        import torch

        if isinstance(weight, list):
            weight = torch.tensor(weight)
        if not weight.is_cuda:
            weight = weight.to("cuda")
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def load_weights_from_file(self, path):
        """Load all weights from a saved state dict file (HF-format names).

        Handles the name mapping between HuggingFace (separate q/k/v, gate/up)
        and vLLM (fused qkv_proj, gate_up_proj).
        """
        import torch

        # Load to CPU to avoid doubling GPU memory during copy
        hf_state = torch.load(path, weights_only=True, map_location="cpu")
        loaded = 0
        for name, param in self.model_runner.model.named_parameters():
            if name in hf_state:
                param.data.copy_(hf_state[name].to(param.device))
                loaded += 1
            elif "qkv_proj" in name:
                # vLLM fuses q_proj + k_proj + v_proj -> qkv_proj
                suffix = ".bias" if name.endswith(".bias") else ".weight"
                base = name.replace("qkv_proj" + suffix, "")
                q = hf_state.get(base + "q_proj" + suffix)
                k = hf_state.get(base + "k_proj" + suffix)
                v = hf_state.get(base + "v_proj" + suffix)
                if q is not None:
                    fused = torch.cat([q, k, v], dim=0).to(param.device)
                    param.data.copy_(fused)
                    del fused
                    loaded += 1
            elif "gate_up_proj" in name:
                # vLLM fuses gate_proj + up_proj -> gate_up_proj
                suffix = ".bias" if name.endswith(".bias") else ".weight"
                base = name.replace("gate_up_proj" + suffix, "")
                gate = hf_state.get(base + "gate_proj" + suffix)
                up = hf_state.get(base + "up_proj" + suffix)
                if gate is not None:
                    fused = torch.cat([gate, up], dim=0).to(param.device)
                    param.data.copy_(fused)
                    del fused
                    loaded += 1
        del hf_state
        torch.cuda.empty_cache()

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Receive broadcast weight from DeepSpeed rank 0."""
        import torch

        assert dtype == self.model_config.dtype
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(
                weight, src=0, stream=torch.cuda.current_stream()
            )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_weight_cuda_ipc(
        self, name, dtype, shape, ipc_handles=None, empty_cache=False
    ):
        """Receive weight via CUDA IPC handles (same-node only)."""
        import torch
        from crisp.infra.utils import get_physical_gpu_id

        assert dtype == self.model_config.dtype
        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        list_args[6] = device_id
        weight = func(*list_args)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
