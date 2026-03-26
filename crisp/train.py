"""End-to-end training entry point for CRISP."""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Optional

from crisp.evaluation.aime import load_aime24_problems, load_aime25_problems
from crisp.evaluation.bayes_at_n import bayesian_pass_at_n
from crisp.evaluation.benchmarks import evaluate_on_problems
from crisp.evaluation.dapo import load_dapo_problems

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="CRISP multi-agent RL training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides as key=value (e.g. infra.learning_rate=1e-4)",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to DeepSpeed checkpoint directory to resume from")
    parser.add_argument("--resume-hf", type=str, default=None,
                        help="Path to HF weights directory to resume from "
                             "(contains alice_hf/, bob_hf/, coach_hf/, iteration.txt)")
    parser.add_argument("--save-lora", type=str, default=None,
                        help="Save LoRA adapters to this path at end of training")
    parser.add_argument("--merge-lora", type=str, default=None,
                        help="Merge LoRA into base model and save to this path")
    parser.add_argument("--save-hf", type=str, default=None,
                        help="Save HF-format weights to this path at end of training")
    parser.add_argument("--save-hf-home", type=str, default=None,
                        help="Save player-only HF weights to this persistent path "
                             "for periodic saves and SIGTERM (survives job termination)")
    return parser.parse_args(argv)


def parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    """Parse key=value override strings into a dict."""
    overrides: Dict[str, Any] = {}
    for item in override_list:
        key, _, value = item.partition("=")
        # Try numeric conversion
        try:
            parsed: Any = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                if value.lower() == "true":
                    parsed = True
                elif value.lower() == "false":
                    parsed = False
                else:
                    parsed = value
        overrides[key] = parsed
    return overrides


def init_infra(config: Any) -> Any:
    """Initialize Ray, vLLM engines, DeepSpeed strategies, ref model.

    Returns a WorkflowContext ready for main_loop.step().
    """
    import ray

    from crisp.infra.strategy import DeepSpeedStrategy
    from crisp.infra.vllm_engine import create_vllm_engines
    from crisp.rewards.ema_tracker import EMATracker
    from crisp.rewards.repetition_buffer import RepetitionBuffer
    from crisp.workflow.context import WorkflowContext

    tcfg = config.training
    icfg = config.infra
    coach_model = tcfg.coach_model_name or tcfg.model_name

    # NOTE: expandable_segments is set AFTER vLLM init (below) because
    # vLLM's CuMemAllocator is incompatible with expandable_segments.

    # 1. Init Ray
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized")

    # 2. Create vLLM engines (player + coach)
    #
    # 2-GPU mode: both engines share GPU 0 via sleep/wake time-sharing.
    # Player and coach never run simultaneously — coach generates problems,
    # sleeps, then player wakes for rollouts.
    #
    # For shared-GPU mode, we create a single placement group and use
    # fractional GPU claims so Ray schedules both actors on GPU 0.
    use_shared_gpu = (icfg.num_gpus_per_node == 2
                      and coach_model != tcfg.model_name)

    if use_shared_gpu:
        from ray.util.placement_group import placement_group as _pg
        shared_pg = _pg([{"GPU": 1, "CPU": 2}], strategy="PACK")
        ray.get(shared_pg.ready())
    else:
        shared_pg = None

    vllm_kwargs = dict(
        num_engines=icfg.vllm_num_engines,
        tensor_parallel_size=icfg.vllm_tensor_parallel_size,
        pretrain=tcfg.model_name,
        seed=icfg.seed,
        full_determinism=False,
        enable_prefix_caching=icfg.enable_prefix_caching,
        enforce_eager=True,
        max_model_len=icfg.max_model_len,
        gpu_memory_utilization=icfg.vllm_gpu_memory_utilization,
        vllm_enable_sleep=icfg.vllm_enable_sleep,
        shared_pg=shared_pg,
    )
    player_vllm = create_vllm_engines(**vllm_kwargs)
    # Wait for vLLM engine to fully initialize before proceeding
    ray.get([e._check_vllm_version.remote() for e in player_vllm])
    logger.info("Player vLLM engine ready")

    # Coach vLLM: create on same GPU 0 (shared placement group)
    if use_shared_gpu:
        # Sleep player first so coach can claim KV cache memory
        ray.get([e.sleep.remote() for e in player_vllm])
        logger.info("Player vLLM asleep, initializing coach vLLM on GPU 0...")

        coach_vllm_kwargs = dict(
            num_engines=icfg.vllm_num_engines,
            tensor_parallel_size=icfg.vllm_tensor_parallel_size,
            pretrain=coach_model,
            seed=icfg.seed,
            full_determinism=False,
            enable_prefix_caching=False,
            enforce_eager=True,
            max_model_len=icfg.coach_vllm_max_model_len,
            gpu_memory_utilization=icfg.coach_vllm_gpu_memory_utilization,
            vllm_enable_sleep=True,
            shared_pg=shared_pg,
        )
        coach_vllm = create_vllm_engines(**coach_vllm_kwargs)
        ray.get([e._check_vllm_version.remote() for e in coach_vllm])
        # Sleep coach too — main_loop will wake as needed
        ray.get([e.sleep.remote() for e in coach_vllm])
        logger.info("Coach vLLM ready (both engines sleeping on GPU 0)")
    elif icfg.num_gpus_per_node <= 2:
        if coach_model == tcfg.model_name:
            coach_vllm = player_vllm
            logger.info("vLLM engines created (player=%d, coach=shared)",
                         len(player_vllm))
        else:
            coach_vllm = None
            logger.info("vLLM engines created (player=%d, coach=HF-generate)",
                         len(player_vllm))
    else:
        coach_vllm_kwargs = dict(
            num_engines=icfg.vllm_num_engines,
            tensor_parallel_size=icfg.vllm_tensor_parallel_size,
            pretrain=coach_model,
            seed=icfg.seed,
            full_determinism=False,
            enable_prefix_caching=False,
            enforce_eager=True,
            max_model_len=icfg.coach_vllm_max_model_len,
            gpu_memory_utilization=icfg.coach_vllm_gpu_memory_utilization,
            vllm_enable_sleep=icfg.vllm_enable_sleep,
        )
        coach_vllm = create_vllm_engines(**coach_vllm_kwargs)
        ray.get([e._check_vllm_version.remote() for e in coach_vllm])
        logger.info("vLLM engines created (player=%d, coach=%d)",
                     len(player_vllm), len(coach_vllm))

    # GPU pinning for asymmetric 2-GPU mode:
    # vLLM Ray actor already occupies GPU 0; pin training to GPU 1.
    # Can't use CUDA_VISIBLE_DEVICES (CUDA already initialized by imports).
    # Instead, set LOCAL_RANK=1 so DeepSpeed uses cuda:1, and call
    # set_device(1) so all subsequent CUDA ops default to GPU 1.
    if icfg.num_gpus_per_node == 2:
        import torch as _torch
        os.environ["LOCAL_RANK"] = "1"
        _torch.cuda.set_device(1)
        logger.info("2-GPU mode: vLLM inference on GPU 0, training on GPU 1")
    elif icfg.num_gpus_per_node == 4:
        import torch as _torch
        os.environ["LOCAL_RANK"] = "2"
        _torch.cuda.set_device(2)
        logger.info("4-GPU mode: vLLM on GPU 0-1, training on GPU 2-3")

    # Now that vLLM engines are initialized (with CuMemAllocator), it's safe
    # to enable expandable_segments for DeepSpeed/PyTorch on the training GPU.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Disable SDPA math backend to prevent O(L²) memory fallback.
    # With math disabled, SDPA uses flash or memory-efficient backends
    # (both O(L) memory). Without this, padded attention masks force SDPA
    # to fall back to the math backend which materializes [heads, L, L]
    # attention matrices — e.g. 14 GiB for L=10K with 32 heads.
    import torch as _torch_sdp
    _torch_sdp.backends.cuda.enable_math_sdp(False)
    logger.info("SDPA math backend disabled (using flash/mem_efficient only)")

    # 3. Create DeepSpeed strategies
    coach_lr = icfg.coach_learning_rate or (icfg.learning_rate / 2)

    def _make_strategy(learning_rate: float = icfg.learning_rate) -> DeepSpeedStrategy:
        return DeepSpeedStrategy(
            seed=icfg.seed,
            bf16=icfg.bf16,
            zero_stage=icfg.zero_stage,
            adam_offload=icfg.adam_offload,
            max_norm=icfg.max_norm,
            learning_rate=learning_rate,
            weight_decay=icfg.weight_decay,
            micro_train_batch_size=icfg.micro_train_batch_size,
            gradient_checkpointing=icfg.gradient_checkpointing,
            attn_implementation=tcfg.attn_implementation,
            ref_reward_offload=tcfg.ref_reward_offload,
        )

    actor_kwargs = dict(
        bf16=icfg.bf16,
        attn_implementation=tcfg.attn_implementation,
        lora_rank=icfg.lora_rank,
        lora_alpha=icfg.lora_alpha,
        lora_dropout=icfg.lora_dropout,
        target_modules=icfg.target_modules or None,
    )

    if icfg.num_gpus_per_node == 4:
        # --- 4-GPU split training path ---
        # GPU 2: Alice + Bob + Ref (ref parked on CPU), GPU 3: Coach
        from crisp.infra.actor_model import Actor

        # Alice model on GPU 2 (current device from pinning above)
        # Use gloo backend: NCCL binds to one GPU, but we need DeepSpeed
        # engines on both GPU 2 and GPU 3. With world_size=1, gloo is
        # functionally identical (all collectives are no-ops).
        ds_alice = _make_strategy()
        ds_alice.setup_distributed(dist_backend="gloo")
        alice_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_alice.prepare(alice_model)
        alice_model.gradient_checkpointing_enable()
        # Verify gradient checkpointing is active on the underlying HF model
        _inner = getattr(alice_model.model, 'model', alice_model.model)
        _gc_active = getattr(_inner, 'gradient_checkpointing', False)
        _gc_func = getattr(_inner, '_gradient_checkpointing_func', None)
        logger.info("Alice model on GPU 2: gc=%s, has_gc_func=%s, training=%s",
                     _gc_active, _gc_func is not None, _inner.training)

        # Bob model on GPU 2 (same GPU, independent weights + optimizer)
        ds_bob = _make_strategy()
        bob_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_bob.prepare(bob_model)
        bob_model.gradient_checkpointing_enable()
        logger.info("Bob model on GPU 2 (independent weights + optimizer)")

        # Reference model on GPU 2, then park on CPU
        ref_strategy = _make_strategy()
        ref_model_actor = Actor(tcfg.model_name, **actor_kwargs)
        ref_strategy.prepare(ref_model_actor, is_rlhf=True)
        ref_model = ref_strategy
        ref_module = getattr(ref_strategy, '_engine', ref_strategy)
        ref_module = getattr(ref_module, 'module', ref_module)
        ref_module.to('cpu')
        _torch.cuda.empty_cache()
        logger.info("Reference model initialized and parked on CPU")

        # Coach model on GPU 3
        os.environ["LOCAL_RANK"] = "3"
        _torch.cuda.set_device(3)
        ds_coach = _make_strategy(learning_rate=coach_lr)
        coach_model_actor = Actor(coach_model, **actor_kwargs)
        ds_coach.prepare(coach_model_actor)
        coach_model_actor.gradient_checkpointing_enable()
        _inner_c = getattr(coach_model_actor.model, 'model', coach_model_actor.model)
        _gc_active_c = getattr(_inner_c, 'gradient_checkpointing', False)
        _gc_func_c = getattr(_inner_c, '_gradient_checkpointing_func', None)
        logger.info("Coach model on GPU 3: gc=%s, has_gc_func=%s, training=%s",
                     _gc_active_c, _gc_func_c is not None, _inner_c.training)

        # Reset default device to GPU 2 (player training runs first each iter)
        os.environ["LOCAL_RANK"] = "2"
        _torch.cuda.set_device(2)

    elif icfg.num_gpus_per_node > 2:
        # --- Multi-GPU distributed path (5+ GPUs) ---
        from crisp.infra.distributed import CRISPModelActor, DistributedStrategy
        from crisp.infra.ray_launcher import RayActorGroup

        strategy_kwargs = dict(
            seed=icfg.seed, bf16=icfg.bf16, zero_stage=icfg.zero_stage,
            adam_offload=icfg.adam_offload, max_norm=icfg.max_norm,
            learning_rate=icfg.learning_rate, weight_decay=icfg.weight_decay,
            micro_train_batch_size=icfg.micro_train_batch_size,
            gradient_checkpointing=icfg.gradient_checkpointing,
            attn_implementation=tcfg.attn_implementation,
            ref_reward_offload=tcfg.ref_reward_offload,
        )
        coach_strategy_kwargs = {**strategy_kwargs, "learning_rate": coach_lr}

        # Alice
        alice_group = RayActorGroup(
            num_nodes=icfg.num_nodes, num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        ray.get(alice_group.async_init_model_from_pretrained(
            strategy_kwargs=strategy_kwargs, pretrain=tcfg.model_name,
            actor_kwargs=actor_kwargs,
        ))
        ds_alice = DistributedStrategy(alice_group)

        # Bob
        bob_group = RayActorGroup(
            num_nodes=icfg.num_nodes, num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        ray.get(bob_group.async_init_model_from_pretrained(
            strategy_kwargs=strategy_kwargs, pretrain=tcfg.model_name,
            actor_kwargs=actor_kwargs,
        ))
        ds_bob = DistributedStrategy(bob_group)

        # Coach
        coach_group = RayActorGroup(
            num_nodes=icfg.num_nodes, num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        ray.get(coach_group.async_init_model_from_pretrained(
            strategy_kwargs=coach_strategy_kwargs, pretrain=coach_model,
            actor_kwargs=actor_kwargs,
        ))
        ds_coach = DistributedStrategy(coach_group)

        # Reference model (frozen)
        ref_group = RayActorGroup(
            num_nodes=icfg.num_nodes, num_gpus_per_node=icfg.num_gpus_per_node,
            ray_actor_type=CRISPModelActor,
        )
        ray.get(ref_group.async_init_model_from_pretrained(
            strategy_kwargs=strategy_kwargs, pretrain=tcfg.model_name,
            actor_kwargs=actor_kwargs, is_rlhf=True,
        ))
        ref_model = DistributedStrategy(ref_group)

        logger.info("Multi-GPU: alice, bob, coach, ref initialized on %d GPUs",
                     icfg.num_gpus_per_node)
    else:
        # --- Single/2-GPU path ---
        from crisp.infra.actor_model import Actor

        # 4. Alice model
        ds_alice = _make_strategy()
        ds_alice.setup_distributed()
        alice_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_alice.prepare(alice_model)
        alice_model.gradient_checkpointing_enable()
        logger.info("Alice model initialized (gradient checkpointing ON)")

        # 4b. Bob model (same GPU, independent weights + optimizer)
        ds_bob = _make_strategy()
        bob_model = Actor(tcfg.model_name, **actor_kwargs)
        ds_bob.prepare(bob_model)
        bob_model.gradient_checkpointing_enable()
        logger.info("Bob model initialized (gradient checkpointing ON)")

        # 5. Coach model (same architecture, independent weights)
        ds_coach = _make_strategy(learning_rate=coach_lr)
        coach_model_actor = Actor(coach_model, **actor_kwargs)
        ds_coach.prepare(coach_model_actor)
        coach_model_actor.gradient_checkpointing_enable()
        logger.info("Coach model initialized (gradient checkpointing ON)")

        # 6. Reference model (frozen, local for single-GPU)
        ref_strategy = _make_strategy()
        ref_model_actor = Actor(tcfg.model_name, **actor_kwargs)
        ref_strategy.prepare(ref_model_actor, is_rlhf=True)
        ref_model = ref_strategy
        logger.info("Reference model initialized (frozen)")

        # Park ref model on CPU to free ~8 GB GPU memory.
        # Stage=0 (no optimizer state) so simple .to('cpu') works.
        # _chunked_ref_forward will move it back to GPU temporarily.
        ref_module = getattr(ref_strategy, '_engine', ref_strategy)
        ref_module = getattr(ref_module, 'module', ref_module)
        ref_module.to('cpu')
        import torch as _torch_park
        _torch_park.cuda.empty_cache()
        logger.info("Reference model parked on CPU")

    # 7. Stateful objects
    acfg = config.advantage
    ccfg = config.coach
    alice_ema = EMATracker(
        mu=acfg.ema_init_mu,
        sigma_sq=acfg.ema_init_sigma_sq,
        eta=acfg.ema_eta,
    )
    bob_ema = EMATracker(
        mu=acfg.ema_init_mu,
        sigma_sq=acfg.ema_init_sigma_sq,
        eta=acfg.ema_eta,
    )
    coach_ema = EMATracker(
        mu=acfg.coach_ema_init_mu,
        sigma_sq=acfg.coach_ema_init_sigma_sq,
        eta=acfg.ema_eta,
    )
    rep_buffer = RepetitionBuffer(
        max_batches=ccfg.repetition_window,
        embedding_dim=ccfg.embedding_dim,
    )

    # 8. Tokenizer (for pad_token_id)
    from crisp.workflow.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(tcfg.model_name)
    coach_tokenizer = get_tokenizer(coach_model) if coach_model != tcfg.model_name else None

    return WorkflowContext(
        player_vllm=player_vllm,
        coach_vllm=coach_vllm,
        ref_model=ref_model,
        ds_alice=ds_alice,
        ds_bob=ds_bob,
        ds_coach=ds_coach,
        config=config,
        alice_ema=alice_ema,
        bob_ema=bob_ema,
        coach_ema=coach_ema,
        rep_buffer=rep_buffer,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        tokenizer=tokenizer,
        coach_tokenizer=coach_tokenizer,
    )


def save_checkpoint(ctx: Any, path: str) -> None:
    """Save model checkpoints via DeepSpeed.

    Uses separate subdirectories for alice, bob, and coach to avoid
    DeepSpeed ``latest`` file conflicts. Tag is "ckpt" (not "latest")
    because DeepSpeed writes a *file* called ``latest`` to record the
    current tag — using "latest" as the tag creates a directory that
    clashes with that file.
    """
    import shutil
    os.makedirs(path, exist_ok=True)
    alice_path = os.path.join(path, "alice")
    bob_path = os.path.join(path, "bob")
    coach_path = os.path.join(path, "coach")
    # DeepSpeed writes a *file* called "latest" to record the active tag.
    # If a previous run used tag="latest", a *directory* named "latest"
    # exists instead, causing IsADirectoryError. Remove it defensively.
    for sub in (alice_path, bob_path, coach_path):
        latest = os.path.join(sub, "latest")
        if os.path.isdir(latest):
            shutil.rmtree(latest)
            logger.warning("Removed stale 'latest' directory: %s", latest)
    if hasattr(ctx.ds_alice, "_engine") and ctx.ds_alice._engine is not None:
        ctx.ds_alice._engine.save_checkpoint(
            alice_path, tag="ckpt",
            client_state={"iteration": ctx.iteration},
        )
    if hasattr(ctx.ds_bob, "_engine") and ctx.ds_bob._engine is not None:
        ctx.ds_bob._engine.save_checkpoint(bob_path, tag="ckpt")
    if hasattr(ctx.ds_coach, "_engine") and ctx.ds_coach._engine is not None:
        ctx.ds_coach._engine.save_checkpoint(coach_path, tag="ckpt")
    logger.info("Checkpoint saved at iteration %d to %s", ctx.iteration, path)


def load_checkpoint(ctx: Any, path: str) -> None:
    """Load model checkpoints and restore iteration counter.

    Tries tag=None first (reads ``latest`` marker file written by
    DeepSpeed). Falls back to explicit tags "ckpt" then "latest"
    if the marker file is missing (e.g. after a crash during save).

    Backwards compat: if "alice" subdir doesn't exist but "player" does,
    load the legacy player checkpoint as Alice (Bob starts from scratch).
    """
    alice_path = os.path.join(path, "alice")
    bob_path = os.path.join(path, "bob")
    coach_path = os.path.join(path, "coach")

    # Backwards compat: legacy checkpoints have "player" instead of "alice"
    if not os.path.isdir(alice_path) and os.path.isdir(os.path.join(path, "player")):
        alice_path = os.path.join(path, "player")
        logger.info("Legacy checkpoint: using 'player' dir as Alice")

    def _load(engine, ckpt_path):
        """Try loading checkpoint with auto-detection, then explicit tags."""
        # Try auto-detect first (reads 'latest' marker file)
        try:
            return engine.load_checkpoint(ckpt_path, tag=None)
        except Exception:
            pass
        # Fallback: try known tag names
        for tag in ("ckpt", "latest"):
            tag_dir = os.path.join(ckpt_path, tag)
            if os.path.isdir(tag_dir):
                logger.info("Loading checkpoint with explicit tag=%s from %s", tag, ckpt_path)
                return engine.load_checkpoint(ckpt_path, tag=tag)
        logger.warning("No checkpoint found at %s", ckpt_path)
        return None, None

    # Load Alice (get iteration from client_state)
    if hasattr(ctx.ds_alice, "_engine") and ctx.ds_alice._engine is not None:
        if os.path.isdir(alice_path):
            _, client_state = _load(ctx.ds_alice._engine, alice_path)
            if client_state and "iteration" in client_state:
                ctx.iteration = client_state["iteration"]
                logger.info("Resumed Alice from iteration %d", ctx.iteration)

    # Load Bob (no client_state needed)
    if hasattr(ctx.ds_bob, "_engine") and ctx.ds_bob._engine is not None:
        if os.path.isdir(bob_path):
            _load(ctx.ds_bob._engine, bob_path)
        else:
            logger.warning("No Bob checkpoint at %s — starting Bob from scratch", bob_path)

    # Load coach
    if hasattr(ctx.ds_coach, "_engine") and ctx.ds_coach._engine is not None:
        if os.path.isdir(coach_path):
            _load(ctx.ds_coach._engine, coach_path)
        else:
            logger.warning("No coach checkpoint at %s — starting coach from scratch", coach_path)


def save_hf_players(ctx: Any, path: str) -> None:
    """Save only player (Alice/Bob) weights in HF format.

    Lightweight alternative to save_hf_weights() for periodic mid-training
    saves. Players are ~8 GB each (16 GB total) vs ~44 GB for all three.
    """
    alice_path = os.path.join(path, "alice_hf")
    bob_path = os.path.join(path, "bob_hf")
    os.makedirs(alice_path, exist_ok=True)
    os.makedirs(bob_path, exist_ok=True)

    alice_model = ctx.ds_alice._engine.module.model
    bob_model = ctx.ds_bob._engine.module.model

    # Restore inference config before saving (training sets use_cache=False)
    for model in (alice_model, bob_model):
        model.config.use_cache = True

    alice_model.save_pretrained(alice_path, safe_serialization=True)
    bob_model.save_pretrained(bob_path, safe_serialization=True)
    ctx.tokenizer.save_pretrained(alice_path)
    ctx.tokenizer.save_pretrained(bob_path)

    # Restore training config
    for model in (alice_model, bob_model):
        model.config.use_cache = False

    with open(os.path.join(path, "iteration.txt"), "w") as f:
        f.write(str(ctx.iteration))
    logger.info("Saved player HF weights at iteration %d to %s (~16 GB)", ctx.iteration, path)


def save_hf_weights(ctx: Any, path: str) -> None:
    """Save model weights in HuggingFace format for portability.

    These can be loaded on any cluster via model_name config override,
    or used directly with vLLM for evaluation. Saves alice_hf/, bob_hf/,
    and coach_hf/ subdirectories.
    """
    import torch
    alice_path = os.path.join(path, "alice_hf")
    bob_path = os.path.join(path, "bob_hf")
    coach_path = os.path.join(path, "coach_hf")
    os.makedirs(alice_path, exist_ok=True)
    os.makedirs(bob_path, exist_ok=True)
    os.makedirs(coach_path, exist_ok=True)

    # Extract HF models from DeepSpeed wrappers
    alice_model = ctx.ds_alice._engine.module.model
    bob_model = ctx.ds_bob._engine.module.model
    coach_model = ctx.ds_coach._engine.module.model

    # Restore inference config before saving (training sets use_cache=False)
    for model in (alice_model, bob_model, coach_model):
        model.config.use_cache = True

    # Save with safe_serialization
    alice_model.save_pretrained(alice_path, safe_serialization=True)
    bob_model.save_pretrained(bob_path, safe_serialization=True)
    coach_model.save_pretrained(coach_path, safe_serialization=True)
    ctx.tokenizer.save_pretrained(alice_path)
    ctx.tokenizer.save_pretrained(bob_path)
    if ctx.coach_tokenizer is not None:
        ctx.coach_tokenizer.save_pretrained(coach_path)
    else:
        ctx.tokenizer.save_pretrained(coach_path)

    # Restore training config
    for model in (alice_model, bob_model, coach_model):
        model.config.use_cache = False

    # Write iteration marker
    with open(os.path.join(path, "iteration.txt"), "w") as f:
        f.write(str(ctx.iteration))

    logger.info("HF weights saved at iteration %d to %s", ctx.iteration, path)


def _load_eval_problems(dataset: str) -> list:
    """Load evaluation problems by dataset name."""
    if dataset == "aime24":
        return load_aime24_problems()
    elif dataset == "aime25":
        return load_aime25_problems()
    elif dataset == "dapo":
        return load_dapo_problems()
    else:
        raise ValueError(f"Unknown eval dataset: {dataset!r} (expected aime24/aime25/dapo)")


def run_evaluation(ctx: Any) -> float:
    """Run evaluation on configured dataset and log results.

    Returns pass@1 score for best-checkpoint tracking.
    """
    tcfg = ctx.config.training
    all_problems = _load_eval_problems(tcfg.eval_dataset)

    # Deterministic sample seeded by iteration
    import random
    rng = random.Random(tcfg.eval_n_problems + ctx.iteration)
    n = min(tcfg.eval_n_problems, len(all_problems))
    problems = rng.sample(all_problems, n)

    result = evaluate_on_problems(
        problems, ctx.player_vllm, ctx.tokenizer,
        n_samples=tcfg.eval_n_samples,
    )

    pass_at_1 = bayesian_pass_at_n(
        result["num_correct"], result["num_total"], n=1,
    )

    logger.info(
        "iter=%d eval[%s]: accuracy=%.3f pass@1=%.3f (n_problems=%d, n_samples=%d)",
        ctx.iteration, tcfg.eval_dataset, result["accuracy"], pass_at_1,
        n, tcfg.eval_n_samples,
    )
    return pass_at_1


def run(
    config: Any,
    resume_path: Optional[str] = None,
    resume_hf_path: Optional[str] = None,
    save_lora_path: Optional[str] = None,
    merge_lora_path: Optional[str] = None,
    save_hf_path: Optional[str] = None,
    save_hf_home: Optional[str] = None,
) -> None:
    """Main training loop."""
    import signal
    from crisp.workflow.main_loop import step

    # --resume-hf: override model paths before init so models load fine-tuned weights
    # Alice and Bob both start from the alice_hf weights (init_infra creates
    # independent copies from model_name). Bob's bob_hf is only used if
    # available — otherwise it starts from alice_hf too.
    if resume_hf_path:
        alice_hf = os.path.join(resume_hf_path, "alice_hf")
        bob_hf = os.path.join(resume_hf_path, "bob_hf")
        coach_hf = os.path.join(resume_hf_path, "coach_hf")

        # Backwards compat: legacy checkpoints have "player_hf" instead of "alice_hf"
        if not os.path.isdir(alice_hf) and os.path.isdir(os.path.join(resume_hf_path, "player_hf")):
            alice_hf = os.path.join(resume_hf_path, "player_hf")
            bob_hf = alice_hf  # Both start from same weights
            logger.info("resume-hf: legacy player_hf -> both Alice and Bob")

        if os.path.isdir(alice_hf):
            config.training.model_name = alice_hf
            logger.info("resume-hf: player model -> %s", alice_hf)
        # NOTE: Bob uses the same model_name since init_infra creates
        # independent copies. bob_hf is informational only for now.
        if os.path.isdir(coach_hf):
            config.training.coach_model_name = coach_hf
            logger.info("resume-hf: coach model -> %s", coach_hf)

    ctx = init_infra(config)

    if resume_path:
        load_checkpoint(ctx, resume_path)
        logger.info("Resumed from checkpoint at iteration %d", ctx.iteration)
    elif resume_hf_path:
        # Restore iteration counter from marker file
        iter_file = os.path.join(resume_hf_path, "iteration.txt")
        if os.path.isfile(iter_file):
            with open(iter_file) as f:
                ctx.iteration = int(f.read().strip())
            logger.info("Resumed from HF weights at iteration %d", ctx.iteration)
        else:
            logger.warning("No iteration.txt in %s — starting from iteration 0", resume_hf_path)
    elif config.training.start_iteration > 0:
        ctx.iteration = config.training.start_iteration
        logger.info("Starting from iteration %d (config)", ctx.iteration)

    # Seed accuracy history when resuming so the coach doesn't get the
    # CRITICAL "Students cannot solve ANY" prompt on early iterations.
    # Without this, a fresh optimizer produces 0% accuracy, triggering
    # the CRITICAL message which confuses coach formatting.
    if ctx.iteration > 0 and not ctx.accuracy_history:
        ctx.accuracy_history = [0.3] * 5
        logger.info("Seeded accuracy_history with moderate values for resume")

    tcfg = config.training

    # Graceful shutdown: save checkpoint + HF weights on SIGTERM
    # (Modal sends SIGTERM ~30s before killing the container)
    shutdown_requested = [False]

    def _sigterm_handler(signum, frame):
        logger.info("SIGTERM received at iteration %d — saving...", ctx.iteration)
        shutdown_requested[0] = True
        if save_hf_home:
            # Save player-only weights to persistent home dir (fast, ~16GB)
            save_hf_players(ctx, save_hf_home)
        elif save_hf_path:
            # Skip DS checkpoint (too large for quota) — save HF only
            save_hf_weights(ctx, save_hf_path)
        else:
            save_checkpoint(ctx, tcfg.checkpoint_dir)
            hf_path = os.path.join(tcfg.checkpoint_dir, "hf_weights")
            save_hf_weights(ctx, hf_path)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    logger.info("Starting CRISP training for %d iterations", tcfg.num_iterations)
    best_eval_score = -1.0

    for _ in range(tcfg.num_iterations):
        if shutdown_requested[0]:
            logger.info("Shutdown requested — exiting training loop")
            break

        result = step(ctx)

        if ctx.iteration % tcfg.log_freq == 0:
            logger.info(
                "iter=%d alice_loss=%.4f bob_loss=%.4f coach_loss=%s accuracy=%.3f "
                "problems=%d discussions=%d",
                ctx.iteration, result.alice_loss, result.bob_loss,
                f"{result.coach_loss:.4f}" if result.coach_loss is not None else "N/A",
                result.player_accuracy, result.num_problems, result.num_discussions,
            )

        if tcfg.eval_freq > 0 and ctx.iteration % tcfg.eval_freq == 0:
            eval_score = run_evaluation(ctx)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                logger.info(
                    "iter=%d NEW BEST eval score=%.3f — saving best checkpoint",
                    ctx.iteration, eval_score,
                )
                best_dir = (save_hf_home or save_hf_path or
                            os.path.join(tcfg.checkpoint_dir, "hf_weights"))
                best_dir = best_dir + "_best"
                os.makedirs(best_dir, exist_ok=True)
                save_hf_players(ctx, best_dir)
            else:
                logger.info(
                    "iter=%d eval score=%.3f (best=%.3f) — not saving",
                    ctx.iteration, eval_score, best_eval_score,
                )

        if tcfg.save_freq > 0 and ctx.iteration % tcfg.save_freq == 0:
            if save_hf_home:
                # Save player-only HF weights to persistent home dir (~16 GB)
                save_hf_players(ctx, save_hf_home)
            elif save_hf_path:
                # Save player-only HF weights to /tmp (~16 GB)
                save_hf_players(ctx, save_hf_path)
            else:
                save_checkpoint(ctx, tcfg.checkpoint_dir)

    # Final checkpoint + HF weights (all models)
    if tcfg.save_freq > 0 and not save_hf_path:
        save_checkpoint(ctx, tcfg.checkpoint_dir)
    hf_path = save_hf_path or os.path.join(tcfg.checkpoint_dir, "hf_weights")
    save_hf_weights(ctx, hf_path)

    # LoRA save/merge
    if save_lora_path:
        from crisp.infra.lora_utils import save_lora_adapters, merge_and_save
        save_lora_adapters(ctx.ds_alice, os.path.join(save_lora_path, "alice"))
        save_lora_adapters(ctx.ds_bob, os.path.join(save_lora_path, "bob"))
        save_lora_adapters(ctx.ds_coach, os.path.join(save_lora_path, "coach"))
        logger.info("LoRA adapters saved to %s", save_lora_path)

        if merge_lora_path:
            merge_and_save(
                os.path.join(save_lora_path, "alice"),
                os.path.join(merge_lora_path, "alice"),
                tcfg.model_name, tokenizer_name=tcfg.model_name,
            )
            merge_and_save(
                os.path.join(save_lora_path, "bob"),
                os.path.join(merge_lora_path, "bob"),
                tcfg.model_name, tokenizer_name=tcfg.model_name,
            )
            coach_model_name = tcfg.coach_model_name or tcfg.model_name
            merge_and_save(
                os.path.join(save_lora_path, "coach"),
                os.path.join(merge_lora_path, "coach"),
                coach_model_name, tokenizer_name=coach_model_name,
            )
            logger.info("Merged models saved to %s", merge_lora_path)

    logger.info("Training complete after %d iterations", tcfg.num_iterations)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    overrides = parse_overrides(args.override)

    from crisp.config_loader import load_config

    config = load_config(args.config, overrides=overrides)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        level=logging.INFO,
    )
    run(config, resume_path=args.resume, resume_hf_path=args.resume_hf,
        save_lora_path=args.save_lora, merge_lora_path=args.merge_lora,
        save_hf_path=args.save_hf, save_hf_home=args.save_hf_home)


if __name__ == "__main__":
    main()
