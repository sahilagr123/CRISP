"""Tests for the training entry point."""
from unittest.mock import MagicMock, patch

from crisp.config import CRISPConfig
from crisp.train import parse_args, parse_overrides, save_checkpoint
from crisp.workflow.context import StepResult


def test_parse_args():
    """parse_args handles --config and --override."""
    args = parse_args(["--config", "foo.yaml"])
    assert args.config == "foo.yaml"
    assert args.override == []

    args = parse_args(["--config", "foo.yaml", "--override", "a.b=1", "c.d=true"])
    assert args.override == ["a.b=1", "c.d=true"]


def test_parse_overrides_types():
    """parse_overrides correctly parses int, float, bool, and string values."""
    result = parse_overrides(["a.b=42", "c.d=0.001", "e.f=true", "g.h=false", "i.j=hello"])
    assert result == {
        "a.b": 42,
        "c.d": 0.001,
        "e.f": True,
        "g.h": False,
        "i.j": "hello",
    }


def test_parse_overrides_empty():
    """parse_overrides handles empty list."""
    assert parse_overrides([]) == {}


def test_init_infra_mock():
    """init_infra creates a WorkflowContext with all fields populated."""
    import sys
    from crisp.train import init_infra
    from crisp.workflow.context import WorkflowContext

    config = CRISPConfig()

    mock_ray = MagicMock()
    mock_strategy_instance = MagicMock()
    mock_strategy_instance._engine = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    with patch.dict(sys.modules, {"ray": mock_ray}), \
         patch("crisp.infra.vllm_engine.create_vllm_engines",
               return_value=[MagicMock()]) as mock_vllm, \
         patch("crisp.infra.strategy.DeepSpeedStrategy",
               return_value=mock_strategy_instance) as MockStrategy, \
         patch("crisp.infra.actor_model.Actor") as MockActor, \
         patch("crisp.workflow.tokenizer.get_tokenizer",
               return_value=mock_tokenizer):

        ctx = init_infra(config)

    # Ray initialized
    mock_ray.init.assert_called_once_with(ignore_reinit_error=True)

    # vLLM engine created once (player only — coach uses HF generate
    # when coach model differs from player on single GPU)
    assert mock_vllm.call_count == 1

    # 4 strategies created (alice, bob, coach, ref)
    assert MockStrategy.call_count == 4

    # 4 Actor models created (alice, bob, coach, ref)
    assert MockActor.call_count == 4

    # setup_distributed called once (on first strategy)
    mock_strategy_instance.setup_distributed.assert_called_once()

    # prepare called 4 times (alice, bob, coach, ref)
    assert mock_strategy_instance.prepare.call_count == 4

    # Result is a WorkflowContext
    assert isinstance(ctx, WorkflowContext)
    assert ctx.iteration == 0
    assert len(ctx.player_vllm) == 1
    assert ctx.coach_vllm is None  # different model on single GPU


def test_run_loop_mock():
    """run() calls step() for num_iterations times."""
    from crisp.train import run

    config = CRISPConfig()
    config.training.num_iterations = 3
    config.training.save_freq = 0  # disable checkpoints

    mock_result = MagicMock()
    mock_result.alice_loss = 0.5
    mock_result.bob_loss = 0.3
    mock_result.coach_loss = None
    mock_result.player_accuracy = 0.7
    mock_result.num_problems = 4
    mock_result.num_discussions = 1

    mock_ctx = MagicMock()
    mock_ctx.iteration = 0

    with patch("crisp.workflow.main_loop.step", return_value=mock_result) as mock_step, \
         patch("crisp.train.init_infra", return_value=mock_ctx):
        run(config)

    assert mock_step.call_count == 3


def test_save_checkpoint_calls_engine():
    """save_checkpoint delegates to DeepSpeed engine.save_checkpoint."""
    mock_ctx = MagicMock()
    mock_ctx.iteration = 5
    mock_ctx.ds_alice._engine = MagicMock()
    mock_ctx.ds_bob._engine = MagicMock()
    mock_ctx.ds_coach._engine = MagicMock()

    with patch("crisp.train.os.makedirs"):
        save_checkpoint(mock_ctx, "/tmp/test_ckpt")

    mock_ctx.ds_alice._engine.save_checkpoint.assert_called_once_with(
        "/tmp/test_ckpt/alice", tag="ckpt",
        client_state={"iteration": 5},
    )
    mock_ctx.ds_bob._engine.save_checkpoint.assert_called_once_with(
        "/tmp/test_ckpt/bob", tag="ckpt",
    )
    mock_ctx.ds_coach._engine.save_checkpoint.assert_called_once_with(
        "/tmp/test_ckpt/coach", tag="ckpt"
    )


def test_parse_args_resume():
    """parse_args handles --resume."""
    args = parse_args(["--config", "f.yaml", "--resume", "checkpoints/debug"])
    assert args.resume == "checkpoints/debug"

    # Default is None
    args = parse_args(["--config", "f.yaml"])
    assert args.resume is None


def test_load_checkpoint_restores_iteration():
    """load_checkpoint restores iteration from client_state."""
    from crisp.train import load_checkpoint

    mock_ctx = MagicMock()
    mock_ctx.iteration = 0
    mock_ctx.ds_alice._engine = MagicMock()
    mock_ctx.ds_bob._engine = MagicMock()
    mock_ctx.ds_coach._engine = MagicMock()

    # Simulate load returning client_state with iteration
    mock_ctx.ds_alice._engine.load_checkpoint.return_value = (
        "checkpoints/debug",
        {"iteration": 42},
    )

    with patch("crisp.train.os.path.isdir", return_value=True):
        load_checkpoint(mock_ctx, "checkpoints/debug")

    assert mock_ctx.iteration == 42
    mock_ctx.ds_alice._engine.load_checkpoint.assert_called_once_with(
        "checkpoints/debug/alice", tag=None,
    )
    mock_ctx.ds_bob._engine.load_checkpoint.assert_called_once_with(
        "checkpoints/debug/bob", tag=None,
    )
    mock_ctx.ds_coach._engine.load_checkpoint.assert_called_once_with(
        "checkpoints/debug/coach", tag=None,
    )


def test_save_checkpoint_includes_client_state():
    """save_checkpoint saves iteration in client_state."""
    mock_ctx = MagicMock()
    mock_ctx.iteration = 7
    mock_ctx.ds_alice._engine = MagicMock()
    mock_ctx.ds_bob._engine = MagicMock()
    mock_ctx.ds_coach._engine = MagicMock()

    with patch("crisp.train.os.makedirs"):
        save_checkpoint(mock_ctx, "/tmp/test_ckpt")

    mock_ctx.ds_alice._engine.save_checkpoint.assert_called_once_with(
        "/tmp/test_ckpt/alice", tag="ckpt",
        client_state={"iteration": 7},
    )


def test_parse_args_lora():
    """parse_args handles --save-lora and --merge-lora."""
    args = parse_args(["--config", "f.yaml", "--save-lora", "/tmp/adapters",
                       "--merge-lora", "/tmp/merged"])
    assert args.save_lora == "/tmp/adapters"
    assert args.merge_lora == "/tmp/merged"

    # Defaults to None
    args = parse_args(["--config", "f.yaml"])
    assert args.save_lora is None
    assert args.merge_lora is None


def test_run_saves_lora_at_end():
    """run() calls save_lora_adapters and merge_and_save when flags set."""
    from crisp.train import run

    config = CRISPConfig()
    config.training.num_iterations = 1
    config.training.save_freq = 0

    mock_result = MagicMock()
    mock_result.alice_loss = 0.5
    mock_result.bob_loss = 0.3
    mock_result.coach_loss = None
    mock_result.player_accuracy = 0.7
    mock_result.num_problems = 4
    mock_result.num_discussions = 1

    mock_ctx = MagicMock()
    mock_ctx.iteration = 0

    with patch("crisp.workflow.main_loop.step", return_value=mock_result), \
         patch("crisp.train.init_infra", return_value=mock_ctx), \
         patch("crisp.infra.lora_utils.save_lora_adapters") as mock_save, \
         patch("crisp.infra.lora_utils.merge_and_save") as mock_merge:
        run(config, save_lora_path="/tmp/adapters", merge_lora_path="/tmp/merged")

    # Alice, Bob, and coach adapters saved
    assert mock_save.call_count == 3
    # Merge called for alice, bob, and coach
    assert mock_merge.call_count == 3


def test_init_infra_two_gpu():
    """init_infra creates shared-GPU coach vLLM and pins training to GPU 1."""
    import os
    import sys
    from crisp.train import init_infra
    from crisp.workflow.context import WorkflowContext

    config = CRISPConfig()
    config.infra.num_gpus_per_node = 2

    mock_strategy_instance = MagicMock()
    mock_strategy_instance._engine = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    mock_engine = MagicMock()
    mock_engines = [mock_engine]

    mock_pg = MagicMock()
    mock_pg.ready.return_value = MagicMock()  # ray.get needs something

    old_lr = os.environ.get("LOCAL_RANK")
    try:
        with patch("ray.init"), \
             patch("ray.get"), \
             patch("ray.util.placement_group.placement_group",
                   return_value=mock_pg), \
             patch("crisp.infra.vllm_engine.create_vllm_engines",
                   return_value=mock_engines), \
             patch("crisp.infra.strategy.DeepSpeedStrategy",
                   return_value=mock_strategy_instance), \
             patch("crisp.infra.actor_model.Actor"), \
             patch("crisp.workflow.tokenizer.get_tokenizer",
                   return_value=mock_tokenizer), \
             patch("torch.cuda.set_device"), \
             patch("torch.cuda.empty_cache"):

            ctx = init_infra(config)

        # GPU pinning applied via LOCAL_RANK
        assert os.environ.get("LOCAL_RANK") == "1"
        # Single-process path: 4 strategies (alice, bob, coach, ref), no RayActorGroup
        assert mock_strategy_instance.prepare.call_count == 4
        assert isinstance(ctx, WorkflowContext)
        # Coach vLLM created (shared GPU), not None
        assert ctx.coach_vllm is not None
    finally:
        # Restore env
        if old_lr is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = old_lr


def test_init_infra_four_gpu():
    """init_infra creates split-GPU training (player GPU 2, coach GPU 3)."""
    import os
    from crisp.train import init_infra
    from crisp.workflow.context import WorkflowContext

    config = CRISPConfig()
    config.infra.num_gpus_per_node = 4

    mock_strategy_instance = MagicMock()
    mock_strategy_instance._engine = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    player_engines = [MagicMock()]
    coach_engines = [MagicMock()]

    old_lr = os.environ.get("LOCAL_RANK")
    try:
        with patch("ray.init"), \
             patch("ray.get"), \
             patch("crisp.infra.vllm_engine.create_vllm_engines",
                   side_effect=[player_engines, coach_engines]), \
             patch("crisp.infra.strategy.DeepSpeedStrategy",
                   return_value=mock_strategy_instance), \
             patch("crisp.infra.actor_model.Actor"), \
             patch("crisp.workflow.tokenizer.get_tokenizer",
                   return_value=mock_tokenizer), \
             patch("torch.cuda.set_device"), \
             patch("torch.cuda.empty_cache"):

            ctx = init_infra(config)

        # GPU pinning: LOCAL_RANK ends at "2" (reset after coach init on GPU 3)
        assert os.environ.get("LOCAL_RANK") == "2"
        # Single-process path: 4 strategies (alice, bob, ref, coach)
        assert mock_strategy_instance.prepare.call_count == 4
        assert isinstance(ctx, WorkflowContext)
        # Coach vLLM created (separate engine), not None
        assert ctx.coach_vllm is not None
        assert ctx.coach_vllm is not ctx.player_vllm
    finally:
        if old_lr is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = old_lr


def test_init_infra_multi_gpu():
    """init_infra creates RayActorGroup + DistributedStrategy when num_gpus > 4."""
    import sys
    from crisp.train import init_infra
    from crisp.workflow.context import WorkflowContext

    config = CRISPConfig()
    config.infra.num_gpus_per_node = 5

    mock_ray = MagicMock()
    mock_group = MagicMock()
    mock_group.async_init_model_from_pretrained.return_value = [MagicMock()]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    with patch.dict(sys.modules, {"ray": mock_ray}), \
         patch("crisp.infra.vllm_engine.create_vllm_engines",
               return_value=[MagicMock()]), \
         patch("crisp.infra.ray_launcher.RayActorGroup",
               return_value=mock_group) as MockGroup, \
         patch("crisp.infra.distributed.DistributedStrategy") as MockDS, \
         patch("crisp.workflow.tokenizer.get_tokenizer",
               return_value=mock_tokenizer):

        ctx = init_infra(config)

    # RayActorGroup created 4 times (alice, bob, coach, ref)
    assert MockGroup.call_count == 4
    # DistributedStrategy wraps each group
    assert MockDS.call_count == 4
    assert isinstance(ctx, WorkflowContext)


def test_run_calls_evaluate_at_eval_freq():
    """run() calls run_evaluation every eval_freq iterations."""
    from crisp.config import CRISPConfig
    from crisp.train import run

    config = CRISPConfig()
    config.training.num_iterations = 6
    config.training.eval_freq = 3
    config.training.save_freq = 0

    mock_step_result = StepResult(
        alice_loss=0.1, bob_loss=0.1, coach_loss=0.05,
        num_problems=8, num_discussions=2,
        player_accuracy=0.6, coach_iteration=True,
    )

    with patch("crisp.train.init_infra") as mock_init, \
         patch("crisp.workflow.main_loop.step", return_value=mock_step_result) as mock_step, \
         patch("crisp.train.run_evaluation", return_value=0.5) as mock_eval, \
         patch("crisp.train.save_hf_players"):
        mock_ctx = MagicMock()
        mock_ctx.config = config
        mock_ctx.iteration = 0
        mock_init.return_value = mock_ctx
        run(config)

    assert mock_eval.call_count >= 1


def test_run_evaluation_calls_evaluate():
    """run_evaluation loads problems and calls evaluate_on_problems."""
    from crisp.train import run_evaluation
    from crisp.config import CRISPConfig

    config = CRISPConfig()
    config.training.eval_n_problems = 50
    config.training.eval_n_samples = 4

    ctx = MagicMock()
    ctx.config = config
    ctx.iteration = 10
    ctx.player_vllm = [MagicMock()]
    ctx.tokenizer = MagicMock()

    mock_problems = [MagicMock() for _ in range(100)]
    mock_eval_result = {"accuracy": 0.75, "num_correct": [1]*50, "num_total": [4]*50}

    with patch("crisp.train.load_dapo_problems", return_value=mock_problems) as mock_load, \
         patch("crisp.train.evaluate_on_problems", return_value=mock_eval_result) as mock_eval, \
         patch("crisp.train.bayesian_pass_at_n", return_value=0.8):
        run_evaluation(ctx)

    mock_load.assert_called_once()
    mock_eval.assert_called_once()
    # Should have sampled eval_n_problems (50) from the 100
    call_args = mock_eval.call_args
    assert len(call_args[0][0]) == 50 or len(call_args[1].get("problems", call_args[0][0])) == 50
