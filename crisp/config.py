"""Hyperparameter configuration for CRISP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerConfig:
    """Per-player training hyperparameters."""
    rollouts_per_problem: int = 8
    solve_reward: float = 1.0
    wrong_reward: float = 0.0
    no_box_penalty: float = -0.5
    persuader_bonus: float = 0.3  # gamma_1

    # Per-player personalities (Alice=0, Bob=1)
    alice_temperature: float = 0.8
    bob_temperature: float = 1.0
    alice_seed_offset: int = 0
    bob_seed_offset: int = 1000
    alice_system_prompt: str = (
        "You are a methodical math solver. Work through problems step by step: "
        "define variables, set up equations, and solve algebraically. "
        "Show your work, then put your final numerical answer within \\boxed{}. "
        "You MUST end your response with \\boxed{} containing your numerical answer."
    )
    bob_system_prompt: str = (
        "You are a creative math solver. Look for patterns, try different "
        "approaches, and seek elegant shortcuts. "
        "Show your work, then put your final numerical answer within \\boxed{}. "
        "You MUST end your response with \\boxed{} containing your numerical answer."
    )
    max_new_tokens: int = 4096
    solve_prompt_template: str = (
        "{problem}\n\n"
        "Solve this problem. Show your reasoning, then give your final answer "
        "inside \\boxed{{}}."
    )


@dataclass
class CoachConfig:
    """Coach training hyperparameters."""
    batch_size: int = 8
    coach_temperature: float = 0.7  # lower than player for reliable instruction-following
    discussion_alpha: float = 0.3
    repetition_lambda: float = 2.5
    repetition_tau: float = 0.85
    too_hard_penalty: float = -0.2
    too_easy_penalty: float = -0.3  # Applied when p_hat >= too_easy_threshold
    too_easy_threshold: float = 1.0  # All rollouts correct
    unsolvable_penalty: float = -0.5  # Coach can't solve its own problem
    repetition_window: int = 10  # W batches
    embedding_dim: int = 384
    update_freq: int = 1  # Coach trains every iteration
    coach_solve_max_new_tokens: int = 4096  # Token budget for coach self-solve (thinking mode)
    warmup_iters: int = 25  # Phase 1: AMC 10 anchor (first N iters)
    rampup_iters: int = 100  # Phase 2: AMC 12 anchor (iters warmup..rampup)
    discussion_system_prompt: str = (
        "You are a math student reviewing two different solutions to the same problem. "
        "Carefully check each solution for errors in reasoning or computation. "
        "Then provide your own final answer inside \\boxed{}."
    )
    discussion_template: str = (
        "Problem:\n{problem}\n\n"
        "Solution A:\n{own_solution}\n\n"
        "Solution B:\n{other_solution}\n\n"
        "Carefully analyze both solutions. Identify any errors in reasoning "
        "or computation. Which solution (if either) is correct?\n\n"
        "You MUST end with your final numerical answer inside \\boxed{{}}."
    )
    coach_system_prompt: str = (
        "You are a math teacher designing problems for students. "
        "Start at AMC 10 difficulty and adjust based on accuracy feedback.\n\n"
        "Problems MUST have a single numerical answer "
        "(an integer or simple fraction). DO NOT generate problems whose answer "
        "is a function, expression, proof, or multiple values.\n\n"
        "BAD (avoid): functional equations, abstract algebra, Kolmogorov "
        "algebras, research-level analysis, multi-page proofs."
    )
    coach_rampup_system_prompt: str = (
        "You are a math teacher designing problems for students. "
        "Target AMC 12 / early AIME difficulty and adjust based on accuracy "
        "feedback below.\n\n"
        "Problems MUST have a single numerical answer "
        "(an integer or simple fraction). DO NOT generate problems whose answer "
        "is a function, expression, proof, or multiple values.\n\n"
        "BAD (avoid): functional equations, abstract algebra, Kolmogorov "
        "algebras, research-level analysis, multi-page proofs."
    )
    coach_post_warmup_system_prompt: str = (
        "You are a math teacher designing problems for students. "
        "Target a 40-60% solve rate — adjust difficulty based on accuracy "
        "feedback below.\n\n"
        "Problems MUST have a single numerical answer "
        "(an integer or simple fraction). DO NOT generate problems whose answer "
        "is a function, expression, proof, or multiple values."
    )
    coach_prompt_template: str = (
        "Generate one math problem about {topic}. "
        "The answer must be a single integer or simple fraction.\n\n"
        "Output exactly:\n"
        "<question>\n[PROBLEM]\n</question>\n\n"
        "Do NOT include a solution or answer — only the problem statement."
    )
    coach_solve_prompt_template: str = (
        "Solve this math problem step by step. "
        "Double-check your arithmetic before giving your final answer.\n\n"
        "{problem}\n\n"
        "Put your final numerical answer inside \\boxed{{}}."
    )
    coach_solve_system_prompt: str = (
        "You are an expert mathematician. Solve the problem carefully and verify "
        "your answer. The answer should be a single integer or simple fraction."
    )


@dataclass
class AdvantageConfig:
    """Advantage computation hyperparameters."""
    epsilon: float = 1e-8
    ema_eta: float = 0.2
    ema_init_mu: float = 0.5
    ema_init_sigma_sq: float = 0.25
    coach_ema_init_mu: float = 0.0  # Coach rewards are in [-0.5, 1.0], not [0, 1]
    coach_ema_init_sigma_sq: float = 0.25
    empty_pool_warning_threshold: int = 5


@dataclass
class GRPOConfig:
    """GRPO loss hyperparameters."""
    dcpo_alpha: float = 3.0
    clip_low: float = 0.2
    clip_high: float = 0.28  # asymmetric clip-higher (DAPO): prevents entropy collapse
    js_beta: float = 0.005  # player JS-divergence coefficient
    coach_js_beta: float = 0.1  # coach needs stronger anchoring (self-referencing)
    pre_discussion_l_max: int = 8192
    pre_discussion_buffer: int = 2048
    post_discussion_l_max: int = 4096
    post_discussion_buffer: int = 1024


@dataclass
class InfraConfig:
    """Infrastructure hyperparameters for Ray/vLLM/DeepSpeed."""
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    seed: int = 42
    zero_stage: int = 2
    bf16: bool = True
    adam_offload: bool = False
    gradient_checkpointing: bool = True
    micro_train_batch_size: int = 1
    max_norm: float = 1.0
    learning_rate: float = 1e-6
    coach_learning_rate: Optional[float] = None  # Defaults to learning_rate / 2
    weight_decay: float = 0.01
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    coach_vllm_gpu_memory_utilization: float = 0.60
    coach_vllm_max_model_len: int = 8192  # Coach: 1024 gen + 4096 solve
    vllm_enable_sleep: bool = True
    max_model_len: int = 12288
    enable_prefix_caching: bool = False
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = []


@dataclass
class TrainingConfig:
    """Training run configuration."""
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    coach_model_name: Optional[str] = "Qwen/Qwen3-14B"
    num_iterations: int = 3000
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 10
    log_freq: int = 1
    eval_freq: int = 0  # 0 = disabled
    eval_n_samples: int = 8
    eval_dataset: str = "dapo"
    eval_n_problems: int = 100
    attn_implementation: str = "eager"
    ref_reward_offload: bool = False
    start_iteration: int = 0  # set >0 to resume from a given iteration


@dataclass
class CRISPConfig:
    """Top-level configuration."""
    player: PlayerConfig = None
    coach: CoachConfig = None
    advantage: AdvantageConfig = None
    grpo: GRPOConfig = None
    infra: InfraConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.player is None:
            self.player = PlayerConfig()
        if self.coach is None:
            self.coach = CoachConfig()
        if self.advantage is None:
            self.advantage = AdvantageConfig()
        if self.grpo is None:
            self.grpo = GRPOConfig()
        if self.infra is None:
            self.infra = InfraConfig()
        if self.training is None:
            self.training = TrainingConfig()
