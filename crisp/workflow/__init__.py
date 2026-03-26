"""CRISP workflow orchestration layer.

Wires crisp/infra/ (Ray/vLLM/DeepSpeed) to domain logic
(rewards, discussion, verification, training).
"""
from .context import StepResult, WorkflowContext
from .main_loop import step
