"""Tests for crisp.workflow public API."""


def test_workflow_public_api():
    """crisp.workflow exposes step, WorkflowContext, and StepResult."""
    import crisp.workflow as workflow

    assert hasattr(workflow, "step")
    assert hasattr(workflow, "WorkflowContext")
    assert hasattr(workflow, "StepResult")
