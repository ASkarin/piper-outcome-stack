import torch

from piper_policy.projected_update import step_with_reference_projection


def test_conflicting_step_removes_only_reference_ascent_component():
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0)
    result = step_with_reference_projection(opt, -p[0] + p[1], p[0].square() / 2)
    assert result["projected"]
    assert result["proposed_reference_dots"][0] > 0
    assert abs(result["applied_reference_dots"][0]) < 1e-14
    assert p[0] == 1 and p[1] < 1
    assert opt.state[p]["step"] == 1


def test_aligned_and_missing_reference_preserve_original_adamw_step():
    for reference in [True, False]:
        p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        q = torch.nn.Parameter(p.detach().clone())
        opt = torch.optim.AdamW([p], lr=0.01, weight_decay=0.01)
        control = torch.optim.AdamW([q], lr=0.01, weight_decay=0.01)
        result = step_with_reference_projection(opt, p.square().sum(), p[0] if reference else None)
        q.square().sum().backward()
        control.step()
        assert not result["projected"]
        torch.testing.assert_close(p, q, atol=0, rtol=0)
        for key in opt.state[p]:
            torch.testing.assert_close(opt.state[p][key], control.state[q][key], atol=0, rtol=0)


def test_projection_checks_momentum_displacement_not_raw_gradient_sign():
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=0.01, weight_decay=0)
    (-p.sum()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    before = p.detach().clone()
    result = step_with_reference_projection(opt, 0.001 * p.sum(), p.square().sum() / 2)
    # Positive total gradient alone would suggest descent, but Adam momentum proposes ascent.
    assert result["projected"] and result["proposed_reference_dots"][0] > 0
    torch.testing.assert_close(p, before, atol=1e-14, rtol=0)
    assert opt.state[p]["step"] == 2


def test_two_conflicting_halfspaces_are_satisfied_together():
    from piper_policy.projected_update import project_displacement

    delta = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float64)
    gradients = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0]], dtype=torch.float64)
    actual, projected = project_displacement(delta, gradients)
    assert projected
    torch.testing.assert_close(
        actual, torch.tensor([0.0, 0.0, -3.0], dtype=torch.float64), atol=1e-12, rtol=0
    )
    assert torch.max(gradients @ actual) < 1e-12


def test_redundant_opposite_and_zero_references_have_feasible_solutions():
    from piper_policy.projected_update import project_displacement

    delta = torch.tensor([1.0, 2.0], dtype=torch.float64)
    for rows in [[[1.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]:
        g = torch.tensor(rows, dtype=torch.float64)
        actual, projected = project_displacement(delta, g)
        assert projected
        torch.testing.assert_close(
            actual, torch.tensor([0.0, 2.0], dtype=torch.float64), atol=1e-12, rtol=0
        )


def test_two_losses_change_no_optimizer_moments_outside_normal_adam_step():
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.01)
    control = torch.optim.AdamW([q], lr=0.1, weight_decay=0.01)
    result = step_with_reference_projection(opt, -p[0] - p[1] + p[2], [p[0], -p[0] + p[1]])
    (-q[0] - q[1] + q[2]).backward()
    control.step()
    assert result["projected"]
    assert max(result["applied_reference_dots"]) < 1e-12
    for key in opt.state[p]:
        torch.testing.assert_close(opt.state[p][key], control.state[q][key], atol=0, rtol=0)
