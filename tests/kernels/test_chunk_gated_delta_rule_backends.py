# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GDN prefill backend dispatch and FlashQLA wrapper.

CPU-only tests cover the gate function, the CLI flag, and the state-layout
transpose used to shim between vLLM/FLA's ``[N, H, V, K]`` and FlashQLA's
``[N, H, K, V]``. The end-to-end backend equivalence test requires CUDA
and skips when the relevant backend cannot run on the host.
"""

from __future__ import annotations

import pytest
import torch

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba import gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    _should_use_flashqla_gdn_prefill,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser


# --------------------------- CLI flag (CPU-only) ---------------------------


@pytest.mark.parametrize("backend", ["flashinfer", "triton", "flashqla"])
def test_gdn_prefill_backend_cli_accepts_each_choice(backend: str) -> None:
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model", "dummy-model", "--gdn-prefill-backend", backend]
    )
    assert args.gdn_prefill_backend == backend


def test_gdn_prefill_backend_cli_rejects_unknown() -> None:
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "dummy-model", "--gdn-prefill-backend", "bogus"]
        )


# ------------------------ Gate function (CPU-only) -------------------------


class _FakePlatform:
    """Minimal stand-in for ``current_platform`` so we can drive the gate."""

    def __init__(
        self,
        is_cuda: bool = True,
        device_capability: int = 90,
        family: int = 90,
        cuda_runtime_major: int = 13,
    ) -> None:
        self._is_cuda = is_cuda
        self._dc = device_capability
        self._family = family
        self._cuda_runtime_major = cuda_runtime_major

    def is_cuda(self) -> bool:
        return self._is_cuda

    def is_device_capability(self, cap: int) -> bool:
        return self._dc == cap

    def is_device_capability_family(self, family: int) -> bool:
        return self._family == family

    def get_cuda_runtime_major(self) -> int:
        return self._cuda_runtime_major


@pytest.fixture
def hopper_with_pkg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the host is Hopper (SM90) and flash_qla is installed."""
    monkeypatch.setattr(gdn_linear_attn, "current_platform", _FakePlatform())
    import vllm.utils.import_utils as iu

    monkeypatch.setattr(iu, "has_flash_qla", lambda: True)


def test_flashqla_gate_is_opt_in_only(hopper_with_pkg: None) -> None:
    # Auto / flashinfer / triton must NOT select FlashQLA even on Hopper.
    assert not _should_use_flashqla_gdn_prefill("auto", 128, 128)
    assert not _should_use_flashqla_gdn_prefill("flashinfer", 128, 128)
    assert not _should_use_flashqla_gdn_prefill("triton", 128, 128)


def test_flashqla_gate_accepts_flashqla_on_hopper(hopper_with_pkg: None) -> None:
    assert _should_use_flashqla_gdn_prefill("flashqla", 128, 128)


def test_flashqla_gate_rejects_non_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gdn_linear_attn, "current_platform", _FakePlatform(is_cuda=False)
    )
    import vllm.utils.import_utils as iu

    monkeypatch.setattr(iu, "has_flash_qla", lambda: True)
    assert not _should_use_flashqla_gdn_prefill("flashqla", 128, 128)


def test_flashqla_gate_rejects_non_sm90(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ada (SM89) and Blackwell (SM100) both rejected.
    for cap in (86, 89, 100, 120):
        monkeypatch.setattr(
            gdn_linear_attn,
            "current_platform",
            _FakePlatform(device_capability=cap, family=cap),
        )
        import vllm.utils.import_utils as iu

        monkeypatch.setattr(iu, "has_flash_qla", lambda: True)
        assert not _should_use_flashqla_gdn_prefill("flashqla", 128, 128), cap


@pytest.mark.parametrize(
    ("head_k", "head_v"),
    [(64, 128), (128, 64), (256, 128), (128, None), (None, 128), (None, None)],
)
def test_flashqla_gate_rejects_wrong_head_dims(
    hopper_with_pkg: None, head_k: int | None, head_v: int | None
) -> None:
    assert not _should_use_flashqla_gdn_prefill("flashqla", head_k, head_v)


def test_flashqla_gate_rejects_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdn_linear_attn, "current_platform", _FakePlatform())
    import vllm.utils.import_utils as iu

    monkeypatch.setattr(iu, "has_flash_qla", lambda: False)
    assert not _should_use_flashqla_gdn_prefill("flashqla", 128, 128)


# --------------------- State-layout transpose (CPU-only) ---------------------


def test_state_layout_transpose_is_inverse_for_symmetric_dims() -> None:
    """For K == V (Qwen3-Next default 128/128), transpose is its own inverse."""
    N, H, K, V = 2, 8, 128, 128
    state = torch.randn(N, H, V, K)
    roundtrip = state.transpose(-2, -1).contiguous().transpose(-2, -1).contiguous()
    torch.testing.assert_close(roundtrip, state)


def test_state_layout_transpose_changes_element_order() -> None:
    """Even when K == V, the transpose changes element order — element [a, b]
    becomes element [b, a] — so it is NOT a no-op despite the matching shape."""
    state = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    transposed = state.transpose(-2, -1).contiguous()
    assert transposed.shape == state.shape
    # If the transpose were a no-op the data would match — it should not.
    assert not torch.equal(transposed, state)
    # And the roundtrip recovers the original.
    torch.testing.assert_close(
        transposed.transpose(-2, -1).contiguous(), state
    )


def test_state_layout_transpose_handles_asymmetric_dims() -> None:
    """K != V case sanity-check (covers future models with non-square state)."""
    N, H, K, V = 2, 4, 64, 128
    state = torch.randn(N, H, V, K)
    transposed = state.transpose(-2, -1).contiguous()
    assert transposed.shape == (N, H, K, V)
    torch.testing.assert_close(
        transposed.transpose(-2, -1).contiguous(), state
    )


# ----------------- End-to-end backend equivalence (GPU-only) -----------------


def _device_capability_int() -> int:
    """Return ``major * 10 + minor`` for the current CUDA device."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["flashinfer", "flashqla"])
def test_chunk_gated_delta_rule_backend_matches_fla(backend: str) -> None:
    """All non-FLA backends should produce ``(o, final_state)`` close to the
    FLA reference within FlashQLA's own 2 % tolerance band (5 % here for
    bf16 fp accumulation slack)."""
    cap = _device_capability_int()
    if backend == "flashqla":
        pytest.importorskip("flash_qla")
        if cap != 90:
            pytest.skip("FlashQLA hard-asserts sm_90 at import")
    elif backend == "flashinfer":
        pytest.importorskip("flashinfer.gdn_prefill")
        if cap < 90:
            pytest.skip("FlashInfer GDN prefill requires SM90+")

    from vllm.model_executor.layers.fla.ops import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    )
    from vllm.model_executor.layers.mamba.gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
        fqla_chunk_gated_delta_rule,
    )

    torch.manual_seed(0)
    B, T, H_qk, H_v, D = 1, 64, 4, 8, 128
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(B, T, H_qk, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H_qk, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H_v, D, device=device, dtype=dtype)
    g = torch.randn(B, T, H_v, device=device, dtype=torch.float32) * -0.1
    beta = torch.sigmoid(
        torch.randn(B, T, H_v, device=device, dtype=torch.float32)
    )
    h0 = torch.zeros(1, H_v, D, D, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

    o_ref, h_ref = fla_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    if backend == "flashinfer":
        o, h = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )
    else:  # flashqla
        o, h = fqla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
        )

    torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        h.to(torch.float32), h_ref.to(torch.float32), atol=5e-2, rtol=5e-2
    )
