"""Edge-case tests for the enhanced system monitor."""

from __future__ import annotations

import pytest

from laua.monitor.system import (
    SystemSnapshot,
    SystemMonitor,
    detect_anomalies,
    _gpu_info_blocking,
    _thermal_info_blocking,
)


def _snap(
    cpu: float = 10.0,
    mem: float = 20.0,
    disk: float = 30.0,
    gpu=None,
    thermals=None,
) -> SystemSnapshot:
    return SystemSnapshot(
        cpu_percent=cpu,
        memory_percent=mem,
        memory_available_gb=8.0,
        disk_percent=disk,
        disk_free_gb=100.0,
        top_processes=[],
        gpu=gpu,
        thermals=thermals,
    )


# ── detect_anomalies: boundary and degenerate ─────────────────────────────────

def test_all_zeros_no_anomalies():
    assert detect_anomalies(_snap(cpu=0, mem=0, disk=0)) == []


def test_threshold_minus_epsilon_no_warning():
    warnings = detect_anomalies(_snap(cpu=89.9), cpu_threshold=90.0)
    assert not any("CPU" in w for w in warnings)


def test_custom_low_threshold():
    warnings = detect_anomalies(_snap(cpu=50), cpu_threshold=40.0)
    assert any("CPU" in w for w in warnings)


def test_100_percent_triggers():
    warnings = detect_anomalies(_snap(cpu=100.0, mem=100.0, disk=100.0))
    assert len(warnings) >= 3


# ── GPU: edge cases ───────────────────────────────────────────────────────────

def test_gpu_none_no_gpu_warning():
    """snapshot.gpu=None should produce no GPU warnings."""
    warnings = detect_anomalies(_snap(gpu=None))
    assert not any("GPU" in w for w in warnings)


def test_gpu_empty_gpus_list_no_warning():
    """{'gpus': []} — iterable but empty — no warnings."""
    warnings = detect_anomalies(_snap(gpu={"gpus": []}))
    assert not any("GPU" in w for w in warnings)


def test_gpu_missing_utilization_key_no_crash():
    """GPU entry missing 'utilization_percent' should not crash."""
    gpu = {"gpus": [{"index": 0, "memory_used_mb": 1000, "memory_total_mb": 8192}]}
    warnings = detect_anomalies(_snap(gpu=gpu), cpu_threshold=90.0)
    assert not any("GPU" in w for w in warnings)


def test_gpu_utilization_zero_no_warning():
    gpu = {"gpus": [{"index": 0, "utilization_percent": 0, "memory_used_mb": 0, "memory_total_mb": 8192}]}
    warnings = detect_anomalies(_snap(gpu=gpu))
    assert not any("GPU" in w for w in warnings)


def test_multiple_gpus_one_high_one_low():
    gpu = {"gpus": [
        {"index": 0, "utilization_percent": 95},
        {"index": 1, "utilization_percent": 10},
    ]}
    warnings = detect_anomalies(_snap(gpu=gpu), cpu_threshold=90.0)
    gpu_warnings = [w for w in warnings if "GPU" in w]
    assert len(gpu_warnings) == 1
    assert "0" in gpu_warnings[0]


# ── thermals: not checked by detect_anomalies ────────────────────────────────

def test_thermals_do_not_produce_warnings():
    """Current detect_anomalies does not check thermal data — document this."""
    thermals = {"coretemp": [{"label": "core 0", "current": 95, "high": 80, "critical": 100}]}
    warnings = detect_anomalies(_snap(thermals=thermals))
    thermal_warnings = [w for w in warnings if "temp" in w.lower() or "thermal" in w.lower()]
    assert thermal_warnings == [], "Thermal anomaly detection is not yet implemented"


# ── _gpu_info_blocking / _thermal_info_blocking ───────────────────────────────

def test_gpu_info_never_raises():
    result = _gpu_info_blocking()
    assert result is None or isinstance(result, dict)


def test_thermal_info_never_raises():
    result = _thermal_info_blocking()
    assert result is None or isinstance(result, dict)


def test_thermal_info_structure_when_available():
    result = _thermal_info_blocking()
    if result is not None:
        assert isinstance(result, dict)
        for sensor_name, entries in result.items():
            assert isinstance(entries, list)
            for entry in entries:
                assert "current" in entry


# ── SystemMonitor: async snapshot ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_snapshot_gpu_disabled_returns_none():
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=False)
    snap = await monitor.snapshot()
    assert snap.gpu is None
    assert snap.thermals is None


@pytest.mark.asyncio
async def test_snapshot_gpu_enabled_no_crash():
    """gpu_enabled=True with no GPU available should return None, not crash."""
    monitor = SystemMonitor(gpu_enabled=True, thermal_enabled=False)
    snap = await monitor.snapshot()
    # gpu may be None (no GPU) or a dict — both are fine
    assert snap.gpu is None or isinstance(snap.gpu, dict)


@pytest.mark.asyncio
async def test_snapshot_thermal_enabled_no_crash():
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=True)
    snap = await monitor.snapshot()
    assert snap.thermals is None or isinstance(snap.thermals, dict)


@pytest.mark.asyncio
async def test_snapshot_cpu_in_range():
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=False)
    snap = await monitor.snapshot()
    assert 0.0 <= snap.cpu_percent <= 100.0


@pytest.mark.asyncio
async def test_snapshot_memory_in_range():
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=False)
    snap = await monitor.snapshot()
    assert 0.0 <= snap.memory_percent <= 100.0
    assert snap.memory_available_gb >= 0


@pytest.mark.asyncio
async def test_snapshot_top_processes_list():
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=False)
    snap = await monitor.snapshot()
    assert isinstance(snap.top_processes, list)
    assert len(snap.top_processes) <= 10


@pytest.mark.asyncio
async def test_refresh_excluded_pids_includes_own():
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=False)
    await monitor.refresh_excluded_pids()
    assert monitor._own_pid in monitor._excluded_pids


@pytest.mark.asyncio
async def test_own_pid_not_in_top_processes():
    """Our own PID must never appear in the top_processes list."""
    import os
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=False)
    await monitor.refresh_excluded_pids()
    snap = await monitor.snapshot()
    pids = [p["pid"] for p in snap.top_processes]
    assert os.getpid() not in pids
