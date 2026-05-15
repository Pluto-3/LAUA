"""Tests for enhanced SystemMonitor — anomaly detection, GPU/thermal graceful None."""

import pytest
from laua.monitor.system import detect_anomalies, SystemSnapshot, _thermal_info_blocking, _gpu_info_blocking


def _snap(cpu=10.0, mem=20.0, disk=30.0, gpu=None, thermals=None):
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


# ── anomaly detection ─────────────────────────────────────────────────────────

def test_no_anomalies_when_low():
    warnings = detect_anomalies(_snap(cpu=20, mem=30, disk=40))
    assert warnings == []


def test_cpu_anomaly():
    warnings = detect_anomalies(_snap(cpu=95), cpu_threshold=90)
    assert any("CPU" in w for w in warnings)


def test_memory_anomaly():
    warnings = detect_anomalies(_snap(mem=92), memory_threshold=90)
    assert any("Memory" in w or "memory" in w.lower() for w in warnings)


def test_disk_anomaly():
    warnings = detect_anomalies(_snap(disk=95), disk_threshold=90)
    assert any("Disk" in w or "disk" in w.lower() for w in warnings)


def test_multiple_anomalies():
    warnings = detect_anomalies(_snap(cpu=95, mem=95, disk=95))
    assert len(warnings) >= 3


def test_exact_threshold_triggers():
    warnings = detect_anomalies(_snap(cpu=90.0), cpu_threshold=90.0)
    assert any("CPU" in w for w in warnings)


def test_just_below_threshold_no_warning():
    warnings = detect_anomalies(_snap(cpu=89.9), cpu_threshold=90.0)
    assert not any("CPU" in w for w in warnings)


def test_gpu_anomaly_detected():
    gpu = {"gpus": [{"index": 0, "utilization_percent": 95, "memory_used_mb": 8000, "memory_total_mb": 8192}]}
    warnings = detect_anomalies(_snap(gpu=gpu), cpu_threshold=90)
    assert any("GPU" in w for w in warnings)


def test_gpu_normal_no_warning():
    gpu = {"gpus": [{"index": 0, "utilization_percent": 30, "memory_used_mb": 1000, "memory_total_mb": 8192}]}
    warnings = detect_anomalies(_snap(gpu=gpu))
    assert not any("GPU" in w for w in warnings)


# ── GPU info returns None when unavailable ────────────────────────────────────

def test_gpu_info_returns_none_or_dict():
    """gpu_info_blocking returns None (no GPU) or a dict — never raises."""
    result = _gpu_info_blocking()
    assert result is None or isinstance(result, dict)


# ── thermal info returns None when unavailable ────────────────────────────────

def test_thermal_info_returns_none_or_dict():
    """thermal_info_blocking returns None or a dict — never raises."""
    result = _thermal_info_blocking()
    assert result is None or isinstance(result, dict)


# ── SystemMonitor async snapshot ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_snapshot_returns_snapshot():
    from laua.monitor.system import SystemMonitor
    monitor = SystemMonitor(gpu_enabled=False, thermal_enabled=False)
    snap = await monitor.snapshot()
    assert isinstance(snap, SystemSnapshot)
    assert 0.0 <= snap.cpu_percent <= 100.0
    assert snap.gpu is None
    assert snap.thermals is None
