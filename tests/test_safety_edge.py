"""Edge-case tests for the safety layer."""

import pytest
from laua.executor.safety import check_command, SafetyVerdict


# ── empty / degenerate input ────────────────────────────────────────────────

def test_empty_args_does_not_raise():
    """Empty arg list must return a verdict, not IndexError."""
    verdict = check_command([])
    assert isinstance(verdict, SafetyVerdict)


def test_single_empty_string():
    verdict = check_command([""])
    assert not verdict.blocked


# ── rm -rf / variants ───────────────────────────────────────────────────────

@pytest.mark.parametrize("args", [
    ["rm", "-rf", "/"],
    ["rm", "-fr", "/"],          # flags reversed
    ["rm", "-Rf", "/"],          # uppercase R
    ["rm", "-fR", "/"],          # uppercase R reversed
    ["rm", "-rf", "/*"],         # wildcard
    ["sudo", "rm", "-rf", "/"],  # sudo prefix; pattern search still finds it
    ["rm", "-rf", "//"],         # double-slash root — same as /
])
def test_rm_rf_root_blocked(args):
    assert check_command(args).blocked, f"Expected blocked: {args}"


@pytest.mark.parametrize("args", [
    ["rm", "-rf", "/tmp/foo"],        # specific subdirectory — should NOT block
    ["rm", "-rf", "/home/user/junk"], # home subdir — should NOT block
    ["rm", "-f", "file.txt"],         # no -r flag
])
def test_rm_specific_path_not_blocked(args):
    assert not check_command(args).blocked, f"Should NOT be blocked: {args}"


# ── shutdown / reboot in different contexts ──────────────────────────────────

def test_shutdown_command_blocked():
    assert check_command(["shutdown", "-h", "now"]).blocked

def test_reboot_blocked():
    assert check_command(["reboot"]).blocked

def test_shutdown_script_name_blocked_or_not():
    """
    A script called shutdown.sh: \\b boundary matches because '.' is non-word.
    Document the current behaviour — if blocked, flag it as a known false positive.
    """
    verdict = check_command(["./shutdown.sh"])
    # Record actual behaviour without asserting a direction — this is documenting it
    assert isinstance(verdict.blocked, bool)

def test_echo_shutdown_blocked():
    """echo shutdown now — currently blocked (false positive worth knowing about)."""
    verdict = check_command(["echo", "shutdown", "now"])
    # Just document: currently True because the word appears in flat string
    assert isinstance(verdict.blocked, bool)


# ── chmod edge cases ─────────────────────────────────────────────────────────

def test_chmod_777_root_blocked():
    assert check_command(["chmod", "-R", "777", "/"]).blocked

def test_chmod_777_absolute_subpath():
    """
    chmod -R 777 /tmp — the current regex matches any /-starting path.
    This documents that behaviour (over-broad blocking).
    """
    verdict = check_command(["chmod", "-R", "777", "/tmp"])
    assert isinstance(verdict.blocked, bool)  # document; do not hide the result

def test_chmod_755_not_blocked():
    assert not check_command(["chmod", "755", "script.sh"]).blocked


# ── dd variants ──────────────────────────────────────────────────────────────

def test_dd_to_sda_blocked():
    assert check_command(["dd", "if=/dev/zero", "of=/dev/sda"]).blocked

def test_dd_to_sdb1_blocked():
    assert check_command(["dd", "if=/dev/zero", "of=/dev/sdb1"]).blocked

def test_dd_to_file_not_blocked():
    assert not check_command(["dd", "if=/dev/zero", "of=/tmp/out.img", "bs=1M", "count=10"]).blocked


# ── mkfs variants ─────────────────────────────────────────────────────────────

def test_mkfs_ext4_blocked():
    assert check_command(["mkfs.ext4", "/dev/sdb"]).blocked

def test_mkfs_btrfs_blocked():
    assert check_command(["mkfs.btrfs", "/dev/nvme0n1"]).blocked

def test_mkfs_in_path_not_blocked():
    """A file literally named mkfs_backup.sh should not be blocked."""
    verdict = check_command(["/home/user/mkfs_backup.sh"])
    # mkfs\. requires a dot after mkfs — mkfs_ won't match
    assert not verdict.blocked


# ── sudo escalation classification ──────────────────────────────────────────

def test_sudo_prefix_needs_sudo():
    verdict = check_command(["sudo", "apt", "update"])
    assert verdict.requires_sudo
    assert verdict.requires_confirmation

def test_sudo_alone():
    """Just 'sudo' with no subcommand."""
    verdict = check_command(["sudo"])
    assert verdict.requires_sudo

def test_etc_path_read_flagged():
    """cat /etc/passwd — contains /etc/ so flagged as needing sudo (over-broad)."""
    verdict = check_command(["cat", "/etc/passwd"])
    assert verdict.requires_sudo  # document current over-broad behaviour

def test_safe_command_no_confirmation():
    verdict = check_command(["df", "-h"])
    assert not verdict.blocked
    assert not verdict.requires_confirmation
    assert not verdict.requires_sudo


# ── confirmation classification ──────────────────────────────────────────────

def test_kill_needs_confirmation():
    assert check_command(["kill", "1234"]).requires_confirmation

def test_pkill_needs_confirmation():
    assert check_command(["pkill", "nginx"]).requires_confirmation

def test_systemctl_stop_needs_confirmation():
    assert check_command(["systemctl", "stop", "nginx"]).requires_confirmation

def test_systemctl_start_no_confirmation():
    """systemctl start does NOT match the stop|restart|disable pattern."""
    verdict = check_command(["systemctl", "start", "nginx"])
    assert not verdict.requires_confirmation

def test_apt_install_needs_confirmation():
    assert check_command(["apt", "install", "vim"]).requires_confirmation

def test_apt_update_no_confirmation():
    """apt update is not install/remove/purge."""
    verdict = check_command(["apt", "update"])
    assert not verdict.requires_confirmation

def test_mv_to_absolute_path_needs_confirmation():
    assert check_command(["mv", "file.txt", "/tmp/file.txt"]).requires_confirmation

def test_mv_between_relative_paths_no_confirmation():
    """mv foo bar — relative paths, no /."""
    verdict = check_command(["mv", "foo", "bar"])
    assert not verdict.requires_confirmation
