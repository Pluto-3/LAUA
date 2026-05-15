"""Tests for the command safety layer."""

import pytest
from laua.executor.safety import check_command


@pytest.mark.parametrize("args,should_block", [
    (["rm", "-rf", "/"], True),
    (["rm", "-rf", "/*"], True),
    (["mkfs.ext4", "/dev/sdb"], True),
    (["dd", "if=/dev/zero", "of=/dev/sda"], True),
    (["shutdown", "-h", "now"], True),
    (["chmod", "-R", "777", "/"], True),
    (["ls", "-la"], False),
    (["df", "-h"], False),
    (["cat", "/etc/os-release"], False),
    (["git", "status"], False),
])
def test_blocked_patterns(args, should_block):
    verdict = check_command(args)
    assert verdict.blocked == should_block, f"Expected blocked={should_block} for {args}"


@pytest.mark.parametrize("args,should_confirm", [
    (["rm", "somefile.txt"], True),
    (["kill", "1234"], True),
    (["sudo", "apt", "install", "vim"], True),
    (["ls", "-la"], False),
    (["cat", "file.txt"], False),
])
def test_confirmation_required(args, should_confirm):
    verdict = check_command(args)
    if not verdict.blocked:
        assert verdict.requires_confirmation == should_confirm, (
            f"Expected requires_confirmation={should_confirm} for {args}"
        )
