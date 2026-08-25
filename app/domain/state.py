"""Session state machine.

Replaces the magic `loop_count = 999` / `loop_index = 999` sentinels used in
`app/services/session.py` today. A `SessionStatus` column will be added to the
sessions table in PR3; for PR1 the enum + transition guard exist so the
repository Protocol can be typed against them and so the contract tests have
a target.

Transition rules:
  active    -> active | revealed | solved
  revealed  -> (terminal)
  solved    -> (terminal)

Once terminal, `can_reply` returns False and the service layer should reject
new turns with 409 (currently 409 is emitted but detected via `resolved` bool,
which is fragile — see PR3).
"""

from __future__ import annotations

from enum import StrEnum


class SessionTerminalError(Exception):
    """Raised when an operation is attempted on a terminal (resolved) session."""


class SessionStatus(StrEnum):
    active = "active"
    revealed = "revealed"
    solved = "solved"


_TERMINAL = {SessionStatus.revealed, SessionStatus.solved}


def is_terminal(status: SessionStatus) -> bool:
    return status in _TERMINAL


def can_reply(status: SessionStatus) -> bool:
    return not is_terminal(status)


def transition(from_: SessionStatus, to: SessionStatus) -> SessionStatus:
    """Validate a status transition and return the new status.

    Raises `SessionTerminalError` on illegal transitions (i.e. any transition out of
    a terminal state). The active -> active case is allowed (every hint turn
    keeps the session active).
    """
    if is_terminal(from_):
        raise SessionTerminalError(f"session is terminal ({from_}); cannot transition to {to}")
    if to not in (SessionStatus.active, SessionStatus.revealed, SessionStatus.solved):
        raise ValueError(f"unknown target status: {to}")
    return to
