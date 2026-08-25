"""Unit tests for the session state machine (`app.domain.state`).

These pin the transition rules that PR3 will use to replace the
`loop_count = 999` sentinel. For PR1 they just guard the seam.
"""

from __future__ import annotations

import pytest

from app.domain.state import SessionStatus, SessionTerminalError, can_reply, is_terminal, transition


class TestSessionStatus:
    def test_active_is_not_terminal(self) -> None:
        assert not is_terminal(SessionStatus.active)
        assert can_reply(SessionStatus.active)

    def test_revealed_is_terminal(self) -> None:
        assert is_terminal(SessionStatus.revealed)
        assert not can_reply(SessionStatus.revealed)

    def test_solved_is_terminal(self) -> None:
        assert is_terminal(SessionStatus.solved)
        assert not can_reply(SessionStatus.solved)

    def test_str_values(self) -> None:
        assert SessionStatus.active.value == "active"
        assert SessionStatus.revealed.value == "revealed"
        assert SessionStatus.solved.value == "solved"


class TestTransition:
    def test_active_to_active_ok(self) -> None:
        assert transition(SessionStatus.active, SessionStatus.active) is SessionStatus.active

    def test_active_to_revealed_ok(self) -> None:
        assert transition(SessionStatus.active, SessionStatus.revealed) is SessionStatus.revealed

    def test_active_to_solved_ok(self) -> None:
        assert transition(SessionStatus.active, SessionStatus.solved) is SessionStatus.solved

    def test_revealed_cannot_transition(self) -> None:
        with pytest.raises(SessionTerminalError, match="terminal"):
            transition(SessionStatus.revealed, SessionStatus.active)

    def test_solved_cannot_transition(self) -> None:
        with pytest.raises(SessionTerminalError, match="terminal"):
            transition(SessionStatus.solved, SessionStatus.revealed)

    def test_revealed_to_solved_blocked(self) -> None:
        # Even terminal -> terminal is blocked; a session resolves exactly once.
        with pytest.raises(SessionTerminalError, match="terminal"):
            transition(SessionStatus.revealed, SessionStatus.solved)
