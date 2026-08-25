"""Tests for OCR validation and retry logic."""

from __future__ import annotations

from app.services.ocr import _validate_ocr_text


class TestValidateOcrText:
    def test_accepts_normal_physics_problem(self) -> None:
        text = (
            "A ball of mass 100 g hits the ground at an angle of 45 degrees "
            "and a speed of 10 m/s. What is the average net force?"
        )
        assert _validate_ocr_text(text) is True

    def test_accepts_problem_with_units(self) -> None:
        assert _validate_ocr_text("A 2 kg cart moves at 3 m/s. Find the impulse.") is True

    def test_accepts_question_mark(self) -> None:
        assert _validate_ocr_text("What is the magnitude of the net force on the ball?") is True

    def test_rejects_empty_string(self) -> None:
        assert _validate_ocr_text("") is False

    def test_rejects_whitespace_only(self) -> None:
        assert _validate_ocr_text("   \n  \t  ") is False

    def test_rejects_very_short_text(self) -> None:
        assert _validate_ocr_text("Hello world.") is False

    def test_rejects_text_without_physics_indicators(self) -> None:
        assert _validate_ocr_text(
            "The quick brown fox jumps over the lazy dog and runs away."
        ) is False

    def test_accepts_calculate_keyword(self) -> None:
        assert _validate_ocr_text(
            "Calculate the total energy of the system described above."
        ) is True

    def test_accepts_determine_keyword(self) -> None:
        assert _validate_ocr_text(
            "Determine the acceleration of the block down the incline."
        ) is True


class TestExtractProblemRetry:
    async def test_retries_on_invalid_ocr(self, monkeypatch) -> None:
        """When the first OCR call returns garbage, extract_problem retries."""
        from app.services import ocr

        call_count = 0

        async def fake_ocr_image(image_bytes, *, prompt, mime="image/png", max_tokens=2000):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Hi"  # too short — triggers retry
            return "A ball of mass 100 g hits the ground at 10 m/s. Find the force?"

        async def fake_chat_json(system, user, *, temperature=0.4, max_tokens=1200):
            return {
                "problem_text": "A ball of mass 100 g hits the ground at 10 m/s.",
                "formulas": [],
                "concepts": ["momentum"],
                "topic": None,
                "diagram_description": None,
            }

        monkeypatch.setattr(ocr.llm, "ocr_image", fake_ocr_image)
        monkeypatch.setattr(ocr.llm, "chat_json", fake_chat_json)

        result = await ocr.extract_problem(b"fake-image", "image/png")
        assert call_count == 2, "should have retried once"
        assert "ball" in result.problem_text

    async def test_no_retry_on_valid_ocr(self, monkeypatch) -> None:
        """When the first OCR call returns valid text, no retry happens."""
        from app.services import ocr

        call_count = 0

        async def fake_ocr_image(image_bytes, *, prompt, mime="image/png", max_tokens=2000):
            nonlocal call_count
            call_count += 1
            return "A 2 kg cart moves at 3 m/s. Find the impulse?"

        async def fake_chat_json(system, user, *, temperature=0.4, max_tokens=1200):
            return {
                "problem_text": "A 2 kg cart moves at 3 m/s. Find the impulse?",
                "formulas": [],
                "concepts": ["impulse"],
                "topic": None,
                "diagram_description": None,
            }

        monkeypatch.setattr(ocr.llm, "ocr_image", fake_ocr_image)
        monkeypatch.setattr(ocr.llm, "chat_json", fake_chat_json)

        result = await ocr.extract_problem(b"fake-image", "image/png")
        assert call_count == 1, "should not retry on valid OCR"
        assert "cart" in result.problem_text

    async def test_proceeds_when_both_calls_fail(self, monkeypatch) -> None:
        """When both OCR calls fail validation, we proceed with the best output."""
        from app.services import ocr

        call_count = 0

        async def fake_ocr_image(image_bytes, *, prompt, mime="image/png", max_tokens=2000):
            nonlocal call_count
            call_count += 1
            return "x"  # both calls return garbage

        async def fake_chat_json(system, user, *, temperature=0.4, max_tokens=1200):
            return {
                "problem_text": "x",
                "formulas": [],
                "concepts": [],
                "topic": None,
                "diagram_description": None,
            }

        monkeypatch.setattr(ocr.llm, "ocr_image", fake_ocr_image)
        monkeypatch.setattr(ocr.llm, "chat_json", fake_chat_json)

        result = await ocr.extract_problem(b"fake-image", "image/png")
        assert call_count == 2, "should have retried once even though both failed"
        assert result.problem_text == "x"

    async def test_unwrap_ocr_json_envelope(self) -> None:
        """_unwrap_ocr extracts natural_text from the JSON envelope."""
        from app.services.ocr import _unwrap_ocr

        raw = '{"natural_text": "A ball of mass 100 g...", "primary_language": "en"}'
        assert _unwrap_ocr(raw) == "A ball of mass 100 g..."

    async def test_unwrap_ocr_plain_text(self) -> None:
        """_unwrap_ocr passes through plain text without an envelope."""
        from app.services.ocr import _unwrap_ocr

        raw = "A ball of mass 100 g hits the ground."
        assert _unwrap_ocr(raw) == raw
