import json
from datetime import datetime
from pathlib import Path

import pytest

from src.spanish_encoding_fixer import (
    SpanishEncodingFixer,
    ProcessingStatus,
    ProcessingStats,
    ProcessingResult,
    EncodingIssue
)


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for file operations."""
    return tmp_path


@pytest.fixture
def input_json(temp_dir):
    """Creates a sample JSON file with encoding issues."""
    content = {
        "name": "JosÃ©",
        "city": "MÃ¡laga",
        "description": "Text with Ã± and Ã¡ characters",
        "nested": {
            "text": "More Ã© problems"
        },
        "list": ["ItemÃ³", "ItemÃº"]
    }

    file_path = temp_dir / "test.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(content, f)
    return file_path


@pytest.fixture
def fixer(temp_dir):
    """Creates a SpanishEncodingFixer instance."""
    return SpanishEncodingFixer(
        input_dir=temp_dir,
        output_dir=temp_dir / "output",
        backup=True
    )


class TestSpanishEncodingFixer:
    def test_initialization(self, temp_dir):
        """Test proper initialization of SpanishEncodingFixer."""
        fixer = SpanishEncodingFixer(input_dir=temp_dir)
        assert fixer.input_dir == Path(temp_dir)
        assert fixer.backup is True
        assert fixer.max_retries == 3
        assert hasattr(fixer, 'logger')
        assert fixer.replacements

    def test_invalid_input_dir(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            SpanishEncodingFixer(input_dir="nonexistent_dir")

    def test_fix_text(self, fixer):
        """Test text fixing functionality."""
        test_cases = [
            ("JosÃ©", "José"),
            ("MÃ¡laga", "Málaga"),
            ("Normal text", "Normal text"),
            ("Multiple Ã¡ Ã© Ã\u00AD", "Multiple á é í"),
        ]

        for input_text, expected in test_cases:
            assert fixer.fix_text(input_text) == expected

    def test_fix_json_content(self, fixer):
        """Test JSON content fixing functionality."""
        input_data = {
            "string": "JosÃ©",
            "number": 42,
            "nested": {"text": "MÃ¡laga"},
            "list": ["ItemÃ³", "ItemÃº"]
        }

        expected = {
            "string": "José",
            "number": 42,
            "nested": {"text": "Málaga"},
            "list": ["Itemó", "Itemú"]
        }

        assert fixer.fix_json_content(input_data) == expected

    def test_verify_encoding(self, fixer):
        """Test encoding verification functionality."""
        test_data = {
            "good": "José",
            "bad": "JosÃ©",
            "nested": {"bad": "MÃ¡laga"}
        }

        issues = fixer.verify_encoding(test_data)
        assert len(issues) == 2
        assert all(isinstance(issue, EncodingIssue) for issue in issues)

    @pytest.mark.asyncio
    async def test_process_file(self, fixer, input_json):
        """Test processing of a single file."""
        result = await fixer.process_file(input_json)

        assert isinstance(result, ProcessingResult)
        assert result.status == ProcessingStatus.SUCCESS
        assert result.file_path == input_json
        assert result.chars_processed > 0

    @pytest.mark.asyncio
    async def test_process_directory(self, fixer, input_json):
        """Test processing of directory."""
        stats = await fixer.process_directory()

        assert isinstance(stats, ProcessingStats)
        assert stats.processed >= 1
        assert stats.total_chars_processed > 0
        assert isinstance(stats.duration, float)
        assert stats.duration > 0

    def test_processing_stats_validation(self):
        """Test ProcessingStats model validation."""
        stats = ProcessingStats()
        assert stats.processed == 0
        assert stats.failed == 0
        assert stats.skipped == 0
        assert isinstance(stats.start_time, datetime)

    @pytest.mark.asyncio
    async def test_error_handling(self, fixer, temp_dir):
        """Test error handling with invalid JSON file."""
        # Create invalid JSON file
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{invalid json")

        result = await fixer.process_file(invalid_file)
        assert result.status == ProcessingStatus.FAILURE
        assert result.error_message is not None


if __name__ == "__main__":
    pytest.main([__file__])
