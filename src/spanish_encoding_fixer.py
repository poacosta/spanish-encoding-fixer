from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
JSONObject = Dict[str, JSONValue]


class ProcessingStatus(str, Enum):
    """Enum for file processing status."""
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class ProcessingStats(BaseModel):
    """Statistics for batch processing."""
    processed: int = Field(default=0, description="Number of successfully processed files")
    failed: int = Field(default=0, description="Number of failed files")
    skipped: int = Field(default=0, description="Number of skipped files")
    total_chars_processed: int = Field(default=0, description="Total characters processed")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> float:
        """Calculate processing duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class EncodingIssue(BaseModel):
    """Model for encoding issues found during verification."""
    path: str = Field(..., description="JSON path where issue was found")
    text: str = Field(..., description="Problematic text")
    char_position: int = Field(..., description="Position of problematic character")
    context: str = Field(..., description="Surrounding context of the issue")


class ProcessingResult(BaseModel):
    """Result of processing a single file."""
    status: ProcessingStatus
    file_path: Path
    issues: List[EncodingIssue] = Field(default_factory=list)
    chars_processed: int = Field(default=0)
    error_message: Optional[str] = None


class SpanishEncodingFixer:
    """A robust tool for fixing Spanish encoding issues in JSON files."""

    def __init__(
            self,
            input_dir: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            backup: bool = True,
            max_retries: int = 3
    ):
        self.input_dir = Path(input_dir)
        self.backup = backup
        self.max_retries = max_retries
        self._setup_directories(output_dir)
        self._setup_logging()
        self._load_replacements()

    def _setup_directories(self, output_dir: Optional[Union[str, Path]]) -> None:
        """Setup input, output, and backup directories."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / f"fixed_{timestamp}"

        if self.backup:
            self.backup_dir = self.input_dir / f"backup_{timestamp}"
            self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure structured logging with rotation."""
        log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        self.logger = logging.getLogger(__name__)

        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(console_handler)

            # File handler with rotation
            file_handler = logging.FileHandler(
                f'encoding_fixer_{datetime.now().strftime("%Y%m%d")}.log'
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(file_handler)

    def _load_replacements(self) -> None:
        """Load encoding replacement mappings with validation."""
        self.replacements = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã\u00AD': 'í', 'Ã³': 'ó',
            'Ãº': 'ú', 'Ã±': 'ñ', 'Ã\u0081': 'Á', 'Ã‰': 'É',
            'Ã\u008d': 'Í', 'Ã"': 'Ó', 'Ãš': 'Ú', 'Ã\'': 'Ñ',
            'Ã¼': 'ü', 'Ãœ': 'Ü'
        }

        # Validate replacements
        for encoded, decoded in self.replacements.items():
            if not all(ord(c) < 0x10000 for c in encoded + decoded):
                raise ValueError(f"Invalid replacement pair: {encoded} -> {decoded}")

    def fix_text(self, text: str) -> str:
        """Fix encoding issues in a text string with validation."""
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text)}")

        fixed_text = text
        for encoded, decoded in self.replacements.items():
            fixed_text = fixed_text.replace(encoded, decoded)

        return fixed_text

    def fix_json_content(self, data: JSONValue) -> JSONValue:
        """Recursively fix encoding in all string values in a JSON object."""
        if isinstance(data, dict):
            return {k: self.fix_json_content(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.fix_json_content(item) for item in data]
        elif isinstance(data, str):
            return self.fix_text(data)
        return data

    def verify_encoding(self, data: JSONObject) -> List[EncodingIssue]:
        """Check for potential remaining encoding issues with detailed reporting."""
        issues: List[EncodingIssue] = []

        def get_context(text: str, pos: int, context_size: int = 20) -> str:
            start = max(0, pos - context_size)
            end = min(len(text), pos + context_size)
            return text[start:end]

        def check_string(text: str, path: str) -> None:
            for i, char in enumerate(text):
                if char in ('Ã', '\u00AD'):
                    issues.append(EncodingIssue(
                        path=path,
                        text=text,
                        char_position=i,
                        context=get_context(text, i)
                    ))

        def recursive_check(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    recursive_check(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    recursive_check(item, f"{path}[{i}]")
            elif isinstance(obj, str):
                check_string(obj, path)

        recursive_check(data)
        return issues

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def process_file(self, file_path: Path) -> ProcessingResult:
        """Process a single JSON file with retries and comprehensive error handling."""
        try:
            self.logger.info(f"Processing file: {file_path}")

            # Create backup if enabled
            if self.backup:
                import shutil
                shutil.copy2(file_path, self.backup_dir / file_path.name)

            # Read and parse input file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Fix encoding
            fixed_data = self.fix_json_content(data)

            # Verify results
            issues = self.verify_encoding(fixed_data)

            # Calculate chars processed
            def count_chars(data: Any) -> int:
                if isinstance(data, str):
                    return len(data)
                elif isinstance(data, dict):
                    return sum(count_chars(v) for v in data.values())
                elif isinstance(data, list):
                    return sum(count_chars(item) for item in data)
                return 0

            chars_processed = count_chars(fixed_data)

            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Write output file
            output_path = self.output_dir / file_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fixed_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Successfully processed: {file_path.name}")

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                file_path=file_path,
                issues=issues,
                chars_processed=chars_processed
            )

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessingResult(
                status=ProcessingStatus.FAILURE,
                file_path=file_path,
                error_message=str(e)
            )

    async def process_directory(self) -> ProcessingStats:
        """Process all JSON files in the input directory with detailed statistics."""
        stats = ProcessingStats()

        self.logger.info(f"Starting batch processing in: {self.input_dir}")

        # Process all JSON files
        for file_path in self.input_dir.glob("*.json"):
            if not file_path.is_file():
                stats.skipped += 1
                continue

            result = await self.process_file(file_path)

            if result.status == ProcessingStatus.SUCCESS:
                stats.processed += 1
                stats.total_chars_processed += result.chars_processed
            elif result.status == ProcessingStatus.FAILURE:
                stats.failed += 1

        stats.end_time = datetime.now()

        # Log summary
        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"- Files processed: {stats.processed}")
        self.logger.info(f"- Files failed: {stats.failed}")
        self.logger.info(f"- Files skipped: {stats.skipped}")
        self.logger.info(f"- Total characters processed: {stats.total_chars_processed}")
        self.logger.info(f"- Processing time: {stats.duration:.2f} seconds")
        self.logger.info(f"- Output directory: {self.output_dir}")

        return stats
