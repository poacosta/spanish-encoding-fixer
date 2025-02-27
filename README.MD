# Spanish Encoding Fixer

Ever wrestled with mangled Spanish text in JSON or CSV files? You know, when your beautiful "más allá" turns into a horrifying "mÃ¡s allÃ¡"? Yeah, we've all been there. This tool is your knight in shining armor for those encoding nightmares, now with added CSV superpowers! 🗡️ 🛡️

## Features

- **Multiformat Support**: JSON and CSV files are both fair game now
- **Async Processing**: Because life's too short to wait for sequential file processing
- **Smart Format Detection**: Automatically identifies and handles different file types
- **CSV Dialect Sniffing**: Detects delimiter, quoting style, and headers automatically
- **Robust Error Handling**: With retries that would make a persistence hunter proud
- **Automatic Backups**: Because we're paranoid (in a good way)
- **Detailed Logging**: Know exactly what's happening, when, and where
- **Type Safety**: Pydantic models keeping everything in check
- **Progress Tracking**: Watch those files get fixed in real-time
- **Comprehensive Validation**: No encoding issue escapes our notice

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install pydantic tenacity
```

### Basic Usage

```python
import asyncio
from src.spanish_encoding_fixer import SpanishEncodingFixer, FileType


async def main():
    fixer = SpanishEncodingFixer(
        input_dir="path/to/your/files",
        backup=True,
        max_retries=3,
        # Process both JSON and CSV files (default)
        file_types={FileType.JSON, FileType.CSV}
    )
    stats = await fixer.process_directory()
    print(f"Processing completed in {stats.duration:.2f} seconds")
    print(f"JSON files processed: {stats.json_files_processed}")
    print(f"CSV files processed: {stats.csv_files_processed}")


if __name__ == "__main__":
    asyncio.run(main())
```

### CSV-Only Mode

```python
import asyncio
from src.spanish_encoding_fixer import SpanishEncodingFixer, FileType


async def main():
    # Focus only on CSV files
    fixer = SpanishEncodingFixer(
        input_dir="path/to/your/csv/files",
        backup=True,
        file_types={FileType.CSV}
    )
    stats = await fixer.process_directory()
    print(f"Processed {stats.csv_files_processed} CSV files")


if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 How It Works

The fixer employs a battle-tested strategy to handle common Spanish encoding issues:

1. **Detection**: Identifies file types and problematic character sequences
2. **Format-Specific Parsing**: Handles JSON and CSV with specialized parsers
3. **Transformation**: Applies carefully mapped character replacements
4. **Validation**: Verifies the results to ensure quality
5. **Backup**: Keeps your original files safe
6. **Reporting**: Provides detailed statistics and logs

## 📊 Enhanced Processing Stats

Get comprehensive statistics about your processing run:

```python
ProcessingStats(
    processed=42,  # Successfully processed files
    failed=0,  # Failed files
    skipped=2,  # Skipped files
    json_files_processed=30,  # JSON files fixed
    csv_files_processed=12,  # CSV files fixed
    total_chars_processed=123456,  # Total characters fixed
    duration=3.14  # Processing time in seconds
)
```

## 🛠 Advanced Usage

### Custom CSV Dialect

```python
fixer = SpanishEncodingFixer(
    input_dir="data",
    csv_dialect="excel-tab"  # Use tab-delimited dialect as fallback
)
```

### Custom Replacement Mappings

```python
fixer = SpanishEncodingFixer(
    input_dir="data",
    replacements={
        'Ã¡': 'á',
        'Ã©': 'é',
        # Add your own mappings
    }
)
```

### Error Handling Configuration

```python
from tenacity import stop_after_attempt, wait_exponential

fixer = SpanishEncodingFixer(
    input_dir="data",
    retry_config={
        "stop": stop_after_attempt(5),
        "wait": wait_exponential(multiplier=1, min=4, max=10)
    }
)
```

## 📝 Logging

The tool provides detailed logging with rotation:

```plaintext
2024-12-11 00:41:57,115 - INFO - [main.py:182] - Processing file: data/example.json
2024-12-11 00:41:57,116 - INFO - [main.py:223] - Successfully processed: example.json
2024-12-11 00:41:57,220 - INFO - [main.py:182] - Processing file: data/customers.csv
2024-12-11 00:41:57,225 - INFO - [main.py:223] - Successfully processed: customers.csv
```

## CSV-Specific Features

The CSV processor includes special handling for:

- **Dialect Detection**: Automatically identifies delimiters, quote characters, and other formatting
- **Header Preservation**: Maintains your column headers while fixing the content
- **Row-by-Row Processing**: Handles each CSV record individually for maximum reliability
- **Robust Error Recovery**: Gracefully handles malformed CSV data when possible

## 🧪 Testing

Run the test suite to ensure everything works as expected:

```bash
python -m unittest discover tests
# Or with pytest
python -m pytest tests/
```

## 🤝 Contributing

Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🐛 Known Issues

- Root-level JSON arrays require special handling
- Large files (>1GB) may require chunked processing
- Some exotic Unicode combinations might need manual review
- CSV files with inconsistent columns may need special attention
- Excel-generated CSVs sometimes need specific dialect settings

---

Made with ❤️, plenty of ☕, and now with CSV power! 📊