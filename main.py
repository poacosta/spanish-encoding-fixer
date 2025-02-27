import asyncio
from src.spanish_encoding_fixer import SpanishEncodingFixer, FileType


async def main():
    """Example usage with async support."""
    input_dir = "data"  # Change this to your input directory
    fixer = SpanishEncodingFixer(
        input_dir=input_dir,
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