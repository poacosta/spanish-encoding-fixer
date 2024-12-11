from src.spanish_encoding_fixer import SpanishEncodingFixer


async def main():
    """Example usage with async support."""
    input_dir = "data"  # Change this to your input directory
    fixer = SpanishEncodingFixer(
        input_dir=input_dir,
        backup=True,
        max_retries=3
    )
    stats = await fixer.process_directory()
    print(f"Processing completed in {stats.duration:.2f} seconds")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
