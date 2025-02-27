import json
import tempfile
import unittest
from pathlib import Path
import csv
import asyncio
import shutil
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.spanish_encoding_fixer import (
    SpanishEncodingFixer,
    FileType,
    ProcessingStatus
)


class TestSpanishEncodingFixer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())

        # Prepare test data
        self.sample_json = {
            "name": "JosÃ© GarcÃ\u00ADa",
            "city": "MÃ¡laga",
            "occupation": "MÃºsico",
            "details": {
                "education": "Universidad de SevillÃ¡",
                "hobbies": ["MÃºsica", "FÃºtbol", "MontaÃ±ismo"]
            }
        }

        self.sample_csv_data = [
            ["name", "city", "occupation"],
            ["JosÃ© GarcÃ\u00ADa", "MÃ¡laga", "MÃºsico"],
            ["MarÃ\u00ADa RodrÃ\u00ADguez", "MÃ¡drid", "IngenierÃ\u00ADa"],
            ["Juan PÃ©rez", "BarcelÃ³na", "MÃ©dico"]
        ]

        # Create test files
        self.json_file = self.test_dir / "test.json"
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_json, f, ensure_ascii=False)

        self.csv_file = self.test_dir / "test.csv"
        with open(self.csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for row in self.sample_csv_data:
                writer.writerow(row)

        # Expected fixed data
        self.expected_json = {
            "name": "José García",
            "city": "Málaga",
            "occupation": "Músico",
            "details": {
                "education": "Universidad de Sevillá",
                "hobbies": ["Música", "Fútbol", "Montañismo"]
            }
        }

        self.expected_csv_rows = [
            {"name": "José García", "city": "Málaga", "occupation": "Músico"},
            {"name": "María Rodríguez", "city": "Mádrid", "occupation": "Ingeniaría"},
            {"name": "Juan Pérez", "city": "Barcelóna", "occupation": "Médico"}
        ]

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_fix_text(self):
        fixer = SpanishEncodingFixer(input_dir=self.test_dir)

        test_string = "Hola, cÃ³mo estÃ¡s? Bien, graÃ±as."
        expected = "Hola, cómo estás? Bien, grañas."

        result = fixer.fix_text(test_string)
        self.assertEqual(result, expected)

    def test_determine_file_type(self):
        fixer = SpanishEncodingFixer(input_dir=self.test_dir)

        self.assertEqual(fixer.determine_file_type(Path("test.json")), FileType.JSON)
        self.assertEqual(fixer.determine_file_type(Path("test.csv")), FileType.CSV)
        self.assertEqual(fixer.determine_file_type(Path("test.txt")), FileType.UNKNOWN)

    async def test_process_json_file(self):
        fixer = SpanishEncodingFixer(input_dir=self.test_dir)

        result = await fixer.process_json_file(self.json_file)

        self.assertEqual(result.status, ProcessingStatus.SUCCESS)
        self.assertEqual(result.file_type, FileType.JSON)
        self.assertTrue(result.chars_processed > 0)

        # Check if output file exists and contains fixed data
        output_file = list(fixer.output_dir.glob("*.json"))[0]
        with open(output_file, "r", encoding="utf-8") as f:
            fixed_data = json.load(f)

        self.assertEqual(fixed_data, self.expected_json)

    async def test_process_csv_file(self):
        fixer = SpanishEncodingFixer(input_dir=self.test_dir)

        result = await fixer.process_csv_file(self.csv_file)

        self.assertEqual(result.status, ProcessingStatus.SUCCESS)
        self.assertEqual(result.file_type, FileType.CSV)
        self.assertTrue(result.chars_processed > 0)

        # Check if output file exists and contains fixed data
        output_file = list(fixer.output_dir.glob("*.csv"))[0]
        with open(output_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fixed_rows = list(reader)

        # Compare contents (we only care about the text values being fixed)
        for i, row in enumerate(fixed_rows):
            for key, value in row.items():
                expected_value = self.expected_csv_rows[i][key]
                self.assertEqual(value, expected_value)

    async def test_process_directory(self):
        fixer = SpanishEncodingFixer(input_dir=self.test_dir)

        stats = await fixer.process_directory()

        self.assertEqual(stats.processed, 2)
        self.assertEqual(stats.json_files_processed, 1)
        self.assertEqual(stats.csv_files_processed, 1)
        self.assertEqual(stats.failed, 0)
        self.assertTrue(stats.total_chars_processed > 0)

    async def test_file_type_filtering(self):
        # Test with only JSON files
        json_fixer = SpanishEncodingFixer(
            input_dir=self.test_dir,
            file_types={FileType.JSON}
        )

        json_stats = await json_fixer.process_directory()

        self.assertEqual(json_stats.processed, 1)
        self.assertEqual(json_stats.json_files_processed, 1)
        self.assertEqual(json_stats.csv_files_processed, 0)

        # Test with only CSV files
        csv_fixer = SpanishEncodingFixer(
            input_dir=self.test_dir,
            file_types={FileType.CSV}
        )

        csv_stats = await csv_fixer.process_directory()

        self.assertEqual(csv_stats.processed, 1)
        self.assertEqual(csv_stats.json_files_processed, 0)
        self.assertEqual(csv_stats.csv_files_processed, 1)


# Run tests with async support
def run_async_test(test_case, test_name):
    """Helper function to run an async test method."""
    test_method = getattr(test_case, test_name)
    return asyncio.run(test_method())


if __name__ == "__main__":
    # Modify TestCase to handle async methods
    original_run = unittest.TestCase.run


    def async_aware_run(self, result=None):
        test_method_name = self._testMethodName
        test_method = getattr(self, test_method_name)
        if asyncio.iscoroutinefunction(test_method):
            setattr(self, test_method_name, lambda: asyncio.run(test_method()))
        return original_run(self, result)


    unittest.TestCase.run = async_aware_run

    unittest.main()
