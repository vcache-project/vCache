#!/usr/bin/env python3
import importlib.util
import sys
import unittest
from pathlib import Path


def load_tests_from_file(file_path):
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def run_tests():
    tests_dir = Path(__file__).parent.absolute()
    test_suite = unittest.TestSuite()
    file_paths = list(tests_dir.glob("**/*.py"))

    for file_path in tests_dir.glob("**/*.py"):
        if file_path.name == "runner.py" or file_path.name == "__init__.py":
            continue

        file_is_empty = file_path.stat().st_size == 0
        if file_is_empty:
            continue

        try:
            suite = load_tests_from_file(file_path)
            test_suite.addTest(suite)
            file_paths.append(file_path)
        except Exception as e:
            print(f"Error loading tests from {file_path}: {e}")

    print(f"Running {len(file_paths)} tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
