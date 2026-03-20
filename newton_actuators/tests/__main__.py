# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test runner entry point for control package."""

import os
import sys
import unittest


def main():
    """Discover and run all tests in the tests directory."""
    start_dir = os.path.dirname(__file__)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir, pattern="test*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
