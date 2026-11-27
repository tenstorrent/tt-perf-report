# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os

# Render plots headless and the same way in every test. Set before perf_report imports
# pyplot, so no test has to switch backends mid-run.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from tt_perf_report import perf_report


@pytest.fixture
def reset_classification_cache():
    """Rebuild the category cache and the unclassified-warning set around a test.

    Both are module globals, so a test that relies on a warning being printed, or on a category
    table it patched, would otherwise depend on what earlier tests classified.
    """
    perf_report.OPERATION_CATEGORIES_EXTENDED = None
    perf_report._UNCLASSIFIED_OPS_WARNED.clear()
    yield
    perf_report.OPERATION_CATEGORIES_EXTENDED = None
    perf_report._UNCLASSIFIED_OPS_WARNED.clear()
