Reproducing Kazakh NLP tables and figures

From the repository root:

  python verify_manifest.py
  python -m pip install -r requirements.txt
  python analysis/reproduce.py

Use Python 3.12. The script verifies the saved evidence and original CSV hashes,
recreates ten LaTeX tables and four figure families, and writes to a fresh
analysis/regenerated/ directory. Table outputs must match exactly; figure
metadata and platform font rendering can differ.

This is presentation reproduction from completed aggregate analysis. It does
not generate model predictions or recompute the bootstrap. See
docs/REPRODUCIBILITY.md for experimental source, data identity and rerun scope.
