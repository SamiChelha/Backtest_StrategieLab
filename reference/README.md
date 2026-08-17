# Reference

Standalone academic reference implementation, kept for traceability with the
written thesis.

`thesis_code.py` is a self-contained script (numpy / pandas / scipy only) that
implements the performance metrics and the walk-forward backtest from first
principles. It duplicates parts of `src/strategy_lab/` on purpose: the package
is the maintained implementation used by the app, this file is the frozen
version the thesis results were derived from.

Run it directly:

```bash
uv run python reference/thesis_code.py
```
