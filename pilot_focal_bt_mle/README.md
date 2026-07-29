# Constrained BT-MLE Utilities

This small support package contains the matched-transcript statistics and
box-constrained Bradley-Terry optimizer used by
`icml_pareto_track_stop`. The directory name is retained for import
compatibility with the original exploratory pilot.

The published API is implemented in `matched_fixed_budget.py`:

- focal/random-opponent transcript accumulation;
- focal Borda estimation;
- coordinate-wise BT negative log-likelihood and gradient;
- projection onto the zero-sum box with optional order constraints;
- box-constrained coordinate and multi-objective BT fits.

Historical pilot drivers, generated data, and pilot reports remain local and
are ignored. They are not part of the main synthetic comparison or the current
VB-EGE versus Pareto BT-GLR formal report.

Targeted tests live at `tests/test_pilot_matched_fixed_budget.py` and run with
the repository-wide test command:

```bash
python -m pytest -q
```
