---
name: fix-compare-sorting
overview: Verified `compare_command` already sorts ascending for rmse/mae/mape and descending for r2. I’ll make that intent explicit (and case-insensitive) so the behavior is unambiguous and easier to audit.
todos:
  - id: update-compare-sort
    content: Make metric ordering explicit in compare_command
    status: completed
  - id: sanity-check
    content: Quickly sanity-check sort order for rmse vs r2
    status: completed
---

# Clarify Compare Metric Sorting

## Scope

- Update the sorting logic in [cli.py](/Users/ryanbergner/micromamba/github-repos/EnterpriseDemandForecast/cli.py) inside `compare_command` to derive `ascending` from an explicit list of lower-is-better metrics.

## Steps

- Replace the inline condition `ascending=(args.metric != 'r2')` with a clearer block, e.g.
  ```
  metric_name = args.metric.lower()
  lower_is_better = {"rmse", "mae", "mape"}
  ascending = metric_name in lower_is_better
  ```

- Keep `metric_col` unchanged and use `ascending` in the `sort_values` call; optionally add a short comment that r2 sorts descending while error metrics sort ascending.

## Files

- [cli.py](/Users/ryanbergner/micromamba/github-repos/EnterpriseDemandForecast/cli.py)