# Recreated issue from instructor's repo

Original: https://github.com/alamdari/CPSC620-air-quality-dashboard/issues/1

Issue: Some missing values in the Air Quality dataset are encoded as -200 and are not being treated as NaN. This causes summary stats and visualizations to include these sentinel values.

Steps to reproduce:
1. Load `data/AirQualityUCI.csv` using `analysis.load_data` and `analysis.clean_data`.
2. Inspect numeric columns (for example `CO(GT)`) for values equal to -200.

Expected behavior: All `-200` sentinel values should be treated as NaN and excluded from statistics and plots.

Proposed fix: Ensure `analysis.clean_data` normalizes all common `-200` representations (numeric and string forms) to `NaN` and that `pd.to_numeric` conversion doesn't leave `-200` behind.

Patch: Update `analysis.clean_data` to replace `-200`, `'-200'`, `'-200.0'`, and `'-200,0'` with `NaN` and re-check numeric columns after conversion.
