# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agricultural Statistics Dashboard — a Streamlit app with AI-powered analysis summaries. Source lives in `src/dream_project/`, test data in `data/`.

## Commands

```bash
# Install dependencies
uv sync

# Run the Streamlit app
uv run streamlit run src/dream_project/app.py

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_name -v

# Lint / format
uv run ruff check src/
uv run black src/
```

## Architecture

- **Entry point:** `src/dream_project/app.py` — Streamlit app
- **Data:** `data/` — CSV and Excel files for each statistical test type:
  - `ttest_one_sample.csv`, `ttest_two_sample.csv` — t-tests
  - `ftest_variance_long.csv`, `ftest_variance_wide.csv` — F-test (variance)
  - `wheat_data_Correlation_Regression.csv` — correlation & regression
  - `Agricultural_Data_(Rice_Trail)_Descriptive_Statistics.xlsx` — descriptive stats
- **AI summaries:** OpenAI API via key `dp_OPENAI_API_KEY` in `.env`
- **Charts:** seaborn (not Streamlit's built-in charting)
- **Exports:** python-docx for Word doc generation, openpyxl for Excel

## Style Notes

- Line length: 120 characters (black + ruff configured in `pyproject.toml`)
- Target Python: 3.9+
