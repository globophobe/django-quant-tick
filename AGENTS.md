## Commands

- Run tests with `cd demo && python manage.py test quant_tick`.
- Run `ruff check .` to lint.
- Project tasks live in `demo/tasks.py`; use `cd demo && invoke --list` to inspect them.
- Common task shortcuts:
  - `cd demo && invoke coverage`
  - `cd demo && invoke makemigrations`
  - `cd demo && invoke migrate`
  - `cd demo && invoke format`

## Project Structure

- `demo/` is the thin Django project and deployment harness.
- `quant_tick/` is the reusable application package.
- `demo/tasks.py` is the task automation entrypoint.

## Domain Notes

- `notional` is the physical quantity.
- `volume` is the quote-value amount.
- Use `Decimal` for financial calculations.

## Code Style

- Keep code concise and readable.
- Prefer minimal comments.
- Keep imports at the top of the file.
- Use distinct variable names; avoid shadowing.
- Avoid unnecessary intermediate variables unless they improve readability.
- Use short, clear abbreviations consistently, for example `df` for data frame and `cfg` for config.
- Avoid calling functions directly inside f-strings.
