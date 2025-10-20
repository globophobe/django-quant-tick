## Running Tasks

All project tasks are defined in `demo/tasks.py` using the `invoke` library.

### Common Commands

Run tests:
```bash
cd demo && invoke test
```

Run tests with coverage:
```bash
cd demo && invoke coverage
```

Make migrations:
```bash
cd demo && invoke makemigrations
```

Run migrations:
```bash
cd demo && invoke migrate
```

### Available Tasks
See `demo/tasks.py` for all available tasks. Run `invoke --list` to see all commands.

## Project Structure
- `demo/` - Main Django application
- `quant_tick/` - Core application package
- `demo/tasks.py` - Task automation using invoke

## Code Style
- All imports at top of file
- Prefer minimal code comments
- No parenthetical clarifications in comments or docstrings
- Write concise, readable code
- Use distinct variable names to avoid shadowing
- Avoid unnecessary intermediate variables unless better for readability
- Use short but clear abbreviations consistently, e.g `data_frame` -> `df`, `config` -> `cfg`
- Avoid calling functions inside f-strings
