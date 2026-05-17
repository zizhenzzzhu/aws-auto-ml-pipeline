# Contributing

Use `main` as the stable branch and create short-lived feature branches for
pipeline changes. Protect `main` with pull requests, CI checks, and review before
merge.

Recommended branch flow:

```text
main
  -> feature/databricks-pull
  -> feature/validation-gates
  -> fix/model-validation
```

Run local checks before opening a pull request:

```powershell
python -m compileall -q .
pytest -q
```
