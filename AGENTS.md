# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a TensorFlow educational notebook repository (Korean-language). It contains 17 chapters of Jupyter notebooks (.ipynb), Python generation scripts under `scripts/`, and utility modules under `utils/`. There is no web application, database, or Docker service.

### Running notebooks

- Notebooks use the `tf_study` Jupyter kernel. Execute with:
  ```
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=120 --ExecutePreprocessor.kernel_name=tf_study <notebook.ipynb>
  ```
- Notebooks save plots to hardcoded Apple Silicon paths like `/Users/alex/AI_TestPlayground/...`. The update script creates these directories so `plt.savefig()` calls don't fail. If new chapters reference new sub-paths under `/Users/alex/AI_TestPlayground/`, you may need to create them.
- GPU (CUDA) is not available in this environment; TensorFlow runs on CPU. Notebooks work fine on CPU — GPU-specific warnings in stderr are expected and harmless.
- `matplotlib.use('Agg')` is required (headless). All existing notebooks already use it.

### Lint

No project-level linter config exists. Use `ruff check` for ad-hoc Python linting:
```
ruff check --select E,W,F <file_or_dir>
```

### Generation scripts

Notebook content generators live in `scripts/gen_ch*.py` and use `scripts/nb_helper.py`. Run individually:
```
python3 scripts/gen_ch12_01.py
```

### Key gotcha

- `main.py` has a known typo (`tf.keras.dataset` instead of `tf.keras.datasets`). Do not modify it — it is part of the existing repo.
- The `environment.yml` targets Apple Silicon (`tensorflow-macos`). On Linux, use standard `tensorflow` via pip instead.
- Ensure `$HOME/.local/bin` is on PATH for `jupyter`, `ruff`, and other pip-installed CLI tools.
