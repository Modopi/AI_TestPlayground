"""Helper to create .ipynb notebook files from cell definitions."""
import json
import os

def md(source: str) -> dict:
    """Create a markdown cell."""
    lines = source.split('\n')
    src = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(source: str) -> dict:
    """Create a code cell."""
    lines = source.split('\n')
    src = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def create_notebook(cells: list, path: str):
    """Write a notebook file from a list of cells."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    # Validate
    with open(path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    n_md = sum(1 for c in loaded['cells'] if c['cell_type'] == 'markdown')
    n_code = sum(1 for c in loaded['cells'] if c['cell_type'] == 'code')
    print(f"✅ {path}: {len(loaded['cells'])} cells (MD:{n_md}, CODE:{n_code})")
    return path
