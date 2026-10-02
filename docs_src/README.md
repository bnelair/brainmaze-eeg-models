# Documentation source

Sphinx sources of the brainmaze-eeg-models documentation. Build locally:

```bash
pip install -r docs_src/requirements.txt -e .
sphinx-build -b html docs_src/source docs
```

`docs/` is ignored by git. The **Docs** workflow publishes `main` to the GitHub Pages site
root and `dev` under `/dev/` (gh-pages branch; Pages source "Deploy from a branch: gh-pages /").
