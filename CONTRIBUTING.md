# Contributing

Install the locked environment using the README. Use an isolated database and synthetic plates for development. For changes to geometry, signal logic, tracking or OCR, include the smallest reproducible clip or generated trajectory that demonstrates the failure.

Run the checks used in CI from the repository root.

```sh
python -m pip install ruff==0.17.6
ruff check trafficlab tests
ruff format --check trafficlab tests
python -m unittest discover -s tests -v
python -m trafficlab.experiment --out runs/review
```

Include the command, input manifest hash and before/after event counts in proposals that change the detector. Explain new false positives or misses and their effect on review. Keep held-out labels separate from threshold tuning. Include migrations and replay tests for changes to persisted state.

For dependency updates, edit the constraints in `pyproject.toml`, regenerate the hash-locked base environment using the command at the top of `requirements.lock`, and test on the supported Python versions. Use small commits and describe observable behavior in each pull request. Public issues and attachments should use synthetic or de-identified examples. Handle suspected credential leaks and private evidence through the repository owner's private contact channel.
