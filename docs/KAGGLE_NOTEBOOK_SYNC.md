# Kaggle master notebook sync

The authoritative latest training notebook is maintained on Kaggle:

https://www.kaggle.com/code/muhammadfarhanmrs/noorix-master-ai-training-muhammad-farhan

The GitHub repository intentionally links to this Kaggle notebook but does not contain its latest edited notebook file yet. Do not replace it with an older local master notebook or a generated placeholder: the owner has made manual additions in the Kaggle version.

## Recommended sync into GitHub

1. Open the authoritative Kaggle notebook and confirm the latest saved version contains the manual additions.
2. Use Kaggle's available GitHub connection/export action if it is enabled for the account. Authorize only the intended GitHub repository.
3. Commit the notebook to this repository under:
   `notebooks/official_kaggle/NOORIX_MASTER_AI_TRAINING_MUHAMMAD_FARHAN.ipynb`
4. If the Kaggle UI does not offer a GitHub connection, download the notebook as an .ipynb file and add that exact downloaded file to the same repository path. Do not copy notebook cell output or recreate it manually.
5. Open a pull request. GitHub Actions validates notebook JSON and syntax where applicable.

## Important

- Keep the Kaggle URL as the source-of-truth link until the exact latest notebook file has been committed and reviewed.
- Preserve all manually added cells, metadata, training settings, and output references.
- Notebook validation checks structure and Python syntax; it does not rerun Kaggle GPU training or prove model quality.
- Never commit Kaggle API tokens, secrets, private datasets, or credentials in notebook outputs.
