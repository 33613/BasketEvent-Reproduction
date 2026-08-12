# Git workflow for reproduction work

## Current state

- The local clone's `origin` is the author's repository: `https://github.com/zhangyu2003/BasketEvent.git`.
- Base branch: `main`; base commit: `8a313f3ad4476735ddac38543578e19c1bccebd5`.
- Local changes are currently uncommitted.
- The root BasketEvent repository contains no `LICENSE` file. Absence of a license does **not** mean unrestricted permission to redistribute or publish modified copies.

## Recommended layout

Use two remotes in this working copy:

```text
upstream = author's repository (fetch only in normal work)
origin   = your own repository (push your branches here)
```

Create a new GitHub repository for the reproduction work. Until the author clarifies the license, prefer a **private repository**, credit the original paper/repository, and do not attach model weights or datasets.

After creating an empty private repository on GitHub, run:

```powershell
git remote rename origin upstream
git remote add origin https://github.com/<your-name>/<your-repo>.git
git remote -v
git switch -c repro/windows-rtx5060
git add .gitignore local track_one_video.py recognize.py inference.py src/model.py sam3
git status
git commit -m "Document and adapt local Windows reproduction"
git push -u origin repro/windows-rtx5060
```

Do not run `git add .` until `git status` has been reviewed. The model directories are ignored, but datasets, logs and generated outputs should also be reviewed before committing.

## Daily workflow

```powershell
git fetch upstream
git switch repro/windows-rtx5060
git status
```

When the author updates the code, first inspect the difference, then merge deliberately:

```powershell
git log --oneline --left-right repro/windows-rtx5060...upstream/main
git merge upstream/main
```

For experimental work, create focused branches such as:

```text
repro/windows-rtx5060
experiment/qwen-4bit
experiment/sam3-tracking
refactor/event-model
```

## If publishing a refactor later

Before making the repository public:

1. Ask the authors which license governs BasketEvent code and whether modified redistribution is allowed.
2. Preserve authorship and paper citations; document which parts came from upstream and which were rewritten.
3. Do not publish SAM3/Qwen/TimeSformer weights, gated files, private datasets, Hugging Face tokens or generated caches.
4. If no redistribution permission is granted, keep upstream code as a separate local clone and publish only your independently written implementation, patches, configuration and reproduction notes, after checking that they do not copy protected code.

This is a practical engineering workflow, not legal advice.
