# STA5073Z Assignment 2 - team project (due 16 October)

## Roles (fill in; submission text file needs names + what each person did)
| Member | Main tasks | Files they own |
|---|---|---|
| A | Task 1 exploration, report intro | R/01_explore.R |
| B | Tasks 2 + 4 (CNN, splits/leakage) | R/02_splits_cnn.R |
| C | Tasks 3 + 5 (Siamese, hyperparameters) | R/03_siamese.R, R/04_hpo.R |
| All | Task 6 discussion, report sections, presentation | index.qmd |

## Git workflow (assessed: branches, PRs, shared files)
```bash
git clone <repo-url> && cd <repo>
git checkout -b task2-cnn            # one branch per task/person
# work, then:
git add -p && git commit -m "CNN: add per-digit accuracy"
git push -u origin task2-cnn         # then open a Pull Request on GitHub
git pull --rebase origin main        # before starting new work
```
- Review each other's PRs before merging. Everyone should also edit `index.qmd` (different sections) so history shows shared work.
- Small, frequent commits with clear messages.

## Run order
1. Put `handwriting.rds` in `data/` (it is git-ignored; check whether you may publish it).
2. `R/01_explore.R`, `R/02_splits_cnn.R`, `R/03_siamese.R`, `R/04_hpo.R` (04 needs 02 run first).
3. `quarto render` builds the site into `docs/`.

## Publishing (programmatic, not drag-and-drop)
`quarto render && git add docs && git commit -m "render" && git push`, with Pages set to deploy from `main` / `/docs`
(or `quarto publish gh-pages`).

## Deliverables checklist
- [ ] Quarto report (index.qmd), <= 5000 words, declarations + AI use statement
- [ ] Text file: names + roles, GitHub Pages link, repo link
- [ ] 20-minute presentation, every member presents
- [ ] Every member can explain every step of the code


c = 2
d = 5
f = c+d
print(f)
