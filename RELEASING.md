# Releasing `brainmaze-eeg-models`

This package follows the BrainMaze family release process. The full guide (how a release
works, recovering from a failed release, one-time setup, branch protection, dependency order,
compatibility policy, troubleshooting) is in
**[bnelair/brainmaze-sphinx RELEASING.md](https://github.com/bnelair/brainmaze-sphinx/blob/main/RELEASING.md)**.

## Quick steps

1. Make sure `main` has what you want to release and CI is green.
2. **Actions → Prepare release → Run workflow**, pick `patch` / `minor` / `major`
   (changed numerical results and API changes count as breaking; while the version is 0.x a
   breaking change bumps `minor`). **For the first release pick `minor`** (0.0.0 → 0.1.0, see
   "First release" below).
3. Review the bot's **"Release vX.Y.Z"** PR and check that its diff is exactly the one
   `version = "X.Y.Z"` line in `pyproject.toml`. CI and the Version guard don't run on it
   (PRs opened with `GITHUB_TOKEN` trigger no workflows), so this check is yours. Then
   squash-merge it.
4. The merge triggers the **Release** workflow automatically (it has no manual trigger). It
   tests, builds, checks the artifact contents (only the package and its `.onnx` models),
   publishes to [PyPI](https://pypi.org/project/brainmaze-eeg-models/) with the organisation
   API token `PYPI_Token_General`, pushes tag `vX.Y.Z` and creates the GitHub Release. Watch it
   under *Actions → Release*. Only one Release run executes at a time (concurrency group
   `release-caller-bnelair/brainmaze-eeg-models`).

Never edit `[project].version` in a normal PR. The **Version guard** check flags it (it parses
`[project].version` from `pyproject.toml` on both sides, and for the bot's bump PR checks that
nothing else changed). The check is **advisory**: no ruleset requires it (maintainer decision;
making it required would block the bot's bump PRs, on which no workflow runs), so reviewers must
not merge a PR it flags.

## First release (0.1.0)

`pyproject.toml` starts at the placeholder version **`0.0.0`**, which is never published:

- Every push to `main` runs *Release*. The shared guard sees no `v0.0.0` tag and therefore runs
  the tests and the build, but this repository's `publish` job is skipped for `0.0.0` (and its
  first step refuses `0.0.0` explicitly), so nothing is uploaded or tagged.
- *Prepare release* with `minor` turns `0.0.0` into **`0.1.0`**. Merging that PR publishes
  `brainmaze-eeg-models 0.1.0`, the first version on PyPI.

### The first upload of a new PyPI project and the organisation token

`brainmaze-eeg-models` does not exist on PyPI yet. Whether the organisation token
`PYPI_Token_General` can create it depends on its **scope**:

- **Account-wide token** ("Entire account (all projects)"): the first upload creates the
  project; nothing else to do.
- **Project-scoped token** (scoped to existing projects such as `brainmaze-torch`): PyPI rejects
  the first upload of a new project (HTTP 403, "Invalid or non-existent authentication
  information" / "project-scoped token is not valid for project"). The publish job then fails
  at the upload step; nothing is published or tagged.

Two ways out, in order of preference. In both, **don't re-run the failed run**: *Re-run
failed jobs* re-runs the same commit with the same workflow file, so it repeats the 403
(unless only the token itself changed, e.g. its scope was extended; secrets are read again on a
re-run).

**A. Upload the failed run's build by hand (no workflow change).** The failed run already built
and checked the exact files; nothing was published or tagged.

1. Download its `dist` artifact: `gh run download <run-id> -n dist -D dist` (kept 90 days).
2. An owner of the bnelair PyPI projects uploads it once with a token that may create projects
   (an account-wide token): `twine upload dist/*` (user `__token__`).
3. Tag the commit that run built (the run's head SHA) and create the release by hand, as in the
   "tag push" row of the recovery table below.
4. For later releases, extend `PYPI_Token_General`'s scope to the new project (or replace it by
   a project-scoped token); `release.yml` then works unchanged.

**B. A PyPI "pending trusted publisher" + a workflow PR.** A pending publisher lets a GitHub
workflow create a project that does not exist yet, without any token:

1. On PyPI, log in as an owner of the bnelair projects, open
   **Your account → Publishing → Add a new pending publisher → GitHub**, and enter:
   PyPI project name `brainmaze-eeg-models`, owner `bnelair`, repository `brainmaze-eeg-models`,
   workflow name `release.yml`, environment name empty.
2. Open a normal PR that changes `.github/workflows/release.yml`, `publish` job: add
   `id-token: write` to its `permissions`, remove the "Check the PyPI token is available" step
   and remove the `password:` input of the "Publish to PyPI" step.
3. **Merging that PR is the release**: the merge starts a new *Release* run on the merge
   commit, and because `vX.Y.Z` is still untagged that run publishes and tags `X.Y.Z` from the
   merge commit (same package content; only the workflow file changed). Watch that run; do
   not re-run the old failed one.

After the first upload the pending publisher becomes a normal trusted publisher of the
project. The maintainer can keep Trusted Publishing (as brainmaze-eeg and brainmaze-utils do)
or go back to an API token for later releases (e.g. a token scoped to this project, or extend
the organisation token's scope), restoring the token check and the `password:` input.

## Recovering from a failed release

If the Release run fails part-way, **don't wait for the next push to `main`** (that would
publish and tag a different commit) and **don't run Prepare release again** (that would skip
the version). Go by the step that failed (full table:
[Recovering from a failed release](https://github.com/bnelair/brainmaze-sphinx/blob/main/RELEASING.md#recovering-from-a-failed-release)):

| failed step | state | what to do |
|---|---|---|
| guard / test / build / artifact check, or the token check / upload | nothing published or tagged | If the cause is outside the repository (a flaky runner, the token secret's value or scope, see above), fix it and **Re-run failed jobs** on that same run (possible for 30 days; the `dist` artifact is kept 90 days). If the fix needs a change to the code or a workflow file, a re-run does not see it (it reruns the same commit and workflow file): merge the fix in a PR instead; the merge starts a new *Release* run that publishes the still-untagged version. |
| tag push, after a successful upload | PyPI has X.Y.Z, no tag | Do **not** re-run: PyPI files are immutable, so the upload step would fail. Tag the commit that run built (the run's head SHA) by hand: `git tag vX.Y.Z <sha> && git push origin vX.Y.Z`, then `gh release create vX.Y.Z --verify-tag --title vX.Y.Z --generate-notes`. |
| upload half-succeeded (some files on PyPI, e.g. wheel but not sdist) | PyPI has some files of X.Y.Z, no tag | **Re-run failed jobs** fails (PyPI rejects the files already there) and the workflow deliberately has no `skip-existing`. Download the `dist` artifact of the failed run (`gh run download <run-id> -n dist -D dist`; it is the exact build) and upload only the missing files: `twine upload dist/<missing-file>` (token `PYPI_Token_General`, user `__token__`). Check on PyPI that all files are present, then tag the run's head SHA by hand and create the release as in the tag-push row. |
| `gh release create`, after the tag push | PyPI + tag, no GitHub Release | `gh release create vX.Y.Z --verify-tag --title vX.Y.Z --generate-notes`. |

## This repository

| | |
|---|---|
| PyPI | [`brainmaze-eeg-models`](https://pypi.org/project/brainmaze-eeg-models/) (not yet published) |
| import | `import brainmaze_eeg_models` (`brainmaze_eeg_models.__version__` comes from installed metadata) |
| thin callers of [brainmaze-sphinx](https://github.com/bnelair/brainmaze-sphinx) | `ci.yml` (tests + artifact check), `docs.yml` (GitHub Pages), `prepare-release.yml` |
| with local logic | `release.yml`: calls the shared guard + test + build + artifact check, then its own `publish` job checks the token, uploads to PyPI, pushes the tag and creates the GitHub Release (skipped for the placeholder `0.0.0`). `version-guard.yml`: entirely local. |
| PyPI upload | org-level Actions secret `PYPI_Token_General` (API token), used only by the publish job in `release.yml` |
| release artifacts | only the package and its trained models `brainmaze_eeg_models/*/_models/*.onnx` (`allowed-assets`); every model is listed in `required-assets` in `ci.yml` **and** `release.yml` (added there by the PR that adds the model; `tests/test_package.py` checks that both lists equal the shipped `.onnx` files), so a model missing from the wheel or the sdist fails CI. Tests, docs, demos, the export tooling and sample data never ship. |

`ci.yml` and `release.yml` pass `dist-check`, `allowed-assets` and `required-assets`, which the
shared workflows accept since [brainmaze-sphinx#4](https://github.com/bnelair/brainmaze-sphinx/pull/4).

## Publishing credentials

By maintainer decision (as for brainmaze-torch and brainmaze-zmq), `release.yml` publishes with
the organisation API token `PYPI_Token_General` (brainmaze-eeg and brainmaze-utils use Trusted
Publishing). The repository itself has no Actions secrets, so the token is an
organisation-level secret shared with this repository (organisation settings → Secrets and
variables → Actions → `PYPI_Token_General` → repository access; already the case on
2026-10-02). The publish job
first checks that the secret is non-empty and fails with a clear error otherwise; nothing is
published or tagged in that case. See "First release" above for the token scope.

*Prepare release* needs **Settings → Actions → General → "Allow GitHub Actions to create and
approve pull requests"** (organisation and repository level).

## GitHub Pages

The **Docs** workflow pushes the built site to the `gh-pages` branch (created by its first run on
`main`). After that first run, set **Settings → Pages → Source: Deploy from a branch,
`gh-pages` / (root)**; the site is then <https://bnelair.github.io/brainmaze-eeg-models/>.
