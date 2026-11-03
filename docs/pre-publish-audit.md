# Pre-publish exposure audit

| | |
|---|---|
| audited | `p2-audit` @ `ca7ffeccaa399d12b3d38653169f359915599f10` (2026-11-03 15:37 -0500) |
| scope | what a stranger can see, download and infer from a public clone — not the code's logic |
| method | history and index queries only (`git log`, `git grep`, `git rev-list`); no filesystem walks |
| branches | `main` = 110 commits; all refs = 117 commits across 54 branches |

**Dated snapshot.** Everything below describes the tree at `ca7ffec`, 2026-11-03 15:37 EST,
and every number in it is that tree's. Two of its findings were closed 24 minutes later in
`7113ef8` — §2.2 and §7 item 1, the guard's blind spot to the bare `/home/<name>` form — and
§2.3 carries the status of both. The **Resolution** appendix at the end records which
findings closed in which commit and which are still the owner's decision; read it before
acting on §7.

This document records exposure, not blame. Where I call something *deliberate* I mean the
author clearly chose to publish it; *accidental* means it is a by-product that the author
most likely did not intend to hand to a stranger. Both are inventory, not judgement.

The repo's own convention (`CONTRIBUTING.md:31`, `docs/testing.md:35`) is that personal
absolute paths are written as `<placeholder>`, so this report does the same: the literal
under discussion is `/home/<user>`, where `<user>` is the account name that appears in
`pyproject.toml:13` and in the `origin` URL.

---

## 1. Verdicts

**History — cannot be published as-is.** 108 of 110 commits on `main` have a tree
containing `/home/<user>`, plus the real Windows account name and the hostname — redacted
here as `<user>` and `<host>` for the same reason the path is. Only the first two commits are
clean. HEAD itself still has two occurrences.
A public clone can check out any of those 108 trees.

**Secrets — clean.** No live credential in the index, none in 117 commits of history, no
secret file was ever committed and later deleted. The secret-handling code does not print,
log or transmit a token; it redacts one it finds in an error body.

**Licence — correct and internally consistent.** Apache-2.0 in `LICENSE`, `pyproject.toml`
and the README badge. No `NOTICE` file and none is required: Apache-2.0 §4(d) is
conditional on the work including one, and no third-party code is vendored.

**Tree hygiene — clean.** 281 tracked files, 5.6 MB total. No editor files, caches,
`__pycache__`, `node_modules` or generated output is tracked.

---

## 2. Git history: the exposure

### 2.1 Shape

| measure | value |
|---|---|
| commits on `main` | 110 |
| commits whose **tree** contains `/home/<user>` | **108** |
| clean commits | 2 — `a59f622` (P0), `9049578` (P1) |
| first leaking commit | `c1adb56` 2026-10-17 "P2: live d1-3B routing probe infra + results" |
| last leaking commit | `ca7ffec` (HEAD) |
| distinct files that ever held it | 22 |
| most files in a single tree | 14 (at `ea2d9e9` / `695d1de`) |
| occurrences in the HEAD tree | 2 files |
| commits not on `main` | 7, on 6 unmerged `bench/*` branches |

So the leak is not "a few commits" and not "every commit": it is **effectively the whole
history from the third commit onward**, peaking at 14 files mid-run and reduced to 2 by the
redaction commit.

A first pass measuring `git log --all -S'/home/'` reports only 20 commits. That number is
misleading and I record it here because it is the number one would naturally quote: `-S`
counts commits whose *diff* mentions the string, which includes the commits that **removed**
it. The measure that matters for exposure is the tree state, because `git checkout <sha>`
gives a stranger whatever that tree held.

### 2.2 Why the redaction missed two files

`7872d7d` ("chore: remove machine-personal paths from portable files") parameterised the
paths and added a guard, `tests/test_no_personal_paths.py`. The guard's pattern is:

```python
# tests/test_no_personal_paths.py:35
_HOME = re.compile("/" + "home" + r"/[A-Za-z0-9_.-]+/")
```

That regex requires a **trailing slash after the username**: the character class
takes the account name and the very next character must be `/`. So the pattern matches a
path with a further component after the account (home, slash, account, slash, subdir), and
it does **not** match the bare form (home, slash, account, then end-of-string, backtick,
quote or period). Reproduced by evaluating the pattern from `:35` against the two shapes:

| input shape | matches `_HOME`? |
|---|---|
| `<user>` bare — no trailing slash | **no** |
| `<user>` followed by `/` and a component | **yes** |

`python3 -m pytest tests/test_no_personal_paths.py -q` → `1 passed`. The guard is green
while two files still carry the path:

- `docs/assets-inventory.md:11` — `recursive was run over `/home/<user>` or the sshfs mount.`
  (followed by a backtick)
- `docs/review2-2026-11-03.md:344` — `"run root '.' is outside the permitted roots ('/home/<user>'); ...`
  (followed by a quote — this is a captured service error message)

Both postdate the guard: `73cc66d` and `bdca050`, both 2026-11-03.

### 2.3 What a reader actually reconstructs

From the history alone, not from any single file:

- **Linux layout and working style.** `/home/<user>/toolInterfacer/models/d1-3B-Q4_K_M.gguf`
  appears in `bench/results/*.json` from `c1adb56` onward. A reader learns the repo lived at
  `~/toolInterfacer`, that models sat in a `models/` dir inside it, and which GGUF
  quantisations were on disk.
- **A Windows box reached over sshfs.** `~/mnt/<host>` — the hostname and the
  fact that it was mounted into the Linux home directory. The mount name is the hostname.
- **The real Windows account name.** It is in the llama.cpp `/props` dumps committed in
  `151ba64`: `C:\Users\<user>\…`, where `<user>` is the literal Windows account name in the
  committed blob — not a placeholder.
  This is the single most informative historical artefact — one line gives the account name,
  the directory convention and the sibling-project path.
- **Sibling project names in the history**, even where HEAD redacts them:
  `PersonalCodingProjects`, `jevResearch`, `localAlexa`, `d1bench`.

HEAD redacts the Windows user to `C:\Users\<user>` but leaves the rest of the path, so the
**directory structure remains visible at HEAD** in **41 tracked files under `bench/`** (40
result JSONs under `bench/results/dell/` and `bench/results/dell/props/`, plus
`bench/README.md:75` where the placeholder is documented). Every one carries
`Documents\PersonalCodingProjects\personal\Research\jevResearch\lab\d1bench\models\<model>.gguf`.

**Status of §2.2 and §2.3.** Both were fixed in a forward commit after this audit: the `_HOME`
pattern now matches the bare form, both files in §2.2 are redacted, the 41 files in §2.3 carry
`<workspace>\models\<model>.gguf`, and the guard now catches the Windows structure as well as
the account name. Every measurement above describes the audited tree at `ca7ffec`.

### 2.4 Commit messages

Commit *messages* are a separate vector, and one that a content rewrite does not fix by
default.

| string | commit messages |
|---|---|
| `/home/<user>` | 0 |
| `<host>` | **3** — `c1adb56`, `412da1a`, `7872d7d` |
| `<user>` | 0 |

`7872d7d`'s message is the ironic one: it explains the redaction and names the machine three
times in doing so ("<host> paths come from env (DELL_LAB / DELL_E3_PY / ...)"). The
hostname also survives in the HEAD tree as an env-var *prefix*, `DELL_LAB`, `DELL_E3_PY`,
`DELL_TIBENCH`, `DELL_MOUNT` (`bench/lane.env.example:7-16`, `bench/lane.sh:25-32`).

### 2.5 Author identity — not a problem

All 117 commits have author and committer `Andrew Radu <72272191+andy1245-rgb@users.noreply.github.com>`.
A sweep of every added line in every commit finds exactly one email address in the entire
history: that noreply address. No personal email is present anywhere.

### 2.6 The three options

Rewriting history is destructive and permanent for anyone who has cloned, so this is the
owner's decision. The real consequences, as they apply to this repo:

**Option A — accept it and publish as-is.**
- Cost: nothing to do; the tree shipped is the tree audited.
- Consequence: a reader learns a Linux username, a Windows account name (`<user>`), a
  hostname (`<host>`), the sshfs workflow, and the layout of a personal projects tree. All
  of it is consistent with an identity that is already deliberately public (see §4), so this
  is "my machine's shape is public", not "my identity is revealed".
- Publish cleanly on top: fix the `_HOME` regex, redact the two HEAD files, delete the 6
  unmerged `bench/*` branches, and the *headline* exposure becomes a history footnote rather
  than something visible at HEAD. Note that the two HEAD fixes require a new commit, which
  is a normal forward commit, not a rewrite.
- Reversible? Yes — publishing is reversible only by deleting the repo, but the *decision*
  can be revisited at any time before the push.

**Option B — rewrite with `git filter-repo`.**
- Cost: every one of the 110 commit SHAs changes. The README's own claims that cite commits
  (`CHANGELOG.md` says "every entry traces to a commit in `git log` or to a document under
  `docs/`"; `docs/` cites shas in places) would need re-verification, because the shas they
  cite would no longer exist.
- Must be run with `--replace-text` for contents **and** `--replace-message` for the three
  `<host>` commit messages, or the hostname survives. `filter-repo` does not rewrite
  messages by default.
- The rewritten history cannot be reconciled with any existing clone. Since `origin` is
  already configured (`https://github.com/andy1245-rgb/toolInterfacer.git`) this is only
  clean if nothing has been pushed yet. Verify before starting.
- Must also handle the 6 unmerged branches, or they re-introduce the old blobs; `filter-repo`
  rewrites all refs, so they must be listed or deleted.
- Reversible? No. This is the one-way door.

**Option C — squash and republish.**
- Cost: publication becomes a single commit, "initial public release", containing the
  audited HEAD tree. All development history (110 commits, 3 weeks, the measured dead ends
  that are this project's main epistemic asset) is gone from the public artefact.
- Consequence: exposure goes to zero for everything except what is in HEAD — but two
  occurrences are still in HEAD today, so the squash must include the two file fixes, or it
  squashes the leak forward.
- The trade is real: the commit history is a large part of what makes this repo worth
  reading (`docs/` repeatedly cites commits as evidence), and a squash discards it.
- Reversible? No for the published repo; the private history can be kept on a private remote.

My read: **A, with the forward fixes, is the right default** — the identity is already
deliberately public, so the history leak is machine-shape rather than person-identity.
B is only worth its cost if `<host>`/`<user>` are considered sensitive in their own right.
C is the cleanest exposure and the worst repo.

---

## 3. Secrets

### 3.1 Tracked tree

| sweep | result |
|---|---|
| `sk-`, `ghp_`, `gho_`, `github_pat_`, `AKIA`, `sk-ant-`, `sk-or-v1` | 2 hits, both deliberate |
| `PRIVATE KEY` blocks (RSA/OPENSSH/EC/PGP) | none |
| `Authorization:` / `Bearer` values | 10 hits, all code paths or docs, no literal tokens |
| `.env` | not tracked; `.gitignore:14`; only `bench/lane.env.example` is tracked |
| key/cert file extensions (`.pem`, `.key`, `.p12`, `.pfx`, `id_rsa`) | none |

The two `sk-` hits are the seeded fixtures, and they are the correct kind of fixture:

```
tests/test_distill.py:41:        FAKE_KEY = "sk-or-v1-0000...0000"
tests/test_openrouter_backend.py:33:  TEST_KEY = "sk-or-v1-0000...0000"
```

Both are 60 zeros after the prefix — not a plausible key, and not live. **Seeded-secret check:
pass.** The pattern is named `FAKE_KEY`/`TEST_KEY`, which is what makes it unambiguous to a
future reader and to a scanner.

One near-miss worth recording so it is not re-reported as a finding:
`bench/results/or-bench-tokens-2026-10-26.json` matches a filename grep for "token" and is
**token-accounting telemetry**, not credentials — 0 hits for `api_key|authorization|bearer|secret`
and no key-shaped strings. It is not on `main` at all (it exists only on branch
`bench/or-models`, added in `da16e7f`).

### 3.2 History

- `git log --all --diff-filter=A --name-only` for secret-shaped names returns exactly one
  file: the `or-bench-tokens` JSON above. No `.env`, no key, no credential file was ever
  added.
- Files ever deleted: one, `tests/data/devskills/web-fetch/fixtures/https-example-com/combined.json`
  (`2b4452d` "drop a test artefact that got committed") — a fetched-HTML fixture, not a
  secret. **The "deleted from HEAD but still in history" trap does not bite here.**
- Every added line across all 117 commits contains exactly two key-shaped strings, both the
  all-zeros fixtures above.

### 3.3 The secret-handling code

`service/ti/http/auth.py` — does not print or log a token:

- No `logging` import at all; nothing in the module emits the value.
- Every error message interpolates the **path**, never the value:
  `f"{path!r} has mode {mode:03o}: group and other can read it..."` (`auth.py:122-124`).
- Comparison is constant-time: `hmac.compare_digest(supplied.encode(), expected.encode())`
  (`auth.py:89`).
- Refuses a token file with `mode & 0o077` (`auth.py:121`), and creates one `O_EXCL` at
  `0o600` (`auth.py:143`). Generated from `secrets.token_urlsafe(32)` (`auth.py:141`).
- No network code in the module; the token never leaves as anything but an inbound check.

`pi-extension/src/token.ts` — does not print or transmit:

- `serviceToken()` reads env then file, caches, returns (`token.ts:37-51`). No `console.*`,
  no throw carrying the value; the `catch` at `token.ts:48` returns `""`.
- The only consumer sends it as a header: `bridge.ts:43`
  `headers: { "Content-Type": "application/json", [TOKEN_HEADER]: serviceToken() }`.
- The 401 path names the *knob* rather than echoing the token (`bridge.ts:62-65`).

Adjacent credential path, checked because it handles a paid API key:

- `service/ti/decide/backends/openrouter.py` reads the key per call from the env
  (`:180`, `_api_key` at `:242`), sends it only in the `Authorization` header (`:186`), and
  **redacts it out of anything it builds**: `_redact()` (`:252-254`) is applied to the error
  body (`:191`) and to the transport reason (`:205`). The key is deliberately not an
  attribute (`:128`). The five `log.debug` calls (`:142`, `:156`, `:210`, `:236`, `:239`)
  carry candidate counts, byte counts and provider names — never the key.
- `tools/distill/teacher.py` inherits this unchanged and adds nothing that touches the key.

A repo-wide grep for a logging call near a token/key variable returns only
`bench/demo.py:325` (prose mentioning "API key"), `tools/distill/stats.py:195` and
`tools/distill/utterances.py:126` (both printing a config *name* / grouping key).

---

## 4. What the repo reveals about its owner

The identity is **deliberately public** and consistent across three files, so nothing below
de-anonymises anyone:

| finding | where | deliberate? |
|---|---|---|
| Name "Andrew Radu", copyright | `LICENSE:189`, `pyproject.toml:13` | deliberate |
| GitHub account `andy1245-rgb`, noreply email | `pyproject.toml:13`, `:41-44`, `origin` | deliberate |
| Public repo URL in four places | `pyproject.toml:40-44` | deliberate |

Given that, the remaining disclosures are about **machine and projects**, not person:

| finding | where | my read |
|---|---|---|
| Survey of **35 private sibling project directories** (39 table rows), with file inventories, byte sizes and what to import from each | `docs/assets-inventory.md` (300 lines) | **deliberate inclusion, accidental disclosure.** Committed on purpose (`73cc66d`, "inventory the sibling projects for reusable decision-model assets") but written for the operator as a working note. This is the single largest disclosure surface in the repo. |
| `~/agentFleet`, `~/swarmHarness`, `~/decisionModelTuning`, `~/toolInterfacerExtras` named, plus `mirror/.../localAlexa/{jevResearchCompiled,minicpmTune,...}` and `Research/{jevResearch,localLLMfunReseach,moneyLM}` | `docs/assets-inventory.md:29-53` | same |
| A sibling project characterised as "grey-market compound price/consensus gathering" | `docs/assets-inventory.md:62` | **accidental** — a substantive characterisation of unrelated private work |
| A sibling project described as "a 10.5 KB money-making research sweep", filename `RESEARCH_money_ideas_2026-10-09.md` | `docs/assets-inventory.md:63` | accidental |
| Byte sizes of private checkpoints/adapters (242,047,978 B; 35,335,380 B; LoRA exports at 15.4/27.4/63.4 MB) | `docs/assets-inventory.md:41-44` | accidental — unnecessary detail for a case study |
| `mcm-portal` (a university course portal checkout) as a future data source | `docs/roadmap-2026-11.md:77` | **deliberate** — one line in a roadmap, no private detail |
| `localAlexa` / `jevResearch` provenance for imported data | `tests/data/routing_382_PROVENANCE.md:9` | **deliberate and correct** — this is attribution for imported data, and should stay |
| Hostname `<host>` as env-var prefix | `bench/lane.env.example:7-16`, `bench/lane.sh:25-32` | deliberate anonymisation that stops one step short (machine called "bench-host" in prose) |
| Hardware: Intel Arc 140V iGPU, `-ngl 99`, Python 3.10.12 | `README.md:347`, `docs/classical-baseline-2026-11-03.md:28` | deliberate — it is the measurement context, and a reader needs it |
| Canada (Ubuntu `ca.archive.ubuntu.com` mirror) + `-0400`/`-0500` commit offsets | `docs/vulkan-igpu-2026-11-03.md:78` | accidental, low value: timezone is in every commit anyway |
| Private-voice framing, "the operator" | `README.md`, `docs/` ×4 | accidental — a public doc addressing "the operator" reads as an internal memo. Cosmetic, not exposure. |

The 2 residual `/home/<user>` occurrences at HEAD (§2.2) belong in this table too: they are
**accidental**, and they are the only place where the username itself is visible at HEAD.

---

## 5. Licence and attribution

**Apache-2.0, consistent in three places.** `LICENSE` is the full 201-line Apache License
2.0 with the appendix filled in (`LICENSE:189` "Copyright 2026 Andrew Radu"). `pyproject.toml:11-12`
declares `license = "Apache-2.0"` and `license-files = ["LICENSE"]`. `README.md:4` badge and
`README.md:411-413` both say Apache-2.0. No contradiction anywhere.

**NOTICE: none, and none required.** Apache-2.0 §4(d) (`LICENSE:106`) is conditional — "If
the Work includes a NOTICE text file". No such file exists, no tracked file is third-party,
and `extern/` is gitignored (`.gitignore:13`) and absent from disk. So the §4(d) obligation
is never triggered. There is nothing to declare in `pyproject.toml` about it.

**No per-file licence headers anywhere** (`git grep -l 'SPDX-License-Identifier'` → none).
Apache-2.0 does not require them; the appendix boilerplate is the recommended *application*
form, not a distribution obligation. Worth knowing only because it means a future vendored
file would arrive with no marker and no obvious place to put one.

**Vendored third-party code: none.** A sweep of `bench/`, `tools/`, `service/` and
`pi-extension/` for `copied from|adapted from|vendored|MIT License|BSD|GPL|from github.com`
returns one real hit, `bench/sv2lane.sh:3` "adapted from bench/lane.sh" — same repository.
The rest are `package-lock.json` dependency licence strings, which is correct and required.

**`service/ti/decide/linear_text.py` — an API-shape concern, not a licence concern.**

It reimplements `TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)` and a
one-vs-rest linear SVM in the standard library because `pyproject.toml:35` declares
`dependencies = []`. The module imports only `hashlib, json, math, random, re,
dataclasses` — no scikit-learn source is present, and the docstring is explicit that it is a
"re-implementation" reproducing the *defaults*.

Reproducing documented default hyper-parameters and behaviour is not copyrightable
expression, so there is no licence obligation owed to scikit-learn here. Even if source had
been copied, scikit-learn is BSD-3-Clause, which requires attribution and the BSD text — not
a NOTICE file — so it would still not have created a NOTICE requirement. The real risk is
the other one the docstring already names: **comparability**, not licensing. A score from a
reimplementation is only meaningful next to a recorded scikit-learn score if the defaults
match exactly, which is precisely the claim the docstring makes and the reason the
fingerprint machinery exists. That is an API-fidelity claim to keep honest, not a licence
problem to fix.

**Imported data has attribution.** `tests/data/routing_382.json` and `bench/registry_382.json`
are byte-identical copies from a sibling project, with sha256 digests and the origin
recorded in `tests/data/routing_382_PROVENANCE.md:1-29`. That is the owner's own prior work,
so no third-party licence is implicated — but the provenance record is the thing that makes
the claim auditable, and it should stay.

---

## 6. What should not be published

### 6.1 Artefacts

Clean. `git ls-files | xargs du -h | sort -rh` shows:

| | |
|---|---|
| total tracked | 281 files, **5.6 MB** |
| largest | `demo/toolinterfacer.png` 236K |
| next 20 | all `bench/results/dell/*.json` at 88–100K |

- No editor or OS files (`.DS_Store`, `.swp`, `.idea`, `.vscode`, `Thumbs.db`): none tracked.
- No caches or bytecode (`__pycache__`, `.pyc`, `.pytest_cache`, `htmlcov`): none tracked.
- No `node_modules`, no `*.gguf`, no model weights: `.gitignore:7-12` covers them and none
  is present.
- `bench/results/` is evidence and should stay — it is cited throughout `docs/` as the proof
  of specific claims (the phantom-lane bug, the quantisation comparison at `TODO.md:47`).
  The `-run2` duplicates are a second measurement, not a mistake: they are what makes the
  pooled numbers honest.
- `demo/toolinterfacer.png` and `toolinterfacer.svg` are rendered from the tracked
  `demo/toolinterfacer.cast` (`demo/README.md:38-40`), so their full text content is
  auditable. I checked the cast rather than the pixels: it contains **only** `127.0.0.1` —
  no username, no path, no hostname, no email. The PNG carries no EXIF or textual metadata.
  **The rendered demo does not leak.**

### 6.2 Docs describing systems a reader cannot have

Three dangling references — files cited that do not exist. These are loose ends, not
case studies, and each sends a reader to a dead link on first contact:

| reference | cited at | status |
|---|---|---|
| `docs/session-recall.md` | `docs/roadmap-2026-11.md:32` | missing — and the row is about recalling work across `toolInterfacer, jevResearch, localAlexa, cactus/needle`, which compounds the §4 exposure |
| `docs/dataset-spec.md` | `docs/assets-inventory.md:222` | missing |
| `docs/project_reference.md` | `docs/assets-inventory.md:62` | missing (a path inside a *different private project*, so it can never resolve here) |

`docs/assets-inventory.md` as a whole is the borderline case: it is a legitimate case study
of "what prior work exists and what is worth importing", but written as a private inventory.
It names 35 private directories and their file contents. As a public document its useful
content is the *conclusions* (which assets were worth importing, and which measured
negatives saved work), and roughly none of that requires the directory listings.

`docs/quickstart-stranger-2026-11-03.md` is the opposite case and is the model to follow: it
is written explicitly for a stranger's first clone and names no private system.

---

## 7. The three fixes worth doing before the push

Ranked by exposure removed per unit of effort. All three are forward commits — none of them
requires a rewrite, so none of them forecloses §2.6 option B later.

1. **Close the guard's blind spot and the two HEAD occurrences.** Change
   `tests/test_no_personal_paths.py:35` to drop the mandatory trailing slash (the `_USERS`
   pattern at `:36` already does not require one, so the two patterns are inconsistent), fix
   `docs/assets-inventory.md:11` and `docs/review2-2026-11-03.md:344`, and add a test that
   fails on the bare form. Right now the repo advertises a portability guarantee that its
   own test cannot enforce — that is worse than no guard, because it invites trust.

2. **Decide what `docs/assets-inventory.md` is for, and cut it to that.** Publishing 35
   private sibling directories, checkpoint byte sizes, and the characterisation of two
   unrelated private projects ("grey-market compound price/consensus gathering", "a
   money-making research sweep") is the largest disclosure in the repo and the least
   connected to what a stranger needs. The import conclusions survive the cut; the
   directory listings do not.

3. **Delete the 6 unmerged `bench/*` branches before publishing, or audit them.** They carry
   7 commits that are not on `main`, including
   `bench/results/or-bench-tokens-2026-10-26.json` and a further copy of the pre-redaction
   `C:\Users\<user>\...` paths. They add nothing the public wants (the analysis landed on
   `main`) and they are 6 refs a `git push --all` publishes silently.

Item beyond these three, worth one line: the three `<host>` commit messages
(`c1adb56`, `412da1a`, `7872d7d`) survive any content-only rewrite. If the hostname is
sensitive, only §2.6 option B with `--replace-message` addresses them.

---

## Appendix — commands behind the headline numbers

```
git rev-list HEAD --count                       -> 110
git rev-list --all --count                      -> 117
git log --all --format='%an <%ae>' | sort -u    -> 1 identity (noreply)
# tree-state scan, per commit (the number that matters)
#   for each sha in `git rev-list --reverse main`: `git grep -l <pattern> <sha>`
                                                -> 108 / 110 trees contain the path
# diff-state scan (the misleading one)
git log --all --oneline -S'/home/' -- .          -> 20 commits
python3 -m pytest tests/test_no_personal_paths.py -q
                                                -> 1 passed  (while 2 files still hold it)
git log --all --diff-filter=A --name-only --format=''
    | grep -iE '\.env$|secret|credential|token' -> 1 hit (token-count telemetry)
git log --all -p --format='' | grep -oE '<key patterns>' -> 2 hits, both all-zeros fixtures
git ls-files -z | xargs -0 du -ch | tail -1     -> 5.6M
```

---

## Resolution — appended 2026-11-03 16:20 EST

*Appended to a dated audit. Nothing above this line has been rewritten:
everything there describes the tree at `ca7ffec`, 2026-11-03 15:37 EST. Two of
its headline findings were closed 24 minutes later.*

### What was fixed, in which commit

Commit **`7113ef8`** — "chore: close the path leak the guard could not see",
2026-11-03 16:01:17 — closes §2.2 and §2.3 together.

| finding | was (at `ca7ffec`) | now (at `a105d50`) |
|---|---|---|
| §2.2, `docs/assets-inventory.md:11` | the Linux account path followed by a backtick | `$HOME` |
| §2.2, `docs/review2-2026-11-03.md:344` | the same path captured inside a service error message | `('$HOME')` — which is what the service now prints |
| §2.3, **41 tracked files under `bench/`** | `C:\Users\<user>` + the sibling-project tree + `models\<model>.gguf`, in every recorded `model.path` | `<workspace>\models\<model>.gguf` — 40 result JSONs under `bench/results/dell/` (including `props/`) plus `bench/README.md:75`, which documents the placeholder |
| §2.3's carriage of the **Windows account name** | literal in every recorded blob | gone; `<workspace>` is the repo's own placeholder form |

Measured on this worktree at `a105d50`: all 40 result JSONs carry
`<workspace>\models\`, and no scanned file in the tree matches any of the guard's
three patterns — `python3 -m pytest tests/test_no_personal_paths.py -q` → **8
passed, 39 subtests**. §2.2's appendix recorded `1 passed` while two files still
held the path, which is exactly the defect this commit exists to close.

The same edit also made `bench/classical_baseline.py` runnable again: `e523336`'s
substitution left the account prefix unescaped, so 37 of the 40 result files
failed `json.loads` and the script died in its loader. That episode and its
closure are recorded in `docs/classical-baseline-2026-11-03.md` §10.

### The guard was rewritten, not re-pointed

`tests/test_no_personal_paths.py` no longer has a pattern the bare form slips past.
It carries three, plus a `GuardPatternTest` class that pins what each must catch
*and* the near-misses each must let through — the part §2.2 was really missing:

- `_HOME` — `/home/<name>` with a terminator set covering end-of-string,
  whitespace, a backtick, quotes, and every punctuation a captured error path gets
  quoted with. The mandatory trailing `/` is gone.
- `_USERS` — a drive-rooted `Users\<name>` (either separator), same terminator
  rule, so the bare form matches here too.
- `_WIN_STRUCTURE` — a drive-rooted path that still carries a `Documents` or
  `models` component: the *shape* §2.3 complained about, without hardcoding this
  machine's sibling directories.

Two exemptions were removed rather than widened: `bench/results/` is scanned now
(its recorded paths are redacted and the model basename — the part that proved the
phantom-lane bug — is intact), and `node_modules` is skipped by directory *name*
anywhere instead of by root prefix, which is what had made the suite fail on the
extension's own dependency tree.

### What remains — the owner's decision, unchanged

- **History.** 108 of `main`'s 110 trees still contain the path. Nothing in the
  working tree touches that; only a rewrite does. §2.6 option A remains my read.
- **Commit messages.** Three carry `<host>` (`c1adb56`, `412da1a`, `7872d7d`), and
  a content-only rewrite does not fix them (§2.4).
- **The 6 unmerged `bench/*` branches** — 7 commits, and a further copy of the
  pre-redaction paths (§7 item 3).
- **`docs/assets-inventory.md`'s scope** (§7 item 2): the directory listings,
  checkpoint byte sizes and the two characterisations of unrelated private projects
  are untouched.
- **Three dangling doc references** (§6.2): `docs/session-recall.md`,
  `docs/dataset-spec.md` and `docs/project_reference.md` are still cited and still
  do not exist. This pass found one more in the same family — `docs/assets-inventory.md`
  names `bench/registry.json` as where the imported registry would land; the file
  that landed is `bench/registry_382.json`.

§1's headline — "history cannot be published as-is" — stands. What changed is the
*HEAD* half of it: 2 occurrences became 0, and the guard can see the shape that hid
them.
