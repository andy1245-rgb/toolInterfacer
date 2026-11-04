# What the token grants: the filesystem boundary, written down — 2026-11-04

Read this before you point a model at `useTool`.

The boundary below already existed. Nothing here changes a default, narrows a
policy, or adds a check. What this document adds is the sentence that was missing: a
widening of the access policy was made for a good reason, and the *consequence* of
that widening was never written down. `grep -E "\.ssh|id_ed25519|private key|\.gnupg"
--include='*.md' -r .` returned zero hits in the entire repository before this file
existed, while the default it concerns appears in `README.md`, `CHANGELOG.md` and
`ARCHITECTURE.md`. Three files describe the knob; none said what turning it that far
means.

§7 makes one recommendation and states what it would cost. §8 records the process
failure that produced the gap, because it is the reusable part.

---

## 1. The sentence

> **A valid `X-TI-Token` authorises reading any file the operator owns.** With the
> config the repository ships that is every file under `$HOME` — `~` is a policy root,
> so the home directory itself is a legal run root — plus everything under `/tmp` and
> under the directory the service was started in. Refused: `/etc`, `/root`, `/srv`,
> `/opt`, another user's home, unless the service happens to be running there.

"Reading" here is literal — `filesys_read`, `filesys_glob` and `filesys_grep`, which
are the only ops an L0 request may name, and all three are read-only. There is no
write, no delete, no execute on that surface.

So the honest framing of the token is not "the extension's shared secret". It is a
**complete filesystem-authorisation boundary for everything the operator owns**, and
the secret is the only thing standing between any local process and that boundary.

---

## 2. What the token authenticates

`service/ti/http/auth.py`. The token authenticates **a request**, not a person and not
a session:

- Every `POST /v1/useTool` and `POST /v1/plan` must carry it in `X-TI-Token`, or as
  `Authorization: Bearer <token>`. The comparison is `hmac.compare_digest`, so a
  timing oracle is not available.
- `GET /health` is deliberately open — `ensureService()` polls it to decide whether to
  spawn a second service, so an authenticated health check would spawn one per session.
  It discloses the op list, a `conf: true/false` flag and which decide backend
  answered. It discloses nothing about files.
- There is no unauthenticated mode. `require_token()` raises rather than let a server
  start without one, and `api.serve` calls it before it binds.

The secret itself comes from three places, in order: `TI_HTTP_TOKEN` in the service's
environment; the runtime file at `http.token_file` (default
`~/.local/state/ti/http.token`), which must be `0600` — a file that group or other can
read is refused, with the `chmod` in the message; or generated on first start (32
bytes of `secrets.token_urlsafe`) and written `0600`.

Two consequences worth stating plainly:

1. **The file source makes the token durable and not scoped to a run.** Environment
   tokens die with the session; the file does not. "Authenticated" means "held the
   shared secret", never "was granted this particular run".
2. **The token file is inside the grant it authorises.** `~/.local/state/ti/` is under
   `$HOME`, so a caller who holds a token can read the file that mints the next one
   (§4). That is not an escalation — a holder already has everything the token buys —
   but it does mean the credential survives the caller losing its own copy.

---

## 3. What it authorises, mechanically

`service/ti/http/api.py` runs the guards in a fixed order, and the order is the answer
to "what can a request reach":

| step | guard | what it stops |
|---|---|---|
| token | `auth.authorized()` → `401` | anything on the box that does not have the secret |
| body | `411`/`413`/`400` | no `Content-Length`, over `http.max_body_bytes` (262144 B), not a JSON object |
| route | `404` | anything but the two POSTs |
| op allowlist | `400` | an L0 body naming an op outside `http.allowed_ops` — three `filesys_*` mirrors by default |
| scope root | `403` | a `digest.cwd` that does not resolve inside `ops.filesys.roots` |
| path args | `403` | a `params.args.*` path that escapes the run root |

Past those, a request gets a **run root**: `digest.cwd`, or the service's own cwd when
the body names none. The root becomes the confinement frame for every path the run
touches (`confine.resolve`), so within it:

- a relative path resolves against the root, and `..` is collapsed lexically before
  any syscall — climbing out is refused, not walked out of;
- a symlink is followed component by component, and one whose target leaves the root
  is refused; chains are capped at 32 hops so a cycle terminates;
- a path at or under an `ops.filesys.deny_roots` entry is refused **before the first
  `lstat`**, and the walk prunes it before descending.

That is a sound confinement implementation. The question this document exists to answer
is what the *outer* frame is, because that is the part a caller chooses and the part
the defaults set.

The read surface inside the frame is bounded per request, not per run:

| op | bound | source |
|---|---|---|
| `filesys_read` | 2000 lines / 262144 B per file, `conf` drops to `0.7` when truncated | `ops.filesys.read.*` |
| `filesys_glob` | depth 12, 20000 files visited, 2000 hits, 5 s deadline | `ops.filesys.*` |
| `filesys_grep` | 200 hits, 50 files per directory, pattern ≤ 200 chars, line ≤ 400 chars, pattern shapes refused before compilation | `ops.filesys.grep.*`, `regexguard` |

None of these bound *how many requests* you make. There is a bounded worker pool
(`http.max_workers`, 16) and no rate limit, so `262144 B` per request × as many
requests as you care to send is the real shape of the exposure. An Ed25519 private key
is a few hundred bytes, and the read cap is 262144 of them per request.

One thing this document does **not** cover: the same token also reaches the plan layer,
which spawns the `web_search`, `web_fetch` and `reddit_digest` CLIs with arguments the
planner chooses. That is a different boundary (child processes rather than file reads)
and is out of scope here; the point of mentioning it is that the token is not *only* a
filesystem grant.

---

## 4. What `~` admits — enumerated from `confine.py`, not from this document's sources

The rule, in the order the code applies it (both excerpts abridged to the loop that
decides; the full bodies are `confine.py`):

```python
# ops/filesys/confine.py
for entry in filesys_conf().get("roots") or ["~", "."]:
    text = str(entry)
    ...
    path = real_root(text)     # expanduser -> absolute -> normpath -> deny -> symlinks
```

and then, per request:

```python
def confined_root(cwd=None, *, base=None) -> str:
    roots = policy_roots()
    for policy in roots:
        try:
            return resolve(candidate, base=frame, root=policy)
        except DeniedPath as exc:  denied = exc
        except PathRefused as exc: last = exc
```

A cwd is admitted if it resolves inside **any** policy root. `service/ti/conf/defaults.json`
ships three:

| entry | what it evaluates to | why it is there |
|---|---|---|
| `"~"` | `os.path.expanduser("~")` — the **`$HOME` of the service process** | the session's cwd is some project under the operator's home, while the service itself is spawned in the repository |
| `"."` | the directory the service was **started** in | a clone that lives outside `$HOME` |
| `"/tmp"` | `/tmp` | scratch space, and `git clone` into `/tmp` |

`~` is first in the list, so in practice it decides: any `digest.cwd` under `$HOME`
is accepted at the first root and the other two are never consulted. `deny_roots()`
ships **empty**, so nothing inside `$HOME` is refused.

`~` is not "the user's home" in any abstract sense — it is `expanduser` in the
service's own environment. Run the service as another account, or with `HOME` pointed
elsewhere, and the grant follows that.

### What that rule reaches

The rule is the authority; this table is the evidence — the standard subtrees that
exist under a developer's home, each of which is (a) a legal `digest.cwd`, and (b) a
legal read target once it is the root:

| path under `$HOME` | what a `filesys_read` gets from it |
|---|---|
| `~/.ssh` | `id_ed25519`, `id_rsa`, `authorized_keys`, `known_hosts`, `config` — every private key the operator has |
| `~/.gnupg` | `private-keys-v1.d/`, `pubring.kbx` — the GPG secret keyring |
| `~/.aws` | `credentials`, `config` — long-lived access keys |
| `~/.config` | per-application state, including CLI OAuth tokens (`gh/hosts.yml`, cloud CLIs) |
| `~/.local/state/ti` | **the token file that authorised the request** (`http.token_file`) |
| `~/.kube/config`, `~/.docker/config.json` | cluster and registry credentials |
| `~/.netrc`, `~/.git-credentials`, `~/.npmrc`, `~/.pypirc` | plaintext credentials, by design |
| `~/.bash_history`, `~/.zsh_history` | commands, including any typed with a secret inline |
| browser profiles, `~/.local/share/keyrings` | cookie jars and the desktop keyring |
| any project tree | `.env`, `*.pem`, `secrets.*`, whatever the operator's own code reads |

Measured on the box this document was written on, through `confine` itself:

```
policy_roots()  -> ['/home/<user>', '/home/<user>/<the checkout the service was started in>', '/tmp']
deny_roots()    -> []

confined_root('~/.ssh')               -> /home/<user>/.ssh                      ALLOWED
confined_root('~/.gnupg')             -> /home/<user>/.gnupg                    ALLOWED
confined_root('~/.aws')               -> /home/<user>/.aws                      ALLOWED
confined_root('~/.config')            -> /home/<user>/.config                   ALLOWED
confined_root('~/.local/state/ti')    -> /home/<user>/.local/state/ti           ALLOWED
confined_root('~/mnt')                -> /home/<user>/mnt                       ALLOWED
confined_root('/etc')                 -> OutsideRoot: run root '/etc' is outside the permitted roots …
```

and then, with that root in place:

```
resolve('id_ed25519', root='~/.ssh')  -> /home/<user>/.ssh/id_ed25519   exists: True
```

So the end-to-end request is three fields and a header:

```
POST /v1/useTool
X-TI-Token: <the token>

{"digest": {"cwd": "~/.ssh"},
 "params": {"tool": "filesys_read", "args": {"path": "id_ed25519"}}}
```

Every guard in §3 passes. Nothing in that request is malformed, and no check is
bypassed — the policy root said yes.

### The `~/mnt` carve-out is configuration, not code

The one path in `$HOME` that is refused *on the operator's own box* is `~/mnt` — an
sshfs mount where a single `os.stat` can block a thread in uninterruptible sleep for
hours (see `ARCHITECTURE.md`, "The deny list"). That refusal comes from
`ops.filesys.deny_roots: ["~/mnt"]` in the operator's own overlay. **The repository
ships `deny_roots: []`**, so a stranger who clones this and runs it has no carve-out at
all: `~/mnt` is admitted like everything else. Do not read the operator's behaviour as
the default's behaviour.

---

## 5. Why the default is what it is

The reason is portability, and it is a good reason. Two claims from the code's own
docstring, both verified elsewhere in `docs/`:

1. **The service is not started where the work is.** `pi-extension/src/lifecycle.ts`
   spawns it with the repository as its cwd, while a run is about the *session's* cwd —
   some project under the operator's home. A policy of `["."]` alone therefore refuses
   the ordinary case.
2. **The README's first example must work from a clone anywhere.** This is NB4 from
   `docs/review2-2026-11-03.md`:

   > | NB4 | `ops.filesys.roots` ships as `~`, so a cwd outside `$HOME` is refused on
   > every call | **closed for the case it measured** | `conf/defaults.json` now ships
   > `roots: ["~", ".", "/tmp"]`. Re-measured: `confined_root('/tmp')` returns `/tmp`;
   > `/srv/x`, `/opt` and `/` are still refused. The policy is unchanged; the default
   > breadth is what moved |

   The failure NB4 records is real and was measured against a stranger's first contact:
   `docs/quickstart-stranger-2026-11-03.md` §1 got

   ```
   403 {"error": "scope_refused",
        "detail": "run root '.' is outside the permitted roots ('/tmp/stranger-home'); …"}
   ```

   from the README's own `useTool` example, because the clone was in `/tmp` and the only
   permitted root was `~`.

So the widening was correct and forced. Two details of the implementation are worth
knowing because they are not what a reader would assume:

- **`/tmp` is not in the code's fallback.** `policy_roots()` falls back to `["~", "."]`
  when the key is missing or empty; `/tmp` exists only because `defaults.json` lists it.
  A config that omits the key silently loses `/tmp` too.
- **`~` is checked first, so it is the operative root.** `.` and `/tmp` only decide
  requests whose cwd is outside `$HOME`. When someone says "the roots are `~`, `.`,
  `/tmp`", the part that governs an ordinary session is `~`.

What the widening cost, and what nobody wrote down: it moved the boundary from "the
repository and the temp directory" to "everything the operator owns", and the sentence
that describes it stayed a portability sentence — *"so a clone runs wherever it lives"*
(`CHANGELOG.md`). No security sentence was added next to it, in any of the three files
that document the knob.

---

## 6. What a tighter default would cost

Three candidate changes, each with the cost that has to be paid, not just the risk it
removes.

**(a) Keep `~`, document it.** What this document does. Cost: the exposure above stays,
and the mitigation is entirely "the operator read the docs". There is nothing to
maintain and nothing that can regress.

**(b) Narrow to an explicitly configured root, with `~` opt-in.** Cost: this re-creates
NB4 exactly. `roots` is what makes a clone work with no configuration; requiring an
overlay before the first successful call makes the README's first example fail out of
the box, which is the failure a previous review already found and fixed once. It also
needs the extension to pass an explicit root per session (a protocol change in
`pi-extension`, not a config change), because the service's own cwd is the repository,
not the project. This is the option that looks strictest and is the most likely to be
reverted under pressure.

**(c) Deny-list the credential directories.** Ship something like
`deny_roots: ["~/.ssh", "~/.gnupg", "~/.aws", "~/.local/state/ti", "~/.kube",
"~/.docker", "~/.git-credentials", "~/.netrc"]`. Cost, in order of importance:

- **It is incomplete by construction**, and an incomplete denylist is worse than a
  documented open door, because it produces a *belief* of protection. `.env` files,
  `secrets.yaml`, `~/.config/gh/hosts.yml`, browser cookie stores, keyrings, and every
  token in the operator's own project trees remain readable. The list can only ever
  cover the subtrees someone thought of.
- **It re-uses a mechanism whose stated purpose is different.** `deny_roots` exists so
  that a *dead network mount* is never `lstat`ed; its docstring justifies emptiness on
  exactly that ground ("a deny root is a mount point on *this* machine"). Widening it
  to credentials is defensible — the check already runs before every syscall and on the
  resolved path — but the docstring, `ARCHITECTURE.md` and `docs/ops.md` §"The deny
  list" all have to be rewritten to say both reasons, or the next reader will assume
  the list is about mounts and delete a credential entry as machine-specific noise.
- **It changes what a legitimate run can see.** A walk that would have descended into
  one of those trees now prunes it and reports `conf` `0.7` instead of `1.0`
  (`ops/filesys/impl.py::_confidence`), which is honest but visible.
- **`~`-expanded entries are portable, but only for the conventional layout.** An XDG
  config in a non-default location, or a keyring under a different path, is silently
  not covered.
- **It is a security default, so the change is the owner's call** and needs its own
  test — which is why this document recommends it rather than doing it.

---

## 7. Recommendation

**Keep `["~", ".", "/tmp"]`. Ship a short `deny_roots` covering the credential
subtrees no legitimate `useTool` intent ever needs, and make the incompleteness the
headline rather than a footnote.**

The argument is that these are two different decisions that the current default
conflates:

- **Where a run may be rooted** is a usability question, and `~` is the right answer.
  Narrowing it re-breaks the first-contact path (NB4) and needs a protocol change to
  the extension. Leave it.
- **What a run may read anywhere** is a safety question, and `~` is the wrong answer
  for a fixed set of subtrees that no legitimate `useTool` intent ever needs:
  `~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.local/state/ti` (the token itself), `~/.kube`,
  `~/.docker`, `~/.netrc`, `~/.git-credentials`. Denying those costs nothing that any
  real intent wants and closes the worst single-request exfiltration: `{"cwd": "~/.ssh"}`
  + `filesys_read "id_ed25519"`.

`deny_roots` is the right mechanism for it because it is the only check in `confine.py`
that refuses **before a syscall**, applies to the run root as well as to every path
inside it, and is already `~`-expanded, so the entries stay portable between homes. The
code path exists; only its contents and its documentation would change.

**What it costs, stated as a cost:** the denylist is a denylist. It will not stop a
`filesys_read` of `~/.config/gh/hosts.yml` or `~/project/.env`, and it must be
documented as reducing exposure rather than closing it. It also re-purposes a knob
whose documented reason is uninterruptible-sleep avoidance, so three documents need the
second reason written into them, and the change itself is a security-default change that
needs the owner's sign-off — which is why this document recommends it instead of doing
it.

---

## 8. What the review process did not look at

The gap was produced by a review loop working correctly and then stopping one step
short.

A reviewer found that the default was **too narrow**: a clone outside `$HOME` got a
`403` on the README's first example (`docs/quickstart-stranger-2026-11-03.md` §1,
NB4). The fix was measured — `confined_root('/tmp')` now returns `/tmp`, the example
works — and documented, in a sentence about clones.

What no pass asked was the *other* direction: **what does the widened default now
cover?** The same review that measured `confined_root('/tmp') -> /tmp` could have run
`confined_root('~/.ssh')` for free, and the answer would have been
`/home/<user>/.ssh`. NB4's own row says "The policy is unchanged; the default breadth
is what moved" — which is exactly the sentence that should have triggered the question.

Two reasons it did not:

1. **The review was scoped to the failure it had measured.** The finding was "a cwd
   outside `$HOME` is refused"; the closure criterion was "that cwd is now accepted".
   A widening has two acceptance criteria, and only the reported one was checked.
2. **The change was documented as a portability fact.** `CHANGELOG.md` records it under
   "Changed" as *"`ops.filesys.roots` ships as `["~", ".", "/tmp"]`, so a clone runs
   wherever it lives"* — true, and silent about authority. A reader of the changelog
   learns the default got wider and not what it now includes, and the three documents
   that state the default state it without consequence. The consequence lived only in
   `confine.py`'s reserved word `~`.

The reusable lesson: **when a security default is widened, the review question is not
"does the reported failure go away" but "what does it now admit" — and the answer
belongs in a document, not in the code.** The inverted check is cheap: for each policy
root, name three sensitive paths under it and run `confined_root()` on them. For `~`
the answer was `.ssh`, `.gnupg`, `.aws`, and it took a hand-written probe to surface it
because every existing test passes a `tmp` root and never asserts on what the shipped
default admits.

---

## 9. Checking this yourself

No service, no token, no network:

```sh
cd <the repo>
PYTHONPATH=service python3 - <<'PY'
from ti.ops.filesys import confine
print("policy roots:", confine.policy_roots())
print("deny roots:  ", confine.deny_roots())
for cwd in ("~", "~/.ssh", "~/.aws", "~/.local/state/ti", "~/mnt", "/etc"):
    try:
        print(f"{cwd:22} -> run root {confine.confined_root(cwd)}")
    except confine.PathRefused as exc:
        print(f"{cwd:22} -> refused: {exc}")
PY
```

To see the effect of a change without touching the shipped file, point
`TI_CONF_OVERLAY` at a fragment — lists replace, they do not concatenate:

```sh
cat > /tmp/deny.json <<'JSON'
{"ops": {"filesys": {"deny_roots": ["~/.ssh"]}}}
JSON

TI_CONF_OVERLAY=/tmp/deny.json PYTHONPATH=service python3 -c \
  'from ti.ops.filesys import confine; print(confine.confined_root("~/.ssh"))'
# -> PathRefused: '/home/<user>/.ssh' is at or under a denied root …
```

And to watch the boundary the way an attacker meets it, with a running service. Note
that the body needs a non-empty `intent` even though an L0 passthrough ignores it —
without one the only answer is `{"status": "error", "error": "empty intent"}`, which
reads like a refusal and is not:

```sh
curl -s -X POST http://127.0.0.1:1113/v1/useTool \
  -H "Content-Type: application/json" -H "X-TI-Token: $TI_HTTP_TOKEN" \
  -d '{"intent":"read the key",
       "digest":{"cwd":"~/.ssh"},
       "params":{"tool":"filesys_read","args":{"path":"id_ed25519"}}}'
```

Run against a service on this box on 2026-11-04, that request returned:

```
HTTP 200
status: "done"    result: <the 399-byte OpenSSH private key, whole>    error: null
hops:   [{"step": "L0", "op": "filesys_read", "conf": 1.0, …}]
```

No guard was bypassed and nothing was misconfigured: the run root was inside a policy
root, the file was inside the run root, and the op read it. That is the behaviour §7
recommends changing, and the reason this document exists.
