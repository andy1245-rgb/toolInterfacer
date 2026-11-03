# demo/

A recording of `bench/demo.py`, and the two things you can do with it.

| file | what it is |
|---|---|
| `toolinterfacer.cast` | an [asciinema](https://asciinema.org) v2 recording — `asciinema play demo/toolinterfacer.cast` |
| `toolinterfacer.svg` | the whole session as one self-contained image; no fonts or CSS to fetch, so it renders anywhere |
| `toolinterfacer.png` | the same at 1200 px wide, for a post or a slide |

## What it shows

Three intents, each answered by **one** `useTool` call, with the hop list printed as the
receipt:

1. **L0** — the caller names the operation. No decision is made, so the response carries
   no confidence at all rather than a fabricated `1.0`.
2. **L1** — a vague intent. The service globs the directory, picks a file, reads it, and
   reports the pick's confidence (0.93) beside the two deterministic hops.
3. **L2** — a compound intent. Eight hops: extract queries → search → pick → four gated
   fetches → assemble a digest, ending in a sourced digest.

The number in the last line of each block is the point: *N hops in 1 round trip*.

## Reproducing the recording

The demo is deterministic offline, so the recording is reproducible on any machine with
Python 3.10+ and nothing installed:

```sh
# what the recording runs
python3 bench/demo.py --pace 1.0

# record it again
asciinema rec --cols 100 --rows 40 --idle-time-limit 2 \
  -c "python3 bench/demo.py --pace 1.0" demo/toolinterfacer.cast

# render the image
python3 bench/cast_to_svg.py demo/toolinterfacer.cast demo/toolinterfacer.svg --cols 112
cairosvg demo/toolinterfacer.svg -o demo/toolinterfacer.png --output-width 1200   # optional
```

`--live` runs the same three scenes against the configured decision model and the real
web/reddit CLIs instead of the recorded answers. The header of every run says which mode
it is in, because a demo that quietly runs against fixtures proves nothing.
