# Working agreement

## Read `progress.md` first, update it last

**Before starting any work in this repo, read [`progress.md`](progress.md).**
It carries the known-good commands and parameters, what is already validated on
the real cell, and the failures that cost real debugging time. Much of it is not
recoverable from the code — `--workspace-min … -0.13` for the spoon, why
`"plastic beaker"` beats `"beaker"`, why a whole-tool exemplar drove the gripper
into the holder. Reading it first avoids rediscovering things this project has
already paid for.

**After finishing, update `progress.md` in the same turn.** Do not leave it for
the user to write up. Update `Last updated:` at the top while you are there.

This applies to every task unless the user says otherwise. A one-line answer or
a question that changes nothing needs no entry; anything that changes behaviour,
parameters, or how a part of the system is meant to be used does.

### What is worth writing

- The validated command or parameter set, verbatim enough to paste
- What you **measured**, with numbers and the capture or run it came from —
  "0.95-1.00 on all four cameras" is useful, "works well" is not
- What you tried that did **not** work, and why. This is the highest-value part
  of the file and the easiest to skip. The next session will otherwise try the
  same thing.
- Anything a reader could not infer from the code: hardware quirks, why a
  threshold is the value it is, which of two plausible approaches lost and why

Keep it in the voice of the surrounding sections: specific, dated, and honest
about what is bench-tested versus what merely runs.
