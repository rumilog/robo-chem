# Registered objects (exemplar vocabulary)

One `.npz` per custom object: DINOv2 embeddings of reference crops you boxed by
hand. These are the vocabulary for things SAM 3 has no word for — the printed
stirrer, the cranked scoop — and they are cell calibration, same as
`calibration_out/`: small, and meant to be committed.

## Why these exist

SAM 3 is open-vocabulary, not unlimited. `scripts/sweep_prompts.py` spent twenty
phrases on the printed scoop and the best honest hit was `"scoop"` on 2/4
cameras; the stirrer sweep in `diag_out/stirrer_prompts/` is the same story. The
phrases that score *well* (`"white object"`, 0.95 on 4/4) are matching the cups.
No wording fixes this — the concept is not in the model.

So these objects are matched by appearance instead. The grounding service
segments everything in the frame with SAM's automatic mask generator (no prompt,
class-agnostic), embeds each candidate, and returns whichever one looks most like
a reference here. See `perception_service/exemplar_matcher.py`.

## Adding an object

```bash
# 1. put it on the bench, photograph it from every camera
source scripts/env.sh
python scripts/capture_scene.py --label stirrer

# 2. drag a box around it once per camera
perception_env/bin/python scripts/register_object.py --name stirrer \
    --images-dir scene_captures/stirrer_<ts>

# 3. restart the grounding service (it loads exemplars/ at startup)
# 4. check it, and LOOK at the overlays
python scripts/sweep_prompts.py --images-dir scene_captures/stirrer_<ts> \
    --only stirrer --save-overlays diag_out/stirrer_exemplar
```

Register from several captures (`--images-dir` is repeatable) when the tool will
be seen in genuinely different poses — lying on the bench and standing in its
holder are not the same view.

## Two things that silently cost you a match

**Aliases.** `GroundingClient._resolve` rewrites names through
`SAM_PROMPT_ALIASES` *before* the service sees them, and dispatch here is by
name. A skill asking for `"rectangular scoop"` sends `"white plastic tool"`, so
the scoop must be registered with that phrase as an alias or it is never
consulted:

```bash
perception_env/bin/python scripts/register_object.py --name scoop \
    --images-dir scene_captures/new_scoop_<ts> --alias "white plastic tool"
```

`register_object.py` warns when it spots this.

**The model id.** Every file records the encoder that made it. Embeddings from a
different encoder are not comparable to anything the service computes, so a
mismatched file is skipped with a warning rather than scored — re-register if
`EMBED_MODEL_ID` in `exemplar_matcher.py` ever changes.

## Register what you want to GRASP

`ObjectLocalizer.get_object_pose` puts the grasp at the **mean of the fused mask
points**, so the mask defines where the gripper goes. Register the part you mean,
not the assembly it is sitting in.

The stirrer is the white cube; the black cylinder under it is its holder. A first
pass registered "stirrer" as cube + cylinder together, which scored beautifully
(0.95+) and would have driven the gripper into the middle of the *holder*. They
are now two objects — `stirrer` and `stirrer holder` — which also gives
place-back something to aim at, and keeps `stirrer` recognisable once it has been
lifted out and the assembly no longer exists.

A high score means "this is the shape you showed me". It says nothing about
whether that shape was the right thing to show.

## Register every state the object will be seen in

A holder with the tool in it and a holder standing empty are different pictures.
`stirrer_holder.npz` carries both — four views with the stirrer seated (its top
occluded) and four of the empty bore — because the holder gets looked for in
both states: occupied before a pick, empty when deciding where to put the
stirrer back. They are only 0.49-0.83 similar to each other, so one does not
stand in for the other.

The same applies to a tool that will be seen lying on the bench *and* standing
in its holder. `--append` adds views to an existing object; it never replaces
them.

## Multi-part objects

Box the WHOLE tool when you register, even if it is two colours. A two-tone
object is several objects to the mask generator — the stirrer's black body, the
white cube's top and the cube's shaded front face come back as three separate
masks — so no single candidate is ever the whole thing. The matcher glues
neighbouring pieces back together for as long as that improves the match against
your reference, which took the stirrer from 0.64 (body only, head cut off) to
0.95 on cam2. You do not have to do anything for this; it just means the
reference should be the shape you actually want back.

## When something is not found

The point grid is what finds candidates at all, and at the default 32x32 over an
848x480 frame the points land ~26px apart — wider than the scoop's shank. That
produced half a scoop on one camera, and half a scoop legitimately resembles the
stirrer cube more than it resembles a scoop, so the match was refused. The
service retries a missed frame at 48x48 (`--exemplar-retry-points`), which costs
~9s instead of ~4s but only on frames that actually missed.

A per-camera miss is not a failure of the run: localization fuses whatever
cameras did find the object, and three good views beat four with a bad one.

## Reading the numbers

`register_object.py` prints how well the object stands apart: how much its own
views agree, and its best similarity to every other registered object. Own views
at 0.9 and the nearest neighbour at 0.5 is a clean object. Own views at 0.5 and a
neighbour at 0.8 means the crops are mostly table, and no threshold will save it —
register tighter boxes, or make the objects visually different (colour is the
cheap fix). The suggested `--exemplar-thresh` sits between the two.
