"""
Dispense Skill

Drop-scale liquid transfer with a held pipette or dropper. This backs
robomail_Aliyah's PIPETTE_DISPENSE action, whose semantics are "draw AND
release a small volume" — so this skill supports aspirating from a source as
well as dispensing into a target, and a combined transfer.

The bulb is actuated by squeezing it with the gripper. That makes every squeeze
a risk to the grasp itself, which drives most of the hardening here:
  - the squeeze is measured from the width recorded at skill start, and is
    floored so it can never crush or release the pipette
  - the gripper is restored and re-read after every pulse; a pipette that has
    slipped stops the sequence instead of miming drops into the air
  - the tip is placed over the measured rim centre at a measured height, not
    at centroid + PCA dimensions[2]
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base_skill import BaseSkill


class DispenseSkill(BaseSkill):
    """
    Aspirate from and/or dispense into a container with a held pipette.

    Modes:
      "dispense" (default) — squeeze pulses over the target, releasing drops
      "aspirate"           — squeeze, lower into the source, release to draw in
      "transfer"           — aspirate from source_container, then dispense

    Pipeline (dispense):
    1. Record held width; every squeeze is relative to it
    2. Clear the cameras, scan the target, measure the rim
    3. Position the tip above the opening, pipette vertical
    4. Squeeze / release pulses, verifying the grasp between each
    5. Lift clear and confirm the pipette is still held
    """

    name = "dispense"
    required_params = ["target_container"]

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "mode": "dispense",         # "dispense" | "aspirate" | "transfer"
            "num_drops": 3,
            "source_container": None,   # required for aspirate / transfer
            "drop_interval": 0.6,
            "dispense_height": 0.04,    # tip clearance above the rim
            "aspirate_depth": 0.02,     # tip immersion below the source surface
            # How far to squeeze the bulb, in metres of gripper travel.
            # Small: the bulb is compliant and the pipette must not be released.
            "squeeze_amount": 0.003,
            "squeeze_hold": 0.35,       # seconds held squeezed
            # Never let a squeeze take the jaws below this fraction of the
            # width the pipette was held at. Squeezing past the bulb crushes
            # the barrel and the pipette pops out of the jaws.
            "min_width_fraction": 0.55,
            "approach_height": 0.10,
            "lift_height": 0.10,
            "reset_before_scan": True,
            "tool_length": 0.0,         # TCP -> pipette tip offset, metres
            "hover_tol": 0.05,
            "descend_tol": 0.03,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg

        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a pipette first."

        mode = str(params.get("mode", "dispense")).lower()
        if mode not in ("dispense", "aspirate", "transfer"):
            return False, (f"Unknown mode {mode!r}; expected 'dispense', "
                           f"'aspirate' or 'transfer'")

        if mode in ("aspirate", "transfer") and not params.get("source_container"):
            return False, f"mode={mode!r} requires a source_container"

        num_drops = params.get("num_drops", 3)
        if not isinstance(num_drops, int) or num_drops < 1:
            return False, "num_drops must be a positive integer"

        return True, "Preconditions met"

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        mode = str(params["mode"]).lower()
        target = params["target_container"]
        source = params.get("source_container")

        start_width = self.held_width()
        print(f"[Dispense] mode={mode}; holding pipette at "
              f"{start_width * 1000:.1f}mm")

        result: Dict[str, Any] = {
            "mode": mode,
            "held_width_at_start": start_width,
        }

        if mode in ("aspirate", "transfer"):
            ok, detail = self._aspirate(params, source, start_width)
            result.update(detail)
            if not ok:
                return False, result
            if mode == "aspirate":
                print(f"[Dispense] Successfully aspirated from '{source}'")
                return True, result

        ok, detail = self._dispense(params, target, start_width)
        result.update(detail)
        if not ok:
            return False, result

        print(f"[Dispense] Successfully dispensed into '{target}'")
        return True, result

    # ==================== Phases ====================

    def _aspirate(self, params: Dict[str, Any], source: str,
                  start_width: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Draw liquid in: squeeze the bulb in free air, lower the tip into the
        liquid, then release the squeeze so the bulb refills through the tip.

        Squeezing *before* entering the liquid is the whole point — squeeze it
        while submerged and the air is blown into the source instead.
        """
        tool_length = float(params["tool_length"])

        if not self.clear_cameras(params, tag="Dispense"):
            return False, {"error": "Failed to clear the cameras before scanning"}

        self.vision.clear_cache()
        located = self.locate_container(source, force_refresh=True)
        if located is None:
            return False, {"error": f"Cannot locate source '{source}'"}

        ok, msg = self.check_still_holding(start_width, tag="Dispense")
        if not ok:
            return False, {"error": f"Lost the pipette before aspirating: {msg}"}

        rim_center = located["rim_center"]
        surface_z = located["top_z"]
        rotation = self.tool_down_rotation()

        hover = np.array([rim_center[0], rim_center[1],
                          surface_z + float(params["approach_height"]) + tool_length])
        print(f"[Dispense] Hovering above source '{source}' "
              f"at {np.round(hover, 4)}...")
        if not self.goto_pose_rigid(hover, rotation, duration=3.0):
            return False, {"error": "Failed to command hover over the source"}
        arrived, err = self.reached(hover, float(params["hover_tol"]))
        if not arrived:
            return False, {"error": (f"Hover above '{source}' unreachable "
                                     f"(off by {err * 1000:.0f}mm)")}

        # Squeeze in air, above the liquid.
        print(f"[Dispense] Squeezing the bulb before entry...")
        squeezed_width = self._squeeze(params, start_width)
        if squeezed_width is None:
            return False, {"error": "Failed to squeeze the bulb"}

        # Lower the tip into the liquid while still squeezed.
        entry_z = surface_z - float(params["aspirate_depth"]) + tool_length
        entry = np.array([rim_center[0], rim_center[1], entry_z])
        print(f"[Dispense] Lowering the tip to z={entry_z:.4f}...")
        if not self.goto_pose_rigid(entry, rotation, duration=3.0):
            self._release_squeeze(params, start_width)
            return False, {"error": "Failed to command the source entry pose"}
        arrived, err = self.reached(entry, float(params["descend_tol"]))
        if not arrived:
            self._release_squeeze(params, start_width)
            self.goto_pose_rigid(hover, rotation, duration=3.0)
            return False, {"error": (f"Could not reach the liquid in '{source}' "
                                     f"(off by {err * 1000:.0f}mm)")}

        # Release: the bulb refills, drawing liquid up the tip.
        print(f"[Dispense] Releasing the bulb to draw liquid in...")
        if not self._release_squeeze(params, start_width):
            return False, {"error": "Failed to release the bulb after entry"}
        self.wait(float(params["squeeze_hold"]))

        # Lift clear.
        lift = np.array([rim_center[0], rim_center[1],
                         surface_z + float(params["lift_height"]) + tool_length])
        if not self.goto_pose_rigid(lift, rotation, duration=3.0):
            return False, {"error": "Failed to lift the pipette out of the source"}

        ok, msg = self.check_still_holding(start_width, tag="Dispense")
        if not ok:
            return False, {"error": f"Pipette lost while aspirating: {msg}"}

        self.vision.clear_cache()
        return True, {
            "aspirated_from": source,
            "aspirate_depth": float(params["aspirate_depth"]),
            "source_surface_z": surface_z,
            # There is no sensing of how much was actually drawn.
            "volume_measured": False,
        }

    def _dispense(self, params: Dict[str, Any], target: str,
                  start_width: float) -> Tuple[bool, Dict[str, Any]]:
        """Pulse the bulb over the target to release drops."""
        num_drops = int(params["num_drops"])
        tool_length = float(params["tool_length"])

        if not self.clear_cameras(params, tag="Dispense"):
            return False, {"error": "Failed to clear the cameras before scanning"}

        self.vision.clear_cache()
        located = self.locate_container(target, force_refresh=True)
        if located is None:
            return False, {"error": f"Cannot locate target '{target}'"}

        ok, msg = self.check_still_holding(start_width, tag="Dispense")
        if not ok:
            return False, {"error": f"Lost the pipette before dispensing: {msg}"}

        rim_center = located["rim_center"]
        rim_z = located["top_z"]
        rotation = self.tool_down_rotation()

        # The tip sits *above* the rim: dipping it in would contaminate the
        # pipette and wick liquid back out of the target.
        dispense_xyz = np.array([
            rim_center[0], rim_center[1],
            rim_z + float(params["dispense_height"]) + tool_length,
        ])
        print(f"[Dispense] Positioning the tip above '{target}' "
              f"at {np.round(dispense_xyz, 4)} "
              f"({float(params['dispense_height']) * 1000:.0f}mm above the rim)...")
        if not self.goto_pose_rigid(dispense_xyz, rotation, duration=3.0):
            return False, {"error": "Failed to command the dispense pose"}
        arrived, err = self.reached(dispense_xyz, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": (f"Dispense pose above '{target}' unreachable "
                          f"(off by {err * 1000:.0f}mm); drops would land "
                          f"outside the container"),
            }

        print(f"[Dispense] Dispensing {num_drops} drops...")
        drops = 0
        for i in range(num_drops):
            print(f"[Dispense] Drop {i + 1}/{num_drops}")

            if self._squeeze(params, start_width) is None:
                return False, {"error": f"Squeeze failed on drop {i + 1}",
                               "drops_dispensed": drops}
            self.wait(float(params["squeeze_hold"]))

            if not self._release_squeeze(params, start_width):
                return False, {"error": f"Bulb release failed on drop {i + 1}",
                               "drops_dispensed": drops}

            # A pipette that has slipped out mid-sequence must stop the run,
            # not keep pulsing an empty gripper over the cup.
            ok, msg = self.check_still_holding(start_width, tag="Dispense")
            if not ok:
                return False, {
                    "error": f"Pipette lost after drop {i + 1}: {msg}",
                    "drops_dispensed": drops,
                }

            drops += 1
            if i < num_drops - 1:
                self.wait(float(params["drop_interval"]))

        # Lift clear of the container.
        lift = dispense_xyz.copy()
        lift[2] = rim_z + float(params["lift_height"]) + tool_length
        if not self.goto_pose_rigid(lift, rotation, duration=3.0):
            print("[Dispense] Warning: failed to lift clear after dispensing")

        self.vision.clear_cache()
        return True, {
            "target": target,
            "drops_requested": num_drops,
            "drops_dispensed": drops,
            "dispense_position": dispense_xyz.tolist(),
            "rim_z": rim_z,
            # Drop volume is nominal — nothing measures what left the pipette.
            "volume_measured": False,
        }

    # ==================== Bulb actuation ====================

    def _squeeze_target_width(self, params: Dict[str, Any],
                              start_width: float) -> float:
        """
        Jaw opening for a squeeze, floored so the pipette survives.

        Squeezing is the only place a skill deliberately closes on an object it
        wants to keep holding, so the floor matters: past the bulb's travel the
        jaws crush the barrel and the pipette pops out.
        """
        floor = start_width * float(params["min_width_fraction"])
        return max(floor, start_width - float(params["squeeze_amount"]))

    def _squeeze(self, params: Dict[str, Any],
                 start_width: float) -> Optional[float]:
        """Close the jaws onto the bulb. Returns the achieved width, or None."""
        target_width = self._squeeze_target_width(params, start_width)
        # Position mode (grasp=False): the jaws stop at target_width. Force
        # mode would keep squeezing a compliant bulb until it crushed.
        if not self.close_gripper(target_width=target_width, force_limited=False):
            return None
        self.wait(0.2)
        width = self.held_width()
        print(f"[Dispense]   squeezed to {width * 1000:.1f}mm "
              f"(target {target_width * 1000:.1f}mm, "
              f"held {start_width * 1000:.1f}mm)")
        return width

    def _release_squeeze(self, params: Dict[str, Any],
                         start_width: float) -> bool:
        """
        Return the jaws to the width the pipette was held at.

        Deliberately not ``open_gripper``: that opens to 80mm and drops the
        pipette on the bench.
        """
        if not self.close_gripper(target_width=start_width, force_limited=False):
            return False
        self.wait(0.2)
        width = self.held_width()
        # If the jaws did not come back, the bulb is jammed or the pipette has
        # shifted; either way the next pulse will not dispense anything.
        if width < start_width * 0.8:
            print(f"[Dispense]   bulb did not re-expand: {width * 1000:.1f}mm "
                  f"vs held {start_width * 1000:.1f}mm")
            return False
        return True
