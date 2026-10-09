"""Evaluation builder for ``robomme_ood``: a subclass of upstream ``BenchmarkEnvBuilder`` (0927 plan, part one, sec. 4.2).

Accepts only two evaluation datasets: ``dataset="ood"`` (default) and ``dataset="hard-verify"``; upstream ``train`` / ``test`` / ``val``
and tier names such as ``xhard1``-``xhard5`` all raise ``ValueError`` (for upstream behavior use the upstream ``robomme`` builder directly).

* ``ood``: only the five new-value tiers; reads the in-package ``env_metadata/ood/<tier>/specs.jsonl`` in order (xhard1->xhard5,
  ``hard-specs/4``, whole root validated by ``load_specs_root``), takes this task's rows that are ``selected`` with
  ``rollout.status=="ok"``, sorted by ``candidate`` within each tier, concatenated as episodes 0..N-1 (16 tasks x 50 episodes = 800).
  Row counts per cell are asserted against the delivery cell table ``EXPECTED_CELLS`` (per-cell episode counts for 43 cells):
  cells in the table equal the table value exactly, cells outside it have exactly 0 rows (xhard5 contains only SwingXtimes,
  StopCube). Reads only in-package specs; no external specs root is accepted.
* ``hard-verify``: only xhard0, i.e. the 12 episodes of this task with ``difficulty=="hard"`` in the upstream test metadata
  (original episodes 3, 7, ..., 47), numbered as episodes 0..11 (16 tasks x 12 episodes = 192); no specs root is read.
* The step limit is not provided by the dataset: the evaluation entry ``scripts/evaluation_ood.py`` passes ``max_steps`` per dataset
  (``hard-verify`` 1300, ``ood`` 1800). The V9 delivery set was filtered at 1600; 1800 only relaxes the cap and does not change
  delivered episodes.
* ``make_env_for_episode`` is overridden as a whole: the four runtime items, seed and difficulty follow upstream's construction;
  for ood, ``sampling_config`` and ``native_episode_spec`` (injection) are added before ``gym.make``; the wrapper chain matches
  upstream item by item, but wrappers are always imported absolutely from ``robomme_ood`` (``DemonstrationWrapper`` and
  ``OraclePlannerDemonstrationWrapper`` are copies, the rest are borrowed).

Warning: upstream parent ``__init__``'s ``_ALLOWED_DATASETS`` only accepts train/test/val and upstream code must not change: ood/hard-verify
first pass parent validation with ``dataset="test"``, then ``self.dataset`` is restored; the test metadata read by the parent is
cleared right after extracting xhard0 and never used again.

P2: this subclass overrides ``__init__``, ``resolve_episode``, ``get_episode_num``, ``make_env_for_episode``,
approved by the user on 2026-09-27 ("approve these two items at once now", U-3).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym

from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder as _OfficialBuilder

from . import hard_specs

OOD = "ood"
#: Evaluation dataset containing only xhard0 (hard subset of upstream test, 12 episodes per task); no specs root is read
HARD_VERIFY = "hard-verify"
#: This builder accepts only these two datasets; upstream train/test/val and tier names are always rejected
_ALLOWED_DATASETS = frozenset({OOD, HARD_VERIFY})
_RUNTIME_KEYS = ("obs_mode", "control_mode", "render_mode", "reward_mode")


@functools.lru_cache(maxsize=None)
def _packaged_specs():
    """Read the in-package specs root once: whole-root validation by ``load_specs_root`` (per-tier /4 envelope, cell table,
    selected count per cell, no seed overlap across tiers); both cell table and quota cap come from ``EXPECTED_CELLS``.
    Returns ``{tier: (header, rows)}``, read-only; must not be modified."""
    return hard_specs.load_specs_root(hard_specs.PACKAGED_SPECS_ROOT, dict(hard_specs.EXPECTED_CELLS),
                                      cell_table=hard_specs.EXPECTED_CELLS)


def _xhard0_entries(env_id: str, metadata_index: Dict) -> List[Dict[str, Any]]:
    """xhard0 = all records of this task with ``difficulty=="hard"`` in the upstream test metadata, ascending by original episode (v7 plan, part two, sec. 1.1).

    Used only by ``hard-verify`` (exactly ``XHARD0_PER_TASK`` episodes per task); ``ood`` does not contain xhard0.

    Seeds are copied from the metadata entry by entry, runtime difficulty is ``"hard"``, no ``sampling_config`` and no spec (takes the upstream native hard branch).
    """
    hard = sorted(
        (record for (task, _ep), record in metadata_index.items() if task == env_id and record.get("difficulty") == "hard"),
        key=lambda record: int(record["episode"]),
    )
    episodes = tuple(int(record["episode"]) for record in hard)
    seeds = [int(record["seed"]) for record in hard]
    if episodes != hard_specs.XHARD0_EPISODES or len(set(seeds)) != len(seeds):
        raise ValueError(f"hard-verify {env_id}@xhard0: upstream test hard subset should be original episodes {hard_specs.XHARD0_EPISODES} with unique seeds, "
                         f"got episodes {episodes}")
    return [{
        "tier": hard_specs.XHARD0,
        "row": {"seed": seed, "candidate": None, "source_episode": episode, "spec_sha256": None, "spec": None},
        "sampling_config": None,
        "runtime": dict(hard_specs.RUNTIME),
        "recovery_rule": None,
    } for episode, seed in zip(episodes, seeds)]


def _ood_entries(env_id: str, xhard0: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-episode entries for ood; callers always pass ``xhard0=[]`` (ood only ever has xhard1-5)."""
    if env_id not in hard_specs.ALL_TASKS:
        raise ValueError(f"ood does not contain environment {env_id!r}")
    entries: List[Dict[str, Any]] = list(xhard0)
    specs = _packaged_specs()
    for tier in hard_specs.TIERS:
        expected = hard_specs.EXPECTED_CELLS.get((env_id, tier), 0)
        header, rows = specs[tier]
        chosen = sorted((row for row in rows if row["task"] == env_id and hard_specs.delivered(row)),
                        key=lambda row: int(row["candidate"]))
        if len(chosen) != expected:
            raise ValueError(f"ood {env_id}@{tier} has {len(chosen)} official-episode rows; the delivery cell table requires exactly {expected} rows")
        for row in chosen:
            entries.append({
                "tier": tier,
                "row": row,
                "sampling_config": header["sampling_config"][env_id],
                "runtime": header["runtime"],
                "recovery_rule": header["recovery_rule"],
            })
    return entries


class BenchmarkEnvBuilder(_OfficialBuilder):
    """Subclass of the upstream builder; accepts only ``dataset="ood"`` (default) / ``"hard-verify"``."""

    def __init__(
        self,
        env_id: str,
        dataset: str = OOD,
        action_space: str = "joint_angle",
        gui_render: bool = False,
        override_metadata_path: Optional[Union[str, Path]] = None,
        max_steps: int = 10000,
    ):
        if dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Allowed datasets: {sorted(_ALLOWED_DATASETS)}")
        if override_metadata_path is not None:
            raise ValueError(f"{dataset} reads only upstream test metadata (xhard0) and in-package specs; override_metadata_path is not accepted")
        if env_id not in hard_specs.ALL_TASKS:
            raise ValueError(f"{dataset} does not contain environment {env_id!r}")
        self._episode_map: Optional[Dict[int, Dict[str, Any]]] = None
        # Upstream parent accepts only train/test/val: pass parent validation as test first, then restore the original value
        super().__init__(
            env_id,
            dataset="test",
            action_space=action_space,
            gui_render=gui_render,
            override_metadata_path=override_metadata_path,
            max_steps=max_steps,
        )
        self.dataset = dataset
        if dataset == OOD:
            # ood does not use the upstream test metadata the parent read with dataset="test"; clear it. ood has only the five new-value tiers
            self.metadata_index = {}
            self._episode_map = dict(enumerate(_ood_entries(env_id, [])))
        else:
            # hard-verify: take only the hard subset (12 episodes) of upstream test metadata, then clear the metadata; no specs root is read
            self._episode_map = dict(enumerate(_xhard0_entries(env_id, self.metadata_index)))
            self.metadata_index = {}

    # ── Overrides of upstream members ──────────────────────────────────────
    def _entry(self, episode: int) -> Dict[str, Any]:
        entry = self._episode_map.get(int(episode))
        if entry is None:
            raise KeyError(f"{self.env_id} in {self.dataset} has no episode {episode} ({len(self._episode_map)} episodes total)")
        return entry

    def resolve_episode(self, episode: int):
        """Returns ``(seed, difficulty)``, same shape as upstream's pair; under ood/hard-verify difficulty is the tier (xhard0..5)."""
        if self._episode_map is None:
            return super().resolve_episode(episode)
        entry = self._entry(episode)
        return int(entry["row"]["seed"]), entry["tier"]

    def resolve_identity(self, episode: int) -> Dict[str, Any]:
        """Read-only: this episode's identity ``{episode, tier, candidate, seed, spec_sha256, source_run}`` (upstream pair-returning resolve_episode unchanged).

        xhard0 episodes (all of hard-verify) also carry ``source_dataset="test"`` and ``source_episode`` (upstream original episode);
        ``candidate`` / ``spec_sha256`` / ``source_run`` are None."""
        if self._episode_map is None:
            seed, difficulty = super().resolve_episode(episode)
            return {"episode": int(episode), "tier": difficulty, "candidate": None, "seed": seed,
                    "spec_sha256": None, "source_run": None}
        entry = self._entry(episode)
        row = entry["row"]
        if entry["tier"] == hard_specs.XHARD0:
            identity = {"episode": int(episode), "tier": hard_specs.XHARD0, "candidate": None, "seed": int(row["seed"]),
                        "source_dataset": "test", "source_episode": int(row["source_episode"]),
                        "spec_sha256": None, "source_run": None}
        else:
            identity = {
                "episode": int(episode),
                "tier": entry["tier"],
                "candidate": int(row["candidate"]),
                "seed": int(row["seed"]),
                "spec_sha256": row["spec_sha256"],
                "source_run": (row.get("rollout") or {}).get("source_run"),
            }
            if "layout_parent" in row:
                identity["layout_parent"] = row["layout_parent"]
        return identity

    def get_episode_num(self) -> int:
        if self._episode_map is None:
            return super().get_episode_num()
        return len(self._episode_map)

    def _hard_env_kwargs(self, episode_idx: int) -> Dict[str, Any]:
        """Extra arguments added before ``gym.make`` for ood/hard-verify episodes: xhard0 has only seed and difficulty="hard"; new-value tiers add injection arguments."""
        entry = self._entry(episode_idx)
        if entry["tier"] == hard_specs.XHARD0:
            # xhard0 takes the upstream native hard branch: only seed and difficulty="hard", no sampling_config, no spec (R2, R9)
            return {"seed": int(entry["row"]["seed"]), "difficulty": "hard"}
        runtime = dict(entry["runtime"])
        mine = {"obs_mode": "rgb+depth+segmentation", "control_mode": "pd_joint_pos",
                "render_mode": self.render_mode, "reward_mode": "dense"}
        for key in _RUNTIME_KEYS:
            if key != "render_mode" and runtime.get(key) != mine[key]:
                raise ValueError(f"spec runtime does not match this builder's arguments: {key} spec {runtime.get(key)!r}, builder {mine[key]!r}")
        return {
            "seed": int(entry["row"]["seed"]),
            "difficulty": entry["tier"],
            "sampling_config": entry["sampling_config"],
            "native_episode_spec": entry["row"]["spec"],
        }

    def make_env_for_episode(
        self,
        episode_idx: int,
        max_steps: Optional[int] = None,
        include_maniskill_obs: bool = False,
        include_front_depth: bool = False,
        include_wrist_depth: bool = False,
        include_front_camera_extrinsic: bool = False,
        include_wrist_camera_extrinsic: bool = False,
        include_available_multi_choices: bool = False,
        include_front_camera_intrinsic: bool = False,
        include_wrist_camera_intrinsic: bool = False,
    ):
        """Structurally identical to the same-named upstream method item by item; wrappers come from robomme_ood, and ood new-value tiers add injection arguments.

        hard-verify (all xhard0) starts episodes the same way as upstream test hard episodes: only seed and difficulty="hard" are passed.
        ``max_steps`` is not derived from the dataset automatically: if the caller omits it, the constructor argument ``max_steps`` is used."""
        from robomme_ood.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

        max_steps_without_demo = (
            max_steps + 2 if max_steps is not None else self.max_steps_without_demonstration
        )

        seed, difficulty_hint = self.resolve_episode(episode_idx)
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode=self.render_mode,
            reward_mode="dense",
        )
        if seed is not None:
            env_kwargs["seed"] = seed
        if difficulty_hint:
            env_kwargs["difficulty"] = difficulty_hint
        if self._episode_map is not None:
            env_kwargs.update(self._hard_env_kwargs(episode_idx))

        env = gym.make(self.env_id, **env_kwargs)
        force_front_camera_params = self.action_space == "multi_choice"
        include_front_camera_extrinsic_effective = (
            include_front_camera_extrinsic or force_front_camera_params
        )
        include_front_camera_intrinsic_effective = (
            include_front_camera_intrinsic or force_front_camera_params
        )
        env = DemonstrationWrapper(
            env,
            max_steps_without_demonstration=max_steps_without_demo,
            gui_render=self.gui_render,
            include_maniskill_obs=include_maniskill_obs,
            include_front_depth=include_front_depth,
            include_wrist_depth=include_wrist_depth,
            include_front_camera_extrinsic=include_front_camera_extrinsic_effective,
            include_wrist_camera_extrinsic=include_wrist_camera_extrinsic,
            include_available_multi_choices=include_available_multi_choices,
            include_front_camera_intrinsic=include_front_camera_intrinsic_effective,
            include_wrist_camera_intrinsic=include_wrist_camera_intrinsic,
        )
        if self.action_space == "joint_angle":
            pass
        elif self.action_space == "ee_pose":
            from robomme_ood.env_record_wrapper.EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper

            env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")
        elif self.action_space == "waypoint":
            from robomme_ood.env_record_wrapper.MultiStepDemonstrationWrapper import MultiStepDemonstrationWrapper

            env = MultiStepDemonstrationWrapper(env, gui_render=self.gui_render, vis=self.gui_render)
        elif self.action_space == "multi_choice":
            from robomme_ood.env_record_wrapper.OraclePlannerDemonstrationWrapper import (
                OraclePlannerDemonstrationWrapper,
            )

            env = OraclePlannerDemonstrationWrapper(env, env_id=self.env_id, gui_render=self.gui_render)

        from robomme_ood.env_record_wrapper.FailAwareWrapper import FailAwareWrapper

        env = FailAwareWrapper(env)
        return env
