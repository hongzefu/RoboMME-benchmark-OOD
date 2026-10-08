"""``robomme_hard`` 的评估构建器：官方 ``BenchmarkEnvBuilder`` 的子类（0927 计划第一部分 §4.2）。

对外新增 ``dataset="ood"`` 与 ``dataset="hard-verify"`` 两个取值；``xhard1``～``xhard5`` 不是合法的 ``dataset``。

* ``train`` / ``test`` / ``val``：沿用官方父类的元数据逻辑；只把四个 Unmask 任务的 ``train`` 元数据改读
  ``robomme_hard/env_metadata/train``（400 条，E-12）。
* ``ood``：xhard0 12 局仅在开关 ``hard_specs.XHARD0_IN_TEST_HARD`` 打开时在前（默认关，V9 每任务 50 局），再依次读包内 ``env_metadata/ood/<tier>/specs.jsonl``（xhard1→xhard5，
  v8 ``hard-specs/4``，经 ``load_specs_root`` 整根校验），取本任务 ``selected`` 且 ``rollout.status=="ok"`` 的行，
  档内按 ``candidate`` 升序，拼接编为 episode 0..N-1。每格行数对照交付格表 ``EXPECTED_CELLS``（43 格逐格局数）
  断言：(任务, 档) 必须在表内才可有正式局，表内格恰好等于表值，表外格恰好 0 行（xhard5 只含 SwingXtimes、StopCube）。
  规格根覆盖（冒烟／分片等局部根）只读存在的档文件，按各档 header 的 ``delivery_per_cell`` 自洽校验，且须是表的子集。
  只读 ``hard-specs/4``（旧格式 /2、/3 的读写校验已删除）。
* ``hard-verify``：只含 xhard0，即官方 test 元数据里本任务 ``difficulty=="hard"`` 的 12 局（原 episode 3, 7, …, 47），
  编为 episode 0..11；与开关 ``XHARD0_IN_TEST_HARD`` 无关，不读规格根、不接受 ``specs_root``。
  步数上限不由数据集给出：评估入口按数据集传 ``max_steps``（xhard0 用 1300，新值档用 1600）。
* ``make_env_for_episode`` 整段覆写：runtime 四项、seed、difficulty 照抄官方拼法；ood 时在 ``gym.make`` 前加
  ``sampling_config`` 与 ``native_episode_spec``（回注）；包装链与官方逐项相同，但 wrapper 一律绝对导入
  ``robomme_hard`` 的类（``DemonstrationWrapper``、``OraclePlannerDemonstrationWrapper`` 是复制件，其余是借用）。

⚠ 官方父类 ``__init__`` 的 ``_ALLOWED_DATASETS`` 只认 train/test/val 且官方代码不能改：ood／hard-verify 先以
``dataset="test"`` 过父类校验，再把 ``self.dataset`` 改回原值；父类读的 test 元数据取出 xhard0 后随即清空、不再使用。

P2：本子类覆写 ``__init__``、``_resolve_metadata_path``、``resolve_episode``、``get_episode_num``、
``make_env_for_episode``，已由用户 2026-09-27「现在一次批准这两项」（U-3）批准。
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym

from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder as _OfficialBuilder

from . import hard_specs

OOD = "ood"
#: 只含 xhard0（官方 test 的 hard 子集，每任务 12 局）的评估数据集；不读规格根
HARD_VERIFY = "hard-verify"
_ALLOWED_DATASETS = {"train", "test", "val", OOD, HARD_VERIFY}
_HARD_DATASETS = frozenset({OOD, HARD_VERIFY})
_ALLOWED_ACTION_SPACES = {"joint_angle", "ee_pose", "waypoint", "multi_choice"}
HARD_METADATA_ROOT = Path(__file__).resolve().parents[1] / "env_metadata"
#: 这四个任务的 train 元数据在 robomme_hard 里是 400 条，其余任务读官方
HARD_TRAIN_TASKS = frozenset({"ButtonUnmask", "ButtonUnmaskSwap", "VideoUnmask", "VideoUnmaskSwap"})
_RUNTIME_KEYS = ("obs_mode", "control_mode", "render_mode", "reward_mode")


def _override_cells(root: str) -> Dict[tuple, int]:
    """规格根覆盖（非包内）的格表：只看存在的档文件，取 header ``tasks``／``delivery_per_cell``；须全是 /4。"""
    cells: Dict[tuple, int] = {}
    for tier in hard_specs.TIERS:
        path = hard_specs.packaged_specs_path(tier, root)
        if not path.is_file():
            continue  # 局部根（冒烟／分片）只含部分档；与 hard_regression.delivery_index 的跳过口径相同
        with path.open(encoding="utf-8") as stream:
            header = json.loads(stream.readline())
        if header.get("schema") != hard_specs.SCHEMA:
            raise hard_specs.SpecsError(f"{path}：builder 只读 {hard_specs.SCHEMA}（实为 {header.get('schema')}）")
        for task in header["tasks"]:
            cells[(task, tier)] = int(header["delivery_per_cell"][task])
    if not cells:
        raise hard_specs.SpecsError(f"规格根 {root} 下没有任何 {'／'.join(hard_specs.TIERS)} 规格文件")
    return cells


@functools.lru_cache(maxsize=None)
def _root_specs(root: str):
    """每个规格根（包内或覆盖）只读一次：``load_specs_root`` 整根校验（逐档 /4 封套、格表、每格 selected 数、
    跨档 seed 不交）。包内根的格表必须恰为 ``EXPECTED_CELLS``（配额上限也取它，不写死局数）；覆盖根按
    ``_override_cells``，配额上限由 ``load_specs_root`` 按 ``resolve_cell_table`` 取（先 ``EXPECTED_CELLS``，
    覆盖不了再看 ``CELL_TABLES``）。
    返回 ``({tier: (header, rows)}, cells)``，只读使用，不得修改。"""
    if Path(root) == hard_specs.PACKAGED_SPECS_ROOT:
        cells = dict(hard_specs.EXPECTED_CELLS)
        return hard_specs.load_specs_root(Path(root), cells, cell_table=hard_specs.EXPECTED_CELLS), cells
    cells = _override_cells(root)
    return hard_specs.load_specs_root(Path(root), cells), cells


def _xhard0_entries(env_id: str, metadata_index: Dict) -> List[Dict[str, Any]]:
    """xhard0＝官方 test 元数据里本任务 ``difficulty=="hard"`` 的全部记录，按原 episode 升序（v7 方案第二部分 §1.1）。

    ``ood`` 在开关打开时把它前置，``hard-verify`` 只用它（每任务恰 ``XHARD0_PER_TASK`` 局）。

    seed 逐条照抄元数据、运行难度传 ``"hard"``，无 ``sampling_config``、无规格（走官方原生 hard 分支）。
    与 ``scripts/configs/xhard0/xhard0_manifest.json`` 的逐条核对在 XHARD0_IDENTITY 闸门里做（本包不反向依赖 scripts/）。
    """
    hard = sorted(
        (record for (task, _ep), record in metadata_index.items() if task == env_id and record.get("difficulty") == "hard"),
        key=lambda record: int(record["episode"]),
    )
    episodes = tuple(int(record["episode"]) for record in hard)
    seeds = [int(record["seed"]) for record in hard]
    if episodes != hard_specs.XHARD0_EPISODES or len(set(seeds)) != len(seeds):
        raise ValueError(f"ood {env_id}@xhard0：官方 test hard 子集应为原 episode {hard_specs.XHARD0_EPISODES}、seed 唯一，"
                         f"实际 episode {episodes}")
    return [{
        "tier": hard_specs.XHARD0,
        "row": {"seed": seed, "candidate": None, "source_episode": episode, "spec_sha256": None, "spec": None},
        "sampling_config": None,
        "runtime": dict(hard_specs.RUNTIME),
        "recovery_rule": None,
    } for episode, seed in zip(episodes, seeds)]


def _ood_entries(env_id: str, xhard0: List[Dict[str, Any]], root: str) -> List[Dict[str, Any]]:
    if env_id not in hard_specs.ALL_TASKS:
        raise ValueError(f"ood 不含环境 {env_id!r}")
    entries: List[Dict[str, Any]] = list(xhard0)
    specs, cells = _root_specs(root)
    for tier in hard_specs.TIERS:
        expected = cells.get((env_id, tier), 0)
        if expected and (env_id, tier) not in hard_specs.EXPECTED_CELLS:
            raise ValueError(f"ood ({env_id}, {tier}) 不在交付格表 EXPECTED_CELLS 内")
        if tier not in specs:
            continue  # 覆盖根缺该档文件：本档不发局（cells 里也没有该档）
        header, rows = specs[tier]
        chosen = sorted((row for row in rows if row["task"] == env_id and hard_specs.delivered(row)),
                        key=lambda row: int(row["candidate"]))
        if len(chosen) != expected:
            raise ValueError(f"ood {env_id}@{tier} 正式局 {len(chosen)} 行，交付格表要求恰好 {expected} 行")
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
    """官方构建器 + ``dataset="ood"``／``"hard-verify"``；其余取值行为与官方相同（四个 Unmask 任务的 train 元数据除外）。"""

    def __init__(
        self,
        env_id: str,
        dataset: str = "test",
        action_space: str = "joint_angle",
        gui_render: bool = False,
        override_metadata_path: Optional[Union[str, Path]] = None,
        max_steps: int = 10000,
        specs_root: Optional[Union[str, Path]] = None,
    ):
        if dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Allowed datasets: {sorted(_ALLOWED_DATASETS)}")
        if dataset in _HARD_DATASETS and override_metadata_path is not None:
            raise ValueError(f"{dataset} 的 xhard0 只读官方 test 元数据，不接受 override_metadata_path")
        if dataset == HARD_VERIFY:
            if specs_root is not None:
                raise ValueError("hard-verify 只含 xhard0、不读规格根，不接受 specs_root")
            if env_id not in hard_specs.ALL_TASKS:
                raise ValueError(f"hard-verify 不含环境 {env_id!r}")
        self._episode_map: Optional[Dict[int, Dict[str, Any]]] = None
        self._specs_root: Optional[Path] = None
        super().__init__(
            env_id,
            dataset="test" if dataset in _HARD_DATASETS else dataset,
            action_space=action_space,
            gui_render=gui_render,
            override_metadata_path=override_metadata_path,
            max_steps=max_steps,
        )
        self.dataset = dataset
        if dataset == OOD:
            # 父类按 dataset="test" 读了官方 test 元数据：先取出 xhard0（hard 子集）再清空（v7 §1.1）
            xhard0 = _xhard0_entries(env_id, self.metadata_index) if hard_specs.XHARD0_IN_TEST_HARD else []
            self.metadata_index = {}
            root = hard_specs.specs_root(specs_root)
            self._specs_root = None if root == hard_specs.PACKAGED_SPECS_ROOT else root
            self._episode_map = dict(enumerate(_ood_entries(env_id, xhard0, str(root))))
        elif dataset == HARD_VERIFY:
            # 只取官方 test 元数据的 hard 子集（12 局），随即清空元数据；不读规格根，与 XHARD0_IN_TEST_HARD 无关
            self._episode_map = dict(enumerate(_xhard0_entries(env_id, self.metadata_index)))
            self.metadata_index = {}

    # ── 旧 V4 快照的薄包装（episode 号＝候选序号）；新代码一律用 dataset="ood" ──
    @classmethod
    def from_v4_specs(
        cls,
        env_id: str,
        header: Dict[str, object],
        specs_by_identity: Dict[str, Dict[str, object]],
        action_space: str = "joint_angle",
        gui_render: bool = False,
        max_steps: int = 10000,
    ) -> "BenchmarkEnvBuilder":
        if action_space not in _ALLOWED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space '{action_space}'.")
        builder = cls(env_id, dataset="test", action_space=action_space, gui_render=gui_render, max_steps=max_steps)
        builder.dataset = "v4-specs"
        builder.metadata_index = {}
        runtime = dict(header.get("runtime") or {})
        expected = dict(hard_specs.RUNTIME, render_mode=builder.render_mode)
        if runtime != expected:
            raise ValueError(f"V4 快照 runtime 与本构建器参数不符：快照 {runtime}，构建器 {expected}")
        if env_id not in header.get("sampling_config", {}):
            raise ValueError(f"V4 快照不含环境 {env_id}")
        builder._episode_map = {}
        for key, row in specs_by_identity.items():
            if key.split("/")[0] != env_id:
                continue
            builder._episode_map[int(row["episode"])] = {
                "tier": row["difficulty"],
                "row": {**row, "candidate": int(row["episode"]), "tier": row["difficulty"]},
                "sampling_config": header["sampling_config"][env_id],
                "runtime": runtime,
                "recovery_rule": header.get("recovery_rule"),
            }
        return builder

    def v4_episodes(self) -> List[int]:
        return sorted(self._episode_map) if self._episode_map is not None else []

    # ── 官方成员的覆写 ─────────────────────────────────────────────────────
    def _resolve_metadata_path(self) -> str:
        if self.override_metadata_path is None and self.dataset == "train" and self.env_id in HARD_TRAIN_TASKS:
            return str(HARD_METADATA_ROOT / "train" / f"record_dataset_{self.env_id}_metadata.json")
        return super()._resolve_metadata_path()

    def _entry(self, episode: int) -> Dict[str, Any]:
        entry = self._episode_map.get(int(episode))
        if entry is None:
            raise KeyError(f"{self.env_id} 在 {self.dataset} 里没有 episode {episode}（共 {len(self._episode_map)} 局）")
        return entry

    def resolve_episode(self, episode: int):
        """返回 ``(seed, difficulty)``，与官方二元组同形；ood／hard-verify 下 difficulty 就是档位（xhard0..5）。"""
        if self._episode_map is None:
            return super().resolve_episode(episode)
        entry = self._entry(episode)
        return int(entry["row"]["seed"]), entry["tier"]

    def resolve_identity(self, episode: int) -> Dict[str, Any]:
        """只读：本局身份 ``{episode, tier, candidate, seed, spec_sha256, source_run}``（官方二元 resolve_episode 不动）。

        xhard0 局（ood 前置部分或 hard-verify 全部）另带 ``source_dataset="test"``、``source_episode``（官方原 episode），
        ``candidate``／``spec_sha256``／``source_run`` 为 None；hard-verify 不读规格根，故永不带 ``specs_root``。"""
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
        if self._specs_root is not None:
            identity["specs_root"] = str(self._specs_root)
        return identity

    def get_episode_num(self) -> int:
        if self._episode_map is None:
            return super().get_episode_num()
        return len(self._episode_map)

    def _hard_env_kwargs(self, episode_idx: int) -> Dict[str, Any]:
        """ood／hard-verify 局在 ``gym.make`` 前追加的参数：xhard0 只有 seed 与 difficulty="hard"，新值档加回注参数。"""
        entry = self._entry(episode_idx)
        if entry["tier"] == hard_specs.XHARD0:
            # xhard0 走官方原生 hard 分支：只有 seed 与 difficulty="hard"，无 sampling_config、无规格（R2、R9）
            return {"seed": int(entry["row"]["seed"]), "difficulty": "hard"}
        runtime = dict(entry["runtime"])
        mine = {"obs_mode": "rgb+depth+segmentation", "control_mode": "pd_joint_pos",
                "render_mode": self.render_mode, "reward_mode": "dense"}
        for key in _RUNTIME_KEYS:
            if key != "render_mode" and runtime.get(key) != mine[key]:
                raise ValueError(f"规格 runtime 与本构建器参数不符：{key} 规格 {runtime.get(key)!r}，构建器 {mine[key]!r}")
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
        """与官方同名方法逐项同构；wrapper 取 robomme_hard 的类，ood 新值档加回注参数。

        hard-verify（全为 xhard0）与官方 test 的 hard 局起法相同：只传 seed 与 difficulty="hard"。
        ``max_steps`` 不随数据集自动取值：调用方不传时退回构造参数 ``max_steps``（xhard0 评估用 1300）。"""
        from robomme_hard.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper

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
            from robomme_hard.env_record_wrapper.EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper

            env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")
        elif self.action_space == "waypoint":
            from robomme_hard.env_record_wrapper.MultiStepDemonstrationWrapper import MultiStepDemonstrationWrapper

            env = MultiStepDemonstrationWrapper(env, gui_render=self.gui_render, vis=self.gui_render)
        elif self.action_space == "multi_choice":
            from robomme_hard.env_record_wrapper.OraclePlannerDemonstrationWrapper import (
                OraclePlannerDemonstrationWrapper,
            )

            env = OraclePlannerDemonstrationWrapper(env, env_id=self.env_id, gui_render=self.gui_render)

        from robomme_hard.env_record_wrapper.FailAwareWrapper import FailAwareWrapper

        env = FailAwareWrapper(env)
        return env
