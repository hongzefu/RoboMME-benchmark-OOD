"""每局规格的只读导出与原值回注（newtaskRelease-v3 步 4）。

方案第三节／8.2 规定的两件事，本模块用同一个对象承担：

* **导出（C 路）**：在**原调用点**把每个取值点的结果只读记下来，不多抽、不少抽、
  不改变任何取值，最后封存成 ``episode_spec``。
* **回注（D 路）**：同一调用点照常执行原抽样（随机流不漂移，红线 R8），但**真正用于建场景
  的值来自冻结规格**；原抽样结果只作兼容核验。

这一点是 G4 的核心判据：方案明确拒绝「重抽出相同值却绕过规格」的实现——本模块通过
``value()`` 返回冻结值（而不是返回抽样值）来结构性地保证规格确实被消费。

与既有注入通道的关系：四个早接了 ``episode_spec`` 的环境里，那条「传了规格就跳过抽样」的
分支属于旧注入模式，按红线 R9 原样保留、不复用为 D 路；本模块只挂在原随机分支上，
由新的 ``native_episode_spec`` 开关驱动。

V6 新值模式：xhard1～xhard4 导出的规格标 ``native-newvalue/2``，
原三档仍标 ``native-parity/1``；两类**不许互喂**（回注时 kind 与本局难度不符即拒绝）。
新值规格回注时的每条不等都要尽量归因到某个 ``decision`` 键（``value(..., decision_key=...)``），
归不了因的才算 RNG 漂移；原值规格仍要求零不等。

V7 共用布局的「派生（derive 信封）」与「分层回注（``native-layered/3``）」两种模式已于维护计划阶段 1b（W4）删除；
现行只有导出与回注两种模式，新值族难度只接受 ``native-newvalue/2`` 全量规格。
"""

from __future__ import annotations

import copy
from typing import Any

SPEC_KIND = "native-parity/1"
# V6 新值档规格版本。
SPEC_KIND_NEWVALUE = "native-newvalue/2"
SPEC_KINDS = (SPEC_KIND, SPEC_KIND_NEWVALUE)


def spec_kind_for(difficulty: str | None) -> str:
    """按本局难度决定规格类别：新值族档走 V6 规格类别，其余（含不传）一律原值类别。"""
    from .difficulty import is_newvalue_difficulty

    if is_newvalue_difficulty(difficulty):
        return SPEC_KIND_NEWVALUE
    return SPEC_KIND


class EpisodeSpecError(ValueError):
    """规格的形态、版本或身份与本局不符。"""


def _set_path(tree: dict, path: str, value: Any) -> None:
    node = tree
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _get_path(tree: dict, path: str):
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise EpisodeSpecError(f"规格缺少取值点 {path}")
        node = node[part]
    return node


def _plain(value: Any):
    """把张量／numpy 标量转成可 JSON 化的原生对象，不改数值。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


class SpecRecorder:
    """一个环境实例一份；``spec=None`` 为导出模式，否则为原值回注模式。"""

    def __init__(self, spec: dict | None, task: str, identity: dict | None = None,
                 difficulty: str | None = None):
        self.task = task
        self.identity = dict(identity or {})
        self.mismatches: list[dict] = []
        self.trace: list[dict] = []
        # 本局规格类别由难度决定（spec_kind_for）；不传 difficulty 时与 V3 行为逐字一致。
        self.spec_kind = spec_kind_for(difficulty)
        self.difficulty = difficulty
        if spec is None:
            self.mode = "export"
            self._frozen: dict = {}
            self._document: dict = {
                "spec_kind": self.spec_kind,
                "task": task,
                "identity": self.identity,
            }
        else:
            # 新值族难度只接受 V6 全量规格；原值难度只接受 native-parity/1（原值与新值不许互喂）
            accepted = (self.spec_kind,)
            if not isinstance(spec, dict) or spec.get("spec_kind") not in accepted:
                raise EpisodeSpecError(
                    f"native_episode_spec 需要 spec_kind∈{accepted}"
                    f"（收到 {spec.get('spec_kind') if isinstance(spec, dict) else type(spec).__name__}；"
                    "原值与新值规格不许互喂）"
                )
            if spec.get("task") != task:
                raise EpisodeSpecError(f"规格属于 {spec.get('task')}，不能用于 {task}")
            self.mode = "replay"
            self._frozen = copy.deepcopy(spec)
            self._document = copy.deepcopy(spec)

    # ------------------------------------------------------------------
    @property
    def replaying(self) -> bool:
        return self.mode == "replay"

    @property
    def newvalue(self) -> bool:
        return self.spec_kind == SPEC_KIND_NEWVALUE

    def value(self, path: str, drawn: Any, decision_key: str | None = None):
        """原调用点：导出模式返回抽样值并记录；回注模式返回冻结值并把抽样值记为兼容核验。

        ⚠ 回注模式**一定**返回冻结值——即便原抽样恰好抽出同样的数，也不允许用抽样值，
        否则就成了方案点名拒绝的「重抽相同却绕过规格」。

        ``decision_key``：新值模式下该取值点受哪个 ``decision`` 键控制（如 ``number_range.xhard``）；
        回注不等时记进 mismatch 供归因，原值模式忽略。
        """
        plain = _plain(drawn)
        if self.mode == "export":
            _set_path(self._document, path, plain)
            self.trace.append({"path": path, "drawn": plain, "source": "draw"})
            return drawn
        frozen = _get_path(self._frozen, path)
        self.trace.append({"path": path, "drawn": plain, "frozen": frozen, "source": "spec"})
        if plain != frozen:
            # 兼容核验不通过只记录，不改变「用冻结值」这一事实；上层据此判 RNG 是否漂移。
            self.mismatches.append(self._mismatch(path, plain, frozen, decision_key))
        return frozen

    def _mismatch(self, path, drawn, frozen, decision_key):
        entry = {"path": path, "drawn": drawn, "frozen": frozen}
        if self.newvalue:
            # 新值模式才带归因字段；原值模式的 mismatch 形态与 V3 逐字一致。
            entry["decision_key"] = decision_key
        return entry

    def unattributed_mismatches(self) -> list[dict]:
        """归不了因的不等：原值模式下是全部 mismatch，新值模式下是没有 decision_key 的那些。"""
        if not self.newvalue:
            return list(self.mismatches)
        return [item for item in self.mismatches if not item.get("decision_key")]

    def record(self, path: str, value: Any) -> None:
        """只读记录派生量或运行观测；回注模式下同样核验。

        与 :meth:`value` 的区别只在于「不替换取值」——它同样计入 trace，因为这个路径
        确实在本局被访问过；否则 SPEC_BINDING 会把它误判成「有记录却没被消费」。
        """
        plain = _plain(value)
        self.trace.append({"path": path, "value": plain, "source": "record"})
        if self.mode == "export":
            _set_path(self._document, path, plain)
            return
        try:
            frozen = _get_path(self._frozen, path)
        except EpisodeSpecError:
            _set_path(self._document, path, plain)
            return
        if plain != frozen:
            self.mismatches.append(self._mismatch(path, plain, frozen, None))

    def leaf_paths(self) -> list[str]:
        """规格里全部取值点路径（回注模式用于算「有记录却没被消费」的 unused）。"""
        out: list[str] = []

        def walk(node, prefix):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}.{key}" if prefix else key)
            else:
                out.append(prefix)

        for section in ("layout", "objects", "actions", "initializations"):
            if section in self._frozen:
                walk(self._frozen[section], section)
        return out

    def consumed_paths(self) -> list[str]:
        """本局真正经 ``value()``／``record()`` 消费过的取值点。"""
        return [item["path"] for item in self.trace]

    def to_dict(self) -> dict:
        document = copy.deepcopy(self._document)
        document.setdefault("spec_kind", self.spec_kind)
        document["task"] = self.task
        document["identity"] = self.identity
        document["provenance"] = {
            "mode": self.mode,
            "value_points": len([item for item in self.trace if item["source"] in ("draw", "spec")]),
            "mismatches": len(self.mismatches),
        }
        if self.newvalue:
            document["provenance"]["unattributed_mismatches"] = len(self.unattributed_mismatches())
        return document
