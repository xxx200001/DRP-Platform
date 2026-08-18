"""
检验指标归一化词典引擎（规范 4.1：OCR + 结构化引擎，第一环）。

规范原话："自建检验指标归一化词典，统一所有别名、缩写、大小写。"

【本模块与 reference.resolve_alias 的分工】
reference.ReferenceRegistry.resolve_alias 只做【精确】归一化匹配（去空格、
全角半角、大小写），那是给"数据源本身干净、只是写法不统一"的对接场景用的。
OCR 场景的输入还会带【字符级识别错误】：0 认成 O、1 认成 l、5 认成 S、
"谷丙转氨酶"认成"谷丙转氨梅"。本模块在精确匹配之上加两层容错：

  第 1 层  精确匹配        —— 复用 registry.resolve_alias，置信度 1.0
  第 2 层  OCR 混淆字符折叠 —— 把易混字符映射到同一等价类后再匹配，置信度 0.95
  第 3 层  单字符编辑距离   —— 距离 ≤1 且候选唯一才命中，置信度 0.85

【三条安全红线 —— 为什么容错必须"宁可漏配，不可错配"】
规范 4.1 说"上游错一个数据，下游模型全错"。指标名错配是最恶性的上游错误：
把"血钾"配成"血钠"，数值、单位全都合法，units.py 的生理极限拦不住它，
它会带着完全合理的外表一路走进特征层。所以：

  红线 1  短键禁入模糊层。NA(钠)/CA(钙) 编辑距离恰好为 1，K 只有一个字符，
          任何模糊匹配都会在它们之间制造灾难。长度 < 4 的归一化键只允许精确
          与折叠匹配（折叠是确定性映射，不是猜）。
  红线 2  数字位禁止参与编辑距离。HBA1C 里的 "1" 是语义的一部分；
          若允许数字位替换，未来登记 APOA1/APOB 这类名字时就会互相污染。
          数字↔字母的 OCR 混淆（1↔I、0↔O）由第 2 层确定性处理，不靠猜。
  红线 3  歧义即拒绝。折叠或模糊匹配命中 ≥2 个不同指标码时返回"无匹配"，
          并把候选列表带出去供人工复核。HDLC/LDLC 距离为 1，就是靠这条
          保护的 —— 测试里有专门用例钉死这个行为。

拒绝的代价很低：走 MISSING 三态（constants.py 的设计），模型天生能处理
"这项没测"。错配的代价是一条无法被任何下游校验发现的毒数据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..data.reference import ReferenceRegistry, normalize_alias

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR 混淆字符等价类。
# 折叠方向不重要（等价类内所有字符折到同一代表字符即可），
# 关键性质是：索引键与查询串使用【同一张表】折叠，等价类内互认。
# 表中只放印刷体 OCR 里高频、公认的混淆对，不放低频脑洞 ——
# 每加一对都会扩大碰撞面，必须克制。
# ---------------------------------------------------------------------------
_CONFUSION_FOLD: dict[str, str] = {
    "0": "O", "Q": "O", "D": "O",          # 0/O/Q/D 圆形字符族
    "1": "I", "L": "I", "|": "I", "!": "I",  # 1/I/l/竖线族
    "5": "S",
    "8": "B", "Β": "B",                     # 希腊大写 Beta -> B
    "2": "Z",
    "6": "G",
    "9": "g",  # 归一化后全大写，此行实际不会触发，保留为文档性说明
    "Μ": "M", "Α": "A", "Ε": "E", "Ο": "O", "Ρ": "P", "Τ": "T",  # 形近希腊字母
    "Γ": "R",  # γ-GT 的 γ 大写后为 Γ，OCR 常认成 r/R
    "Μ".lower().upper(): "M",
}


def fold_confusions(normalized: str) -> str:
    """对【已经过 normalize_alias】的串做混淆字符折叠。"""
    return "".join(_CONFUSION_FOLD.get(ch, ch) for ch in normalized)


# 模糊匹配参数。写成模块常量而不是构造参数：这不是调优旋钮，是安全边界。
_FUZZY_MIN_LEN = 4      # 归一化键长度低于此值禁入模糊层（红线 1）
_CONF_EXACT = 1.0
_CONF_FOLD = 0.95
_CONF_FUZZY = 0.85


@dataclass(frozen=True)
class LexiconMatch:
    """一次词典查询的结果。code=None 表示未命中（含歧义拒绝）。"""

    code: str | None
    confidence: float
    method: str  # "exact" / "fold" / "fuzzy" / "none" / "ambiguous"
    query: str
    candidates: tuple[str, ...] = field(default=())

    @property
    def matched(self) -> bool:
        return self.code is not None

    def to_log_dict(self) -> dict:
        return {
            "query": self.query,
            "code": self.code,
            "confidence": self.confidence,
            "method": self.method,
            "candidates": list(self.candidates),
        }


class IndicatorLexicon:
    """
    指标名容错匹配器。无状态查询，构造时一次性建好索引，可跨线程复用。

    索引结构：
      _exact  : normalize_alias(alias) -> code        （registry 已有，此处复用）
      _folded : fold(normalize_alias(alias)) -> code   （折叠键；碰撞键在建索引时剔除）
      _fuzzy_keys : [(归一化键, code)] 仅含长度 >= _FUZZY_MIN_LEN 的键
    """

    def __init__(self, registry: ReferenceRegistry):
        self.registry = registry
        self._folded: dict[str, str] = {}
        folded_collisions: dict[str, set[str]] = {}
        fuzzy_pool: list[tuple[str, str]] = []

        for code in registry.codes:
            meta = registry.require(code)
            names = (code, meta.name_cn, *meta.aliases)
            seen_keys: set[str] = set()
            for name in names:
                key = normalize_alias(name)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)

                fkey = fold_confusions(key)
                prev = self._folded.get(fkey)
                if prev is not None and prev != code:
                    # 两个不同指标折叠后同键：该键永久失效，记录以便运维排查词典
                    folded_collisions.setdefault(fkey, {prev}).add(code)
                else:
                    self._folded[fkey] = code

                if len(key) >= _FUZZY_MIN_LEN:
                    fuzzy_pool.append((key, code))

        for fkey, codes in folded_collisions.items():
            self._folded.pop(fkey, None)
            logger.warning(
                "词典折叠键碰撞，已停用该键的折叠匹配: %r -> %s（仍可精确匹配）",
                fkey, sorted(codes),
            )

        self._fuzzy_keys: tuple[tuple[str, str], ...] = tuple(fuzzy_pool)
        self._collisions = {k: tuple(sorted(v)) for k, v in folded_collisions.items()}

    # ------------------------------------------------------------------
    def _direct_lookup(self, raw_name: str) -> LexiconMatch:
        query = normalize_alias(raw_name)
        if not query:
            return LexiconMatch(None, 0.0, "none", query="")

        # 第 1 层：精确
        code = self.registry.resolve_alias(raw_name)
        if code is not None:
            return LexiconMatch(code, _CONF_EXACT, "exact", query=query)

        # 第 2 层：混淆折叠（确定性映射，不受长度限制）
        fkey = fold_confusions(query)
        if fkey in self._collisions:
            return LexiconMatch(
                None, 0.0, "ambiguous", query=query, candidates=self._collisions[fkey]
            )
        code = self._folded.get(fkey)
        if code is not None:
            return LexiconMatch(code, _CONF_FOLD, "fold", query=query)

        # 第 3 层：编辑距离 <= 1（红线 1：短键禁入）
        if len(query) < _FUZZY_MIN_LEN:
            return LexiconMatch(None, 0.0, "none", query=query)

        hits: dict[str, str] = {}
        for key, kcode in self._fuzzy_keys:
            if kcode in hits:
                continue
            if _within_one_edit_no_digit(query, key):
                hits[kcode] = key
        if len(hits) == 1:
            ((code, _matched_key),) = hits.items()
            return LexiconMatch(code, _CONF_FUZZY, "fuzzy", query=query)
        if len(hits) > 1:
            cands = tuple(sorted(hits))
            logger.info("模糊匹配歧义，拒绝: %r -> %s", raw_name, cands)
            return LexiconMatch(None, 0.0, "ambiguous", query=query, candidates=cands)
        return LexiconMatch(None, 0.0, "none", query=query)

    def lookup(self, raw_name: str) -> LexiconMatch:
        # 1. 尝试直接整串匹配
        res = self._direct_lookup(raw_name)
        if res.matched:
            return res

        # 2. 剥离行首序号如 1. 2、
        import re
        cleaned = re.sub(r"^\s*\d+[\.、:：\-\s]+\s*", "", raw_name).strip()
        if cleaned and cleaned != raw_name:
            res = self._direct_lookup(cleaned)
            if res.matched:
                return res

        # 3. 拆分子词（空格、括号、斜杠），优先匹配括号或各独立单词
        target = cleaned or raw_name
        tokens = [t.strip(" ()[]（）【】:：·") for t in re.split(r"[\s/()（）\[\]]+", target) if t.strip(" ()[]（）【】:：·")]
        for t in tokens:
            if len(t) >= 2 or t in self.registry.codes:
                res = self._direct_lookup(t)
                if res.matched:
                    return res

        return res


# ---------------------------------------------------------------------------
# 编辑距离 <=1 判定，带"数字位不可参与差异"约束（红线 2）。
# 三种允许的差异形态：等长单替换 / 长度差 1 的单插入或单删除。
# 差异位置只要牵涉任一侧的数字字符即判不匹配 —— 数字是语义，不是噪声。
# ---------------------------------------------------------------------------
def _within_one_edit_no_digit(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:  # 单替换
        diff = [(ca, cb) for ca, cb in zip(a, b) if ca != cb]
        if len(diff) != 1:
            return False
        ca, cb = diff[0]
        return not (ca.isdigit() or cb.isdigit())

    # 单插入/删除：保证 a 为短串
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = 0
    edited_char: str | None = None
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if edited_char is not None:
            return False
        edited_char = b[j]
        j += 1
    if edited_char is None:  # 差异在末尾
        edited_char = b[-1]
    return not edited_char.isdigit()
