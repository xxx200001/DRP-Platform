"""
输出合规拦截（规范 7）。

规范原话：
  - "代码层面禁止输出确诊、诊断、治疗、开药等词汇"
  - "所有结果页强制挂载免责声明"

【为什么必须做在输出边界，而不是靠提示词约束大模型】
规范 3.1 允许大模型做"文本报告解析、结构化、解释输出"。大模型的输出是
概率性的：同一套提示词，99 次规规矩矩，第 100 次冒出一句"建议服用水飞蓟宾"。
靠提示词约束是把合规责任交给一个采样过程，靠出口检测才是确定性的。
所以本模块的定位是【最后一道闸】：任何要展示给用户的文案，在离开进程前
必须过一遍 assert_compliant()。

【为什么检测到违规是拒绝而不是自动改写】
自动删词会产出语义残缺的句子（"建议您XX后复查"），用户看不懂反而更危险；
更要命的是它掩盖了上游问题 —— 这次被悄悄改掉了，下次它会以别的形式漏出来。
正确做法是抛异常，由服务层降级到人工审核过的安全兜底文案，同时打点告警，
让"模型又想开药了"这件事被看见。

【关于误伤】
规范 6 要求做"智能就医建议：精准推荐科室、检查项目"，所以"建议""科室"
"检查""复查""随访""就诊"全部是允许词，绝不能进禁用表。禁用表只针对
【医生的执业行为】：下诊断、定治疗方案、开药、给剂量、承诺疗效。
边界拿捏错了会让产品变成哑巴，这比拦不住更常见。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ComplianceError(ValueError):
    """输出文案触碰合规红线。服务层必须捕获并降级到安全兜底文案，不能直接透出。"""


#: 强制免责声明（规范 7）。所有结果页、所有导出文件都要挂。
DISCLAIMER = (
    "本结果由算法模型基于您提供的检验数据估算得出，仅供健康管理参考，"
    "不构成任何医疗意见。模型输出的是统计意义上的风险概率，不代表个体一定会或"
    "不会发生相应情况。如有不适或对结果存疑，请及时前往正规医疗机构就诊，"
    "由执业医师作出判断。"
)

#: 禁用词按类分组，报错时能直接告诉开发者踩的是哪条红线。
FORBIDDEN_TERMS: dict[str, tuple[str, ...]] = {
    "下诊断": (
        "确诊", "诊断为", "可诊断", "诊断结果", "本病例诊断",
        "判定为.{0,4}病", "您患有", "你患有", "已患", "属于.{0,4}患者",
    ),
    "定治疗方案": (
        "治疗方案", "建议治疗", "需要治疗", "进行治疗", "疗程", "手术方案",
        "保守治疗", "对症治疗",
    ),
    "开药与剂量": (
        "开药", "处方", "服用", "口服", "静脉注射", "用药", "停药", "换药",
        "每日.{0,6}(mg|毫克|片|粒)", "剂量", "药物名称",
    ),
    "承诺疗效": (
        "治愈", "痊愈", "根治", "保证不会", "百分之百", "100%(治好|有效|准确)",
        "一定(会|能)康复", "无需就医",
    ),
}

#: 允许词白名单：这些词与禁用词形近，但规范 6 明确要求平台输出它们。
#: 命中白名单的片段在检测时先行剔除，避免"建议就诊消化内科"被误杀。
ALLOWED_PHRASES: tuple[str, ...] = (
    "建议就诊", "建议复查", "建议随访", "建议咨询", "建议前往",
    "推荐科室", "推荐检查", "检查项目", "复查建议", "就医建议",
    "由执业医师", "正规医疗机构", "不构成任何医疗意见",
    # OGTT 是检验医学标准项目名，"口服"是项目名的一部分而非给药指示（规范 6 检查推荐需要它）
    "口服葡萄糖耐量",
)

_COMPILED: dict[str, list[tuple[str, re.Pattern]]] = {
    cat: [(t, re.compile(t)) for t in terms] for cat, terms in FORBIDDEN_TERMS.items()
}


@dataclass(frozen=True)
class Violation:
    category: str
    term: str
    span: tuple[int, int]
    context: str

    def __str__(self) -> str:
        return f"[{self.category}] 命中「{self.term}」于位置 {self.span[0]}：…{self.context}…"


def _mask_allowed(text: str) -> str:
    """把白名单片段替换成等长占位，既不改变偏移量，又不让它们参与匹配。"""
    out = text
    for phrase in ALLOWED_PHRASES:
        if phrase in out:
            out = out.replace(phrase, "\u3000" * len(phrase))
    return out


def scan(text: str) -> list[Violation]:
    """扫描文案，返回全部违规命中（空列表 = 合规）。"""
    if not text:
        return []
    masked = _mask_allowed(text)
    found: list[Violation] = []
    for cat, pats in _COMPILED.items():
        for term, pat in pats:
            for m in pat.finditer(masked):
                lo, hi = m.span()
                found.append(
                    Violation(
                        category=cat,
                        term=term,
                        span=(lo, hi),
                        context=text[max(0, lo - 12) : hi + 12],
                    )
                )
    return found


def is_compliant(text: str) -> bool:
    return not scan(text)


def assert_compliant(text: str, source: str = "") -> None:
    """
    硬断言。所有对外文案在离开进程前必须过这一关。

    source 传上游来源（如 "llm_explain" / "tier_advice"），
    出问题时能直接定位是哪个环节生成的，而不是在一大堆日志里翻。
    """
    v = scan(text)
    if v:
        detail = "\n".join(f"  - {x}" for x in v[:5])
        raise ComplianceError(
            f"文案触碰合规红线（来源={source or '未标注'}，共 {len(v)} 处）：\n{detail}\n"
            "规范 7 禁止输出确诊/治疗/开药类内容。请降级到安全兜底文案并告警，"
            "不要在此处自动删词 —— 那只会掩盖上游问题。"
        )


def attach_disclaimer(text: str) -> str:
    """挂载免责声明（规范 7）。重复调用不会叠加。"""
    if DISCLAIMER in text:
        return text
    return f"{text.rstrip()}\n\n——\n{DISCLAIMER}"


def safe_fallback(tier_name: str = "") -> str:
    """
    合规拦截触发后的兜底文案。内容经人工审核，不含任何模型生成成分。
    宁可给一句正确的废话，也不能把可能违规的句子推给用户。
    """
    head = f"您本次的风险分层结果为「{tier_name}」。" if tier_name else "您的风险评估已完成。"
    return attach_disclaimer(
        head + "详细的指标解读暂不可用，建议携带完整检验报告就诊，由医师结合临床情况判断。"
    )
