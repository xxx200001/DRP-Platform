"""
AI 大模型临床辅助解读与干预方案引擎 (LLM Clinical Advisor)。

V3.3 重写要点（对应反馈"很多地方还是固态的，不同化验单给不出不同结果"）：
  1. 内置兜底引擎不再输出整段固定文案。五个章节（机制/膳食/生活方式/
     随访/红旗）全部由【该患者最新快照 + 本次 vs 上次真实变化 + 风险轨迹】
     逐条拼装，句子里写入真实数值、超限倍数与变化方向 —— 两份不同的
     化验单必然得到不同输出；全部平稳时明确说平稳，而不是硬凑建议。
  2. 密钥与端点只从环境变量读取（不再有任何硬编码缺省 key/URL）；
     超时通过 SOULHEALTH_LLM_TIMEOUT 配置（默认 25s——旧默认 2s 使在线
     LLM 实际上永远失败，界面看似"AI 生成"实为兜底套话）。
  3. 在线 LLM 文本仍必须过 compliance.is_compliant 出口闸；兜底引擎的
     所有措辞避开禁用词（不出现"处方/服用/口服/用药/治疗/确诊"等），
     由 tests/test_dynamic_advice.py 逐句扫描钉死。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from .compliance import is_compliant

logger = logging.getLogger(__name__)

# 时程的中文标签（风险走势叙述用）
_H_CN = {"1y": "未来 1 年", "3y": "未来 3 年", "5y": "未来 5 年"}


# ---------------------------------------------------------------------------
# 环境配置（单一真源：server.py 的两处 LLM 调用也从这里取，杜绝硬编码）
# ---------------------------------------------------------------------------
def resolve_llm_env() -> dict[str, Any]:
    """
    返回 {"provider","api_key","base_url","model","timeout"}；未配置密钥时
    provider 为 None —— 调用方必须跳过在线调用，直接走兜底引擎。
    """
    timeout = 25.0
    try:
        timeout = float(os.environ.get("SOULHEALTH_LLM_TIMEOUT", "25"))
    except ValueError:
        pass

    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if anthropic_key and "change-me" not in anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "base_url": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            "model": os.environ.get("SOULHEALTH_LLM_MODEL", "claude-sonnet-4-6"),
            "timeout": timeout,
        }
    openai_key = (os.environ.get("OPENAI_API_KEY")
                  or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if openai_key and "change-me" not in openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
            "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
            "timeout": timeout,
        }
    return {"provider": None, "api_key": "", "base_url": "", "model": "", "timeout": timeout}


def _call_openai_compatible(api_key: str, base_url: str, model: str,
                            prompt: str, system_prompt: str,
                            timeout: float = 25.0) -> str | None:
    """调用兼容 OpenAI 规范的大模型接口 (如 DeepSeek, OpenAI, 通义千问等)。"""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.info("[LLM] 外部在线接口不可用 (%s)，切换至内置引擎", str(e))
        return None


def _call_anthropic(api_key: str, base_url: str, model: str,
                    prompt: str, system_prompt: str,
                    timeout: float = 25.0) -> str | None:
    """调用 Anthropic Claude 规范接口。"""
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "User-Agent": "SoulHealth-DRP/3.3",
    }
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"].strip()
    except Exception as e:
        logger.info("[LLM] 外部 Claude 接口不可用 (%s)，切换至内置引擎", str(e))
        return None


def call_llm(prompt: str, system_prompt: str) -> str | None:
    """按环境配置调用在线大模型；未配置或失败返回 None。"""
    env = resolve_llm_env()
    if env["provider"] == "anthropic":
        return _call_anthropic(env["api_key"], env["base_url"], env["model"],
                               prompt, system_prompt, timeout=env["timeout"])
    if env["provider"] == "openai":
        return _call_openai_compatible(env["api_key"], env["base_url"], env["model"],
                                       prompt, system_prompt, timeout=env["timeout"])
    return None


# ---------------------------------------------------------------------------
# 内置数据驱动引擎的构件
# ---------------------------------------------------------------------------
_SYS_DEFS: tuple[dict, ...] = (
    {"key": "liver", "name": "肝胆代谢", "dept": "消化内科（或肝病科）",
     "codes": ("ALT", "AST", "GGT", "ALP", "TBIL", "DBIL", "ALB"),
     "recheck": ("肝功能全套", "肝胆胰脾超声")},
    {"key": "lipid", "name": "血脂与心血管", "dept": "心血管内科（或内分泌科）",
     "codes": ("TG", "TC", "LDLC", "HDLC"),
     "recheck": ("血脂四项", "颈动脉超声（评估内中膜）")},
    {"key": "glucose", "name": "血糖代谢", "dept": "内分泌科",
     "codes": ("GLU", "HBA1C", "INS"),
     "recheck": ("空腹静脉血糖", "糖化血红蛋白 (HbA1c)")},
    {"key": "renal", "name": "肾功能与尿酸", "dept": "肾内科",
     "codes": ("UA", "CREA", "UREA", "UACR"),
     "recheck": ("肾功能与血尿酸", "晨尿常规")},
    {"key": "bp", "name": "血压", "dept": "心血管内科",
     "codes": ("SBP", "DBP"),
     "recheck": ("多次静息血压测量", "动态血压监测")},
    {"key": "body", "name": "体重管理", "dept": "临床营养科（或内分泌科）",
     "codes": ("BMI",),
     "recheck": ("体成分分析",)},
)


def _fmt_val(e: dict) -> str:
    return f"{e['name_cn']} {e['value']:g}{e.get('unit') or ''}"


def _fmt_over(e: dict) -> str:
    """"达上限的 1.8 倍 / 低于下限约 25%"，参考界缺失时返回空串。"""
    g = e.get("grade", 0)
    try:
        if g > 0 and e.get("ref_high"):
            k = e["value"] / float(e["ref_high"])
            if k > 1.005:
                txt = f"{k:.1f}"
                return "略超参考上限" if txt == "1.0" else f"达参考上限的 {txt} 倍"
        if g < 0 and e.get("ref_low"):
            pct = (float(e["ref_low"]) - e["value"]) / abs(float(e["ref_low"])) * 100
            if pct > 0.5:
                return f"低于参考下限约 {pct:.0f}%"
    except (TypeError, ZeroDivisionError, ValueError):
        pass
    return ""


def _trend_phrase(comp: dict | None) -> str:
    if not comp:
        return ""
    if comp.get("is_real_change"):
        pct = comp.get("delta_pct")
        pt = f"（{pct:+.0%}）" if isinstance(pct, (int, float)) else ""
        return f"，较上次{comp.get('direction', '变化')}{pt}" + \
               ("且程度加重" if comp.get("worsened") else "")
    return "，较上次基本平稳"


def _sentence(e: dict, comp: dict | None) -> str:
    over = _fmt_over(e)
    return _fmt_val(e) + (f"，{over}" if over else "") + _trend_phrase(comp)


def _build_expert_knowledge_analysis(
    patient_info: dict[str, Any],
    comparisons: list[dict[str, Any]],
    risk_trajectories: dict[str, Any],
    factors: list[str] | None = None,
    snapshot: list[dict] | None = None,
    risk_tier: str | None = None,
    span_label: str | None = None,
) -> dict[str, Any]:
    """
    内置医学知识引擎：全部内容由该患者的真实数据逐条拼装。
    snapshot（每指标最新值+分级+参考界）是主判据；comparisons 只补充趋势。
    """
    sex = str(patient_info.get("sex") or "未知")
    age_raw = patient_info.get("age")
    try:
        age_num: float | None = float(age_raw)
    except (TypeError, ValueError):
        age_num = None
    age_txt = f"{age_num:.0f}" if age_num is not None else "—"

    comp_map = {str(c.get("code")): c for c in (comparisons or [])}
    snap = [dict(e) for e in (snapshot or []) if e.get("value") is not None]
    abnormal = {e["code"]: e for e in snap if e.get("grade")}
    if not snap and comparisons:  # 兼容旧调用（无快照）：从对比还原
        for c in comparisons:
            if c.get("curr_grade") or c.get("is_real_change"):
                abnormal[c["code"]] = {
                    "code": c["code"], "name_cn": c.get("name_cn", c["code"]),
                    "unit": c.get("unit", ""), "value": c.get("curr_value"),
                    "grade": c.get("curr_grade", 0), "ref_low": None, "ref_high": None,
                }

    # 命中的系统（按最严重排序）
    hits: list[dict] = []
    for d in _SYS_DEFS:
        es = sorted((abnormal[c] for c in d["codes"] if c in abnormal),
                    key=lambda e: -abs(e.get("grade", 0)))
        if not es:
            continue
        worsened = [e for e in es
                    if (comp_map.get(e["code"]) or {}).get("worsened")]
        hits.append({**d, "entries": es, "worsened": worsened,
                     "max_grade": max(abs(e.get("grade", 0)) for e in es)})
    hits.sort(key=lambda h: (-len(h["worsened"]), -h["max_grade"], -len(h["entries"])))

    n_abn = sum(len(h["entries"]) for h in hits)
    span_txt = f"（数据跨度 {span_label}）" if span_label else ""

    # ---------------- 章节一：机制剖析（逐系统、带真实数值） ----------------
    mech: list[str] = []
    if hits:
        lead_bits = "；".join(
            _sentence(h["entries"][0], comp_map.get(h["entries"][0]["code"]))
            for h in hits[:3]
        )
        mech.append(
            f"本次纵向核对{span_txt}共发现 {n_abn} 项指标偏离参考区间，"
            f"集中在{ '、'.join(h['name'] for h in hits) }：{lead_bits}。")
    else:
        mech.append(
            f"本次纵向核对{span_txt}各项主要指标均在参考区间内，"
            "与既往相比未见超出个体生物学变异的实质性波动。")

    for h in hits:
        parts = "、".join(_sentence(e, comp_map.get(e["code"])) for e in h["entries"][:3])
        if h["key"] == "liver":
            mech.append(
                f"【肝细胞与胆道酶学】{parts}。转氨酶反映肝细胞膜通透性变化，"
                "GGT/ALP 对酒精、脂肪蓄积与胆汁排泄负担敏感；结合超声若提示脂肪肝，"
                "提示肝内脂质蓄积是当前酶学偏离的主要背景，需通过体重与生活方式管理逆转。")
        elif h["key"] == "lipid":
            mech.append(
                f"【脂质代谢】{parts}。致动脉粥样硬化脂质谱的持续偏离会增加血管内皮"
                "慢性炎症负担；甘油三酯偏高叠加高密度脂蛋白偏低时，脂质清除能力下降，"
                "与肝内脂肪蓄积互为因果。")
        elif h["key"] == "glucose":
            mech.append(
                f"【糖代谢】{parts}。提示外周组织胰岛素敏感性下降、胰岛处于代偿阶段，"
                "长期高糖环境会加速微血管与大血管改变。")
        elif h["key"] == "renal":
            mech.append(
                f"【肾脏与嘌呤代谢】{parts}。尿酸偏高可在肾小管形成微晶负担，"
                "并与代谢综合征相互放大；肌酐/尿素的走向需要连续观察而非单点判断。")
        elif h["key"] == "bp":
            mech.append(
                f"【血压负荷】{parts}。持续偏高会逐步累及心、脑、肾等靶器官，"
                "家庭自测血压的多次记录比单次门诊测量更能反映真实负荷。")
        elif h["key"] == "body":
            mech.append(
                f"【体成分】{parts}。超重是上述代谢指标共同的上游因素，"
                "减重 5%~10% 通常可带来肝酶、血脂与血糖的联动改善。")

    # 风险轨迹叙述：首末两点真实变化（有 ≥2 点才叙述"走势"）
    risk_summary: list[str] = []
    for hkey, traj in sorted((risk_trajectories or {}).items()):
        pts = (traj or {}).get("points") or []
        if not pts:
            continue
        cur = pts[-1]
        label = _H_CN.get(hkey, hkey)
        if len(pts) >= 2:
            first = pts[0]
            d0 = str(first.get("at", ""))[:10]
            d1 = str(cur.get("at", ""))[:10]
            risk_summary.append(
                f"{label}风险由 {first.get('probability', 0):.1%}（{d0}）"
                f"变化至 {cur.get('probability', 0):.1%}（{d1}），"
                f"当前分层「{cur.get('risk_tier', '—')}」")
        else:
            risk_summary.append(
                f"{label}风险当前为 {cur.get('probability', 0):.1%}"
                f"（分层「{cur.get('risk_tier', '—')}」）")
    if risk_summary:
        mech.append("【模型风险轨迹】" + "；".join(risk_summary) + "。以上概率由统计模型"
                    "基于全部历史检查数据计算，AI 仅作解释，不另行生成数值。")

    hit_keys = {h["key"] for h in hits}
    by_key = {h["key"]: h for h in hits}
    gv = lambda c: abnormal[c]["value"]  # noqa: E731

    # ---------------- 章节二：膳食（只给命中的系统，句子带真实数值） ----------------
    diet_sections: list[dict] = []
    if "liver" in hit_keys:
        items = []
        if "GGT" in abnormal:
            items.append(f"严格戒酒（含啤酒、红酒与含酒精饮料）——GGT {gv('GGT'):g} 对酒精负担最敏感，戒断后通常数周内回落")
        else:
            items.append("严格戒酒，减少对肝实质细胞的持续刺激")
        if "ALT" in abnormal or "AST" in abnormal:
            e = abnormal.get("ALT") or abnormal.get("AST")
            items.append(f"{e['name_cn']}已达 {e['value']:g}{e.get('unit') or ''}——限制油炸食品与高果糖浆（奶茶、含糖饮料），每日烹调油 ≤25g")
        if {"ALP", "TBIL", "DBIL"} & abnormal.keys():
            items.append("胆道相关指标偏离期间饮食宜清淡规律、少食多餐，避免一次性大量油腻加重胆汁排泄负担")
        items.append("以清蒸鱼、脱脂奶、豆制品补充优质蛋白，配深色蔬菜提供抗氧化底物")
        diet_sections.append({"title": "肝胆负担管理（按本次肝功能结果定制）", "items": items[:4]})
    if "lipid" in hit_keys or "body" in hit_keys:
        items = []
        if "TG" in abnormal:
            items.append(f"甘油三酯 {gv('TG'):g} mmol/L——甜食、精制碳水与酒精是其最直接的原料，优先从这三样减起")
        if "LDLC" in abnormal or "TC" in abnormal:
            e = abnormal.get("LDLC") or abnormal.get("TC")
            items.append(f"{e['name_cn']} {e['value']:g} mmol/L——限制动物内脏、肥肉与含反式脂肪的起酥食品，烹调换用橄榄油/茶籽油")
        if "HDLC" in abnormal and abnormal["HDLC"].get("grade", 0) < 0:
            items.append(f"高密度脂蛋白 {gv('HDLC'):g} mmol/L 偏低——每周 2~3 次深海鱼补充 Omega-3，配合规律运动最能把它抬回来")
        items.append("主食中掺入燕麦、糙米等可溶性膳食纤维，帮助结合胆酸、延缓糖脂吸收")
        diet_sections.append({"title": "血脂结构调整（按本次血脂四项定制）", "items": items[:4]})
    if "glucose" in hit_keys:
        items = []
        if "GLU" in abnormal:
            items.append(f"空腹血糖 {gv('GLU'):g} mmol/L——控制每餐碳水总量，以低 GI 粗杂粮替代 1/3~1/2 精制米面")
        if "HBA1C" in abnormal:
            items.append(f"糖化血红蛋白 {gv('HBA1C'):g}%——反映近 3 个月平均水平，长期戒断含糖饮料比单日控糖更关键")
        items.append("进餐顺序：先汤和蔬菜、再蛋白质、最后主食，可显著削平餐后血糖峰值")
        diet_sections.append({"title": "血糖平稳策略（按本次血糖结果定制）", "items": items[:3]})
    if "renal" in hit_keys:
        items = []
        if "UA" in abnormal:
            items.append(f"血尿酸 {gv('UA'):g} μmol/L——限制动物内脏、浓肉汤/火锅汤底、贝类与啤酒等高嘌呤来源")
        items.append("每日饮水 2000~2500 mL 分次均匀摄入，促进代谢废物经肾排出")
        if {"CREA", "UREA"} & abnormal.keys():
            items.append("蛋白质以适量优质蛋白为主，避免长期高蛋白饮食增加肾小球滤过负担")
        diet_sections.append({"title": "肾脏与尿酸友好饮食（按本次肾功能结果定制）", "items": items[:3]})
    if "bp" in hit_keys:
        e = abnormal.get("SBP") or abnormal.get("DBP")
        diet_sections.append({"title": "限盐方案（按本次血压定制）", "items": [
            f"{e['name_cn']}达 {e['value']:g} mmHg——每日食盐 ≤5g（约一平啤酒瓶盖），警惕酱菜、加工肉与外卖的隐形钠",
            "增加富钾食物（新鲜蔬果、豆类、低脂奶）帮助对冲钠负荷",
        ]})
    if not diet_sections:
        diet_sections.append({"title": "均衡维持（本次未发现异常，给通用原则）", "items": [
            "食物多样、荤素搭配，多食新鲜蔬果与全谷物粗杂粮",
            "清淡少油少盐、规律进餐，避免暴饮暴食",
        ]})

    # ---------------- 章节三：生活方式（运动参数按真实年龄计算） ----------------
    lifestyle_sections: list[dict] = []
    ex_items = []
    if age_num is not None:
        hr_lo, hr_hi = int((220 - age_num) * 0.6), int((220 - age_num) * 0.7)
        ex_items.append(
            f"每周 ≥5 次、每次 30~45 分钟中等强度有氧（快走/慢跑/游泳），"
            f"按你的年龄（{age_txt} 岁）运动时心率维持在 {hr_lo}~{hr_hi} 次/分")
    else:
        ex_items.append("每周 ≥5 次、每次 30~45 分钟中等强度有氧运动（快走/慢跑/游泳）")
    if "glucose" in hit_keys or "lipid" in hit_keys:
        ex_items.append("餐后 30 分钟起进行 15~20 分钟轻度活动，削平餐后血糖与甘油三酯峰值")
    ex_items.append("每周 2 次全身大肌群抗阻训练（深蹲、弹力带、哑铃），提升肌肉对葡萄糖的非胰岛素依赖摄取")
    lifestyle_sections.append({"title": "分级运动方案", "items": ex_items[:3]})

    ls_items = []
    if "liver" in hit_keys or "glucose" in hit_keys:
        ls_items.append("23 点前入睡、保证 7~8 小时高质量睡眠——昼夜节律紊乱会直接推高皮质醇与游离脂肪酸")
    if "body" in hit_keys:
        ls_items.append(f"BMI {gv('BMI'):g}——以每月减重 1~2 公斤为节奏，配合抗阻训练防止肌肉流失")
    if "lipid" in hit_keys or "bp" in hit_keys:
        ls_items.append("戒烟并远离二手烟，限制饮酒，避免剧烈情绪波动与骤冷骤热刺激")
    if "bp" in hit_keys:
        ls_items.append("每天早晚各测一次静息血压并记录，复诊时把血压日记带给医生")
    if not ls_items:
        ls_items.append("保持心态平衡、劳逸结合，养成稳定的睡眠生物钟")
    lifestyle_sections.append({"title": "作息与节律管理", "items": ls_items[:4]})

    # ---------------- 章节四：随访日程（周期按分层，项目按命中系统） ----------------
    tier = (risk_tier or "").strip()
    cyc = {"极高危": ("2 周内", "1 个月后", "3 个月后"),
           "高危": ("2~4 周后", "1~3 个月后", "6 个月后"),
           "中危": ("1~2 个月后", "3~6 个月后", "12 个月后")}.get(
        tier, ("1~3 个月后", "3~6 个月后", "12 个月后"))
    short_items = [f"{h['recheck'][0]}（重点跟踪 {h['entries'][0]['name_cn']}）"
                   for h in hits[:2]] or ["本次异常项对应的复查"]
    medium_items = [it for h in hits for it in h["recheck"][1:]][:3] or ["常规生化指标跟踪"]
    long_items = ["全面健康体检与风险模型复测"]
    if "lipid" in hit_keys or "bp" in hit_keys:
        long_items.append("颈动脉超声（评估内中膜厚度）")
    depts = list(dict.fromkeys(h["dept"] for h in hits))[:3]
    followup_plan = {
        "cycle_short": cyc[0], "cycle_short_items": short_items,
        "cycle_medium": cyc[1], "cycle_medium_items": medium_items,
        "cycle_long": cyc[2], "cycle_long_items": long_items,
        "recommend_dept": " / ".join(depts) if depts else "全科医学科（定期体检随访）",
    }

    # ---------------- 章节五：红旗信号（只列与命中系统相关的） ----------------
    red_flags: list[str] = []
    if "liver" in hit_keys:
        red_flags.append("皮肤或巩膜发黄、尿色深如浓茶、持续右上腹胀痛伴明显乏力厌油——建议尽快就诊消化内科")
    if "lipid" in hit_keys or "bp" in hit_keys:
        red_flags.append("持续胸前区压榨性疼痛、胸闷伴大汗或向左肩放射，或突发单侧肢体无力、口角歪斜、言语不清——立即拨打急救电话")
    if "glucose" in hit_keys:
        red_flags.append("多饮多尿多食伴体重快速下降，或视物模糊、手足麻木——建议尽快就诊内分泌科")
    if "renal" in hit_keys:
        red_flags.append("关节急性红肿热痛发作，或眼睑/下肢浮肿、尿中泡沫经久不散——建议尽快就诊肾内科")
    if not red_flags:
        red_flags.append("日常如出现明显不适或症状持续不缓解，请及时前往正规医疗机构就诊")

    n_rec = patient_info.get("n_records", 0)
    summary_bits = [f"{sex} · {age_txt}岁", f"累计 {n_rec} 条检验记录"]
    if span_label:
        summary_bits.append(f"跨度 {span_label}")
    if n_abn:
        summary_bits.append(f"本次 {n_abn} 项指标异常")

    return {
        "source": "AI_CLINICAL_ENGINE",
        "patient_name": str(patient_info.get("name") or "受检者"),
        "patient_summary": " · ".join(summary_bits),
        "risk_trajectory_summary": risk_summary,
        "pathology_mechanism": mech,
        "diet_interventions": diet_sections,
        "lifestyle_interventions": lifestyle_sections,
        "followup_plan": followup_plan,
        "red_flags": red_flags,
        "n_abnormal": n_abn,
        "abnormal_systems": [h["name"] for h in hits],
        "disclaimer": (
            "本内容由平台根据检验时序数据与统计风险估算自动生成，"
            "仅供健康管理参考，不构成任何医疗意见；具体安排请遵从执业医师判断。"
        ),
    }


# ---------------------------------------------------------------------------
# 对外统一入口
# ---------------------------------------------------------------------------
def generate_llm_trend_analysis(
    patient_info: dict[str, Any],
    comparisons: list[dict[str, Any]],
    risk_trajectories: dict[str, Any],
    factors: list[str] | None = None,
    snapshot: list[dict] | None = None,
    risk_tier: str | None = None,
    span_label: str | None = None,
) -> dict[str, Any]:
    """
    优先调用外部配置的 LLM 产出叙述文本（必须过合规出口闸）；
    结构化五章节始终由内置数据驱动引擎按该患者真实数据生成 ——
    在线与否只影响是否多一段大模型叙述，不影响个体化程度。
    """
    system_prompt = (
        "你是一名临床检验医学与慢性病健康管理专家。"
        "请根据患者的多期化验单时序变化、异常指标（含真实数值与超限程度）"
        "及统计模型给出的多时程风险概率，输出专业、结构化且通俗的病情深度剖析"
        "与个性化生活方式应对办法。只解释已给出的模型概率，不要自行编造新概率。"
        "严格合规：不使用『确诊/治疗/开药/剂量』等词，就医相关一律用『建议就诊/建议复查』。"
    )
    abn_lines = []
    for e in (snapshot or []):
        if e.get("grade"):
            comp = next((c for c in comparisons or []
                         if c.get("code") == e.get("code")), None)
            abn_lines.append("- " + _sentence(e, comp))
    prompt = (
        f"患者：{patient_info.get('sex', '未知')}，{patient_info.get('age', '—')}岁，"
        f"累计 {patient_info.get('n_records', 0)} 条检验记录"
        + (f"，数据跨度 {span_label}" if span_label else "") + "。\n"
        f"当前异常指标（真实值+较上次变化）：\n" + ("\n".join(abn_lines) or "（无）") + "\n"
        f"指标时序对比数据：{json.dumps(comparisons, ensure_ascii=False)}\n"
        f"模型风险走势：{json.dumps(risk_trajectories, ensure_ascii=False)}\n"
        f"主要风险驱动因素：{json.dumps(factors or [], ensure_ascii=False)}\n\n"
        "请生成针对这位患者具体数值与走势的深度解读与生活方式应对办法。"
    )

    raw_llm_text = call_llm(prompt, system_prompt)

    expert_data = _build_expert_knowledge_analysis(
        patient_info, comparisons, risk_trajectories, factors,
        snapshot=snapshot, risk_tier=risk_tier, span_label=span_label,
    )

    if raw_llm_text and is_compliant(raw_llm_text):
        expert_data["llm_narrative_text"] = raw_llm_text
        expert_data["source"] = "AI_ONLINE_LLM"
    else:
        if raw_llm_text:
            logger.warning("[LLM] 在线叙述未通过合规出口闸，已降级为内置引擎输出")
        expert_data["source"] = "AI_CLINICAL_KNOWLEDGE_ENGINE"

    return expert_data
