/* =====================================================================
   病情预测平台 · 前端逻辑 v2（零构建、零 CDN：echarts 已本地化）
   ---------------------------------------------------------------------
   信息层级只有一条主线：选患者 → 录化验单 → 跑预测 → 看归因 → 回流随访。
   页面按这条主线切成四个目的地，而不是把所有面板铺在一屏里。

   两个自绘元件（没用 echarts，因为它们要在移动端窄屏里保持可读）：
     · band      参考区间带：值落在区间的哪里 + 上次在哪里
     · tierscale 分层刻度：四段等宽，切点用 /api/meta.tiers 的真实值标注
   ===================================================================== */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const H_LABEL = { "1y": "1 年", "3y": "3 年", "5y": "5 年" };
const TIER_IDX = { "低危": 1, "中危": 2, "高危": 3, "极高危": 4 };
const GRADE_CN = { "-3": "重度偏低", "-2": "中度偏低", "-1": "轻度偏低", "0": "正常",
                   "1": "轻度偏高", "2": "中度偏高", "3": "重度偏高" };

const state = {
  meta: null, patients: [], pid: null, patient: null,
  records: [], refMap: {}, trend: null, charts: {},
  predicted: false, followedUp: false,
  // 改版新增
  reports: null,          // /reports 响应（历史检查资料清单）
  timeline: null,         // /timeline 响应（健康时间轴确认数据）
  riskTimeline: null,     // /risk-timeline 响应（按检查日期回溯的风险轨迹）
  pending: [],            // 批量上传后待确认入库的报告 [{id,name,text,date,detected,src,nlines}]
  concern: "all",        // 本次最关注的问题
  lastPredict: null,      // 最近一次 /predict 响应
  selHorizon: "3y",      // 未来预测区当前选中的时程
  riskHorizon: "3y",     // 趋势页风险走势选中的时程
  seriesRange: "all",    // 指标历史趋势时间范围
};

/* ---------------- 基础设施 ---------------- */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct1 = (p) => (p * 100).toFixed(1) + "%";
/** 有效位随量级走，并去掉尾随零：8.40 → 8.4，1.70 → 1.7。 */
const num = (v) => (Number.isFinite(v)
  ? String(parseFloat((Math.abs(v) >= 100 ? v.toFixed(0)
      : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3))))
  : "—");

function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = bad ? "on bad" : "on";
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.className = ""), bad ? 5200 : 2600);
}

async function api(path, opts = {}) {
  const res = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* 保留 statusText */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* 隐私模式下忽略 */ } },
};

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function chart(id) {
  if (!state.charts[id]) {
    state.charts[id] = echarts.init($("#" + id), null, { renderer: "canvas" });
    window.addEventListener("resize", () => state.charts[id].resize());
  }
  return state.charts[id];
}
/** 所有图表共用的底座：去掉 echarts 默认的边框、亮色和粗轴线。 */
function baseOption() {
  const line = token("--line"), ink3 = token("--ink-3");
  return {
    textStyle: { fontFamily: token("--sans"), color: token("--ink-2") },
    grid: { left: 8, right: 14, top: 26, bottom: 6, containLabel: true },
    tooltip: {
      trigger: "axis",
      backgroundColor: token("--surface"), borderColor: line,
      textStyle: { color: token("--ink"), fontSize: 12 },
      extraCssText: "box-shadow:0 8px 24px -12px rgba(0,0,0,.4);border-radius:10px",
    },
    xAxis: {
      type: "time", axisLine: { lineStyle: { color: line } },
      axisTick: { show: false }, axisLabel: { color: ink3, fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value", axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: ink3, fontSize: 11 },
      splitLine: { lineStyle: { color: line, type: "dashed" } },
      nameTextStyle: { color: ink3, fontSize: 11 },
    },
  };
}
const tierColor = (tier) => token("--t" + (TIER_IDX[tier] || 1));

/* ---------------- 导航 ---------------- */
function go(page) {
  // 自动流转倒计时进行中，用户任何手动切页都视为接管，取消自动跳转
  if (state.autoTimer && !state._autoNav) cancelAutoJump();
  $$(".navbtn").forEach((b) => b.setAttribute("aria-current", String(b.dataset.go === page)));
  $$(".page").forEach((p) => p.classList.toggle("on", p.id === "page-" + page));
  window.scrollTo({ top: 0, behavior: "instant" });
  if (page === "trend") loadTrend();
  if (page === "admin") refreshAdmin();
  setTimeout(() => Object.values(state.charts).forEach((c) => c.resize()), 30);
}
$$(".navbtn").forEach((b) => b.addEventListener("click", () => go(b.dataset.go)));
$("#btnSwitch").addEventListener("click", () => go("patients"));
$$(".step").forEach((b) => b.addEventListener("click", () => {
  const map = { ingest: "#sec-ingest", timeline: "#sec-timeline",
                predict: "#sec-predict", followup: "#sec-followup" };
  $(map[b.dataset.step]).scrollIntoView({ behavior: "smooth", block: "start" });
}));

/* ---------------- 启动 ---------------- */
async function boot() {
  state.meta = await api("/meta");
  const m = state.meta;

  $("#disclaimerBar").textContent = "免责声明：" + m.disclaimer;
  $("#railVersion").textContent = m.active_version || "未上线";
  $("#railVersion").title = m.canary
    ? `灰度 ${m.canary.version} @ ${m.canary.traffic_pct}%` : "无灰度";
  $("#repDate").value = new Date().toISOString().slice(0, 10);

  const opts = m.horizons.map((h) => `<option value="${h}">${H_LABEL[h] || h}</option>`).join("");
  $("#driftHorizon").innerHTML = opts;
  $("#abHorizon").innerHTML = `<option value="">全部时程</option>` + opts;

  $("#dbStats").innerHTML = [
    ["患者", m.stats.patients], ["化验记录", m.stats.lab_records],
    ["预测次数", m.stats.predictions],
  ].map(([k, v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");

  await refreshPatients();
  const saved = store.get("drp.pid");
  if (saved && state.patients.some((p) => p.patient_id === saved)) await selectPatient(saved, false);
}

/* ---------------- 患者 ---------------- */
async function refreshPatients() {
  state.patients = await api("/patients");
  $("#patientCount").textContent = `${state.patients.length} 位`;
  renderPatientList();
}

function renderPatientList() {
  const q = $("#patientSearch").value.trim().toLowerCase();
  const list = state.patients.filter((p) => !q || p.patient_id.toLowerCase().includes(q));
  const box = $("#patientList");
  if (!list.length) {
    box.innerHTML = `<div class="empty"><b>${state.patients.length ? "没有匹配的编号" : "还没有患者档案"}</b>${
      state.patients.length ? "换个关键词试试" : "点下面的按钮建第一个"}</div>`;
    return;
  }
  box.innerHTML = list.map((p) => `
    <button class="rowitem" data-pid="${esc(p.patient_id)}">
      <span class="avatar">${esc(p.patient_id.slice(-2))}</span>
      <span class="grow">
        <span class="t1line"><span class="mono-id">${esc(p.patient_id)}</span>
          <span class="tag">${p.sex === "M" ? "男" : "女"} ${age(p.birth_date)}岁</span></span>
        <span class="sub">${p.n_records} 条化验记录 ·
          ${p.last_predicted_at ? "最近预测 " + esc(p.last_predicted_at.slice(0, 10)) : "尚未预测"}</span>
      </span>
      <svg class="chev" width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg>
    </button>`).join("");
  $$("#patientList .rowitem").forEach((b) =>
    b.addEventListener("click", () => selectPatient(b.dataset.pid, true)));
}
$("#patientSearch").addEventListener("input", renderPatientList);

function age(birth) {
  const d = new Date(birth);
  return Math.max(0, Math.floor((Date.now() - d.getTime()) / 31557600000));
}

async function selectPatient(pid, jump) {
  cancelAutoJump();               // 切换患者必须终止上一位的自动评估倒计时
  state.pid = pid;
  state.patient = state.patients.find((p) => p.patient_id === pid) || null;
  state.trend = null; state.predicted = false; state.followedUp = false;
  state.reports = null; state.timeline = null; state.riskTimeline = null;
  state.pending = []; state.lastPredict = null; state.batchCount = 0;
  store.set("drp.pid", pid);

  renderCtx();
  renderPending();
  $("#predictOut").hidden = true;
  $("#sec-factors").hidden = true;
  $("#sec-referral").hidden = true;
  $("#parseOut").hidden = true;
  $("#autoCompareBox").hidden = true;
  renderFollowupPlain();

  await loadRecords();
  await loadReports();
  await loadTimeline();
  await loadTraces();
  updateSteps();
  if (jump) go("work");
}

function renderCtx() {
  const p = state.patient;
  if (!p) {
    $("#ctxAvatar").textContent = "—";
    $("#ctxId").textContent = "未选择患者";
    $("#ctxSub").textContent = "先在「患者」里选一位，再开始评估";
    return;
  }
  $("#ctxAvatar").textContent = p.patient_id.slice(-2);
  $("#ctxId").textContent = p.patient_id;
  $("#ctxSub").textContent =
    `${p.sex === "M" ? "男" : "女"} · ${age(p.birth_date)} 岁 · ${p.n_records} 条记录`;
}

/* ---------------- 新建患者 ---------------- */
$("#btnNewPatient").addEventListener("click", () => $("#dlgPatient").showModal());
$("#npCancel").addEventListener("click", () => $("#dlgPatient").close());
$("#npOk").addEventListener("click", async () => {
  const body = {
    patient_id: $("#npId").value.trim(),
    sex: $("#npSex").value,
    birth_date: $("#npBirth").value,
  };
  if (!body.patient_id || !body.birth_date) return toast("编号和出生日期都要填", true);
  try {
    await api("/patients", { method: "POST", body });
    $("#dlgPatient").close();
    $("#npId").value = "";
    await refreshPatients();
    await selectPatient(body.patient_id, true);
    toast("档案已创建，接着录入化验单");
  } catch (e) { toast(e.message, true); }
});

/* ---------------- 签名元件：参考区间带 ---------------- */
/** 把一条化验值映射到"区间内/外多远"。区间缺一侧时用另一侧推一个可视量程。 */
function bandGeometry(low, high, values) {
  if (low == null && high == null) return null;
  const lo = low != null ? low : high * 0.6;
  const hi = high != null ? high : low * 1.6;
  const span = Math.max(hi - lo, Math.abs(hi) * 0.05, 1e-6);
  let min = lo - span * 0.85, max = hi + span * 0.85;
  values.filter(Number.isFinite).forEach((v) => {
    min = Math.min(min, v - span * 0.25);
    max = Math.max(max, v + span * 0.25);
  });
  return {
    lo, hi, low, high,
    at: (v) => Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100)),
  };
}

const gradeClass = (g) =>
  g == null ? "gx" : (Math.abs(g) >= 3 ? "g3" : Math.abs(g) === 2 ? "g2" : Math.abs(g) === 1 ? "g1" : "g0");
const gradeTag = (g) =>
  g == null ? "" : `<span class="tag ${["t1", "t2", "t3", "t4"][Math.min(Math.abs(g), 3)]}">${
    GRADE_CN[String(g)] || "—"}</span>`;

/**
 * 一行参考区间带。
 * @param {{name,code,unit,value,prev,low,high,grade,extra}} o
 */
function bandRow(o) {
  const geo = bandGeometry(o.low, o.high, [o.value, o.prev]);
  const head = `
    <div class="head">
      <span class="name">${esc(o.name)}</span>
      <span class="code">${esc(o.code)}</span>
      <span class="val">${num(o.value)}<small>${esc(o.unit || "")}</small></span>
    </div>`;
  if (!geo) {
    return `<div class="band ${gradeClass(o.grade)}">${head}
      <div class="muted" style="margin-top:4px">该指标无适用参考区间${o.extra || ""}</div></div>`;
  }
  const p = geo.at(o.value);
  const refL = geo.at(geo.lo), refR = geo.at(geo.hi);
  const prevP = Number.isFinite(o.prev) ? geo.at(o.prev) : null;
  const linkL = prevP == null ? 0 : Math.min(prevP, p);
  const linkW = prevP == null ? 0 : Math.abs(p - prevP);
  return `
    <div class="band ${gradeClass(o.grade)}">
      ${head}
      <div class="track">
        <span class="ref" style="left:${refL}%;width:${Math.max(refR - refL, 1)}%"></span>
        ${prevP == null ? "" : `<span class="link" style="left:${linkL}%;width:${linkW}%"></span>
        <span class="prev" style="left:${prevP}%" title="上次 ${num(o.prev)}"></span>`}
        <span class="dot" style="left:${p}%"></span>
      </div>
      <div class="ticks">
        <span>${o.low != null ? num(o.low) : ""}</span>
        <span>${gradeTag(o.grade)}${o.extra || ""}</span>
        <span>${o.high != null ? num(o.high) : ""}</span>
      </div>
    </div>`;
}

/* ---------------- 化验记录 ---------------- */
async function loadRecords() {
  if (!state.pid) return;
  state.records = await api(`/patients/${encodeURIComponent(state.pid)}/records`);
  state.refMap = {};
  state.records.forEach((r) => {
    state.refMap[r.indicator_code] = {
      low: r.ref_low, high: r.ref_high, unit: r.unit, name: r.name_cn || r.indicator_code,
    };
  });
  renderRecords();
}

function renderRecords() {
  const box = $("#recordList"), note = $("#recordsNote");
  if (!state.records.length) {
    note.textContent = "";
    box.innerHTML = `<div class="empty"><b>还没有化验记录</b>在上面粘贴一份报告文本，或点「填充示例」试一次</div>`;
    return;
  }
  // 按指标分组，只展示每项最近一次（附上一次做位移），比逐行罗列可读得多
  const byCode = {};
  state.records.forEach((r) => (byCode[r.indicator_code] ||= []).push(r));
  const items = Object.entries(byCode).map(([code, rows]) => {
    rows.sort((a, b) => String(a.measured_at).localeCompare(String(b.measured_at)));
    const last = rows[rows.length - 1], prev = rows.length > 1 ? rows[rows.length - 2] : null;
    return { code, last, prev, rows };
  });
  // 异常在前：抓眼睛的顺序要和临床优先级一致
  items.sort((a, b) => Math.abs(b.last.grade ?? 0) - Math.abs(a.last.grade ?? 0)
    || a.code.localeCompare(b.code));

  const dates = [...new Set(state.records.map((r) => String(r.measured_at).slice(0, 10)))].sort();
  note.textContent = `${items.length} 项指标 · ${dates.length} 次检验 · 最近 ${dates[dates.length - 1]}`;

  box.innerHTML = items.map((it) => bandRow({
    name: it.last.name_cn || it.code, code: it.code, unit: it.last.unit,
    value: it.last.value, prev: it.prev ? it.prev.value : null,
    low: it.last.ref_low, high: it.last.ref_high, grade: it.last.grade,
    extra: it.last.status === 3 ? ` <span class="tag t4">数据无效</span>` : "",
  })).join("");
}

/* ---------------- 报告解析与入库（真实检查日期随每份报告走） ---------------- */
async function parseAndIngestText(text, measured_at) {
  if (!state.pid) {
    toast("先在「患者」里选一位", true);
    return null;
  }
  if (!measured_at) {
    toast("请先填写这份报告的检查日期", true);
    return null;
  }
  const r = await api("/reports/parse", {
    method: "POST",
    body: { patient_id: state.pid, text: text, measured_at: measured_at },
  });
  renderParse(r);
  await refreshPatients();
  state.patient = state.patients.find((p) => p.patient_id === state.pid);
  renderCtx();
  await loadRecords();
  await loadReports();
  await loadTimeline();
  state.trend = null; state.riskTimeline = null;
  updateSteps();
  return r;
}

/* ---------------- 批量上传 → OCR → 检查日期确认 → 入库 ---------------- */
$("#btnOCR").addEventListener("click", () => { $("#repImage").click(); });
$("#repImage").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  if (!files.length) return;
  if (!state.pid) { toast("先在「患者」里选一位", true); $("#repImage").value = ""; return; }

  const btn = $("#btnOCR");
  btn.disabled = true;
  let done = 0;
  for (const file of files) {
    done += 1;
    $("#parseStat").innerHTML =
      `<span class="spinner"></span> 正在识别第 ${done}/${files.length} 份：${esc(file.name)}…`;
    try {
      const b64 = await new Promise((ok, fail) => {
        const reader = new FileReader();
        reader.onload = () => ok(reader.result);
        reader.onerror = fail;
        reader.readAsDataURL(file);
      });
      const ocr = await api("/ocr", { method: "POST", body: { image: b64 } });
      if (!ocr.text || !ocr.text.trim()) {
        toast(`「${file.name}」未识别到文字，已跳过`, true);
        continue;
      }
      state.pending.push({
        id: "p" + Date.now() + "_" + Math.random().toString(36).slice(2, 7),
        name: file.name,
        text: ocr.text,
        date: ocr.detected_date || "",
        detected: !!ocr.detected_date,
        src: ocr.detected_date_source || "",
        nlines: ocr.count || 0,
        rot: ocr.rotation || 0,
        layout: ocr.layout || "single",
        nred: ocr.n_redacted || 0,
      });
      renderPending();
    } catch (err) {
      toast(`「${file.name}」识别失败：${err.message}`, true);
    }
  }
  $("#parseStat").textContent = "";
  btn.disabled = false;
  $("#repImage").value = "";
  if (state.pending.length) {
    const nAuto = state.pending.filter((p) => p.detected).length;
    toast(`已识别 ${state.pending.length} 份报告，其中 ${nAuto} 份自动识别到检查日期，请确认后入库`);
  }
});

function renderPending() {
  const wrap = $("#pendingWrap"), box = $("#pendingList");
  if (!state.pending.length) { wrap.hidden = true; box.innerHTML = ""; return; }
  wrap.hidden = false;
  $("#pendingCount").textContent = `${state.pending.length} 份待确认`;
  box.innerHTML = state.pending.map((p) => `
    <div class="pend" data-pend="${p.id}">
      <div class="pend-head">
        <span class="pend-name" title="${esc(p.name)}">${esc(p.name)}</span>
        ${p.rot ? `<span class="tag cool">已自动转正 ${p.rot}°</span>` : ""}
        ${p.layout === "two_panel" ? `<span class="tag line">双栏已逐行拆分</span>` : ""}
        ${p.nred ? `<span class="tag line">已脱敏 ${p.nred} 处</span>` : ""}
        <button class="btn ghost sm" data-pv="${p.id}">识别预览</button>
        <span class="muted">识别 ${p.nlines} 行文本</span>
      </div>
      <div class="pend-date">
        ${p.detected
          ? `<span class="tag t1">已识别检查日期</span><span class="muted" style="font-size:11.5px">${esc(p.src)}</span>`
          : `<span class="tag t2">未识别到日期，请填写</span>`}
        <input type="date" class="pend-input" value="${esc(p.date)}">
        <span style="flex:1"></span>
        <button class="btn sm" data-ok="${p.id}">确认入库</button>
        <button class="btn ghost sm" data-rm="${p.id}">移除</button>
      </div>
    </div>`).join("");
  $$("#pendingList [data-pv]").forEach((b) =>
    b.addEventListener("click", () => {
      const it = state.pending.find((p) => p.id === b.dataset.pv);
      if (!it) return;
      $("#dlgReportMeta").textContent =
        `${it.name} · 识别 ${it.nlines} 行` +
        (it.rot ? ` · 已自动转正 ${it.rot}°` : "") +
        (it.layout === "two_panel" ? " · 双栏已拆分" : "");
      $("#dlgReportText").textContent = it.text;
      $("#dlgReport").showModal();
    }));
  $$("#pendingList [data-ok]").forEach((b) =>
    b.addEventListener("click", () => confirmPending(b.dataset.ok)));
  $$("#pendingList [data-rm]").forEach((b) =>
    b.addEventListener("click", () => {
      state.pending = state.pending.filter((p) => p.id !== b.dataset.rm);
      renderPending();
    }));
  $$("#pendingList .pend-input").forEach((inp) =>
    inp.addEventListener("change", () => {
      const it = state.pending.find((p) => p.id === inp.closest(".pend").dataset.pend);
      if (it) { it.date = inp.value; it.detected = false; it.src = "手动填写"; }
    }));
}

async function confirmPending(id, silent = false) {
  const it = state.pending.find((p) => p.id === id);
  if (!it) return false;
  if (!it.date) { if (!silent) toast(`「${it.name}」还没有检查日期`, true); return false; }
  try {
    const r = await parseAndIngestText(it.text, it.date);
    if (!r) return false;
    state.batchCount = (state.batchCount || 0) + 1;
    state.pending = state.pending.filter((p) => p.id !== id);
    renderPending();
    // V3.1 用户反馈："第二张还是 17 条，不知道是不是只识别到一张"。
    // 每次入库都把【本份 + 累计】一起说清楚。
    const sum = state.reports?.summary || {};
    if (!silent)
      toast(`「${it.name}」已入库 ${r.stored} 项（${it.date}）· ` +
            `累计 ${sum.n_reports ?? "?"} 份 / ${sum.n_stored_total ?? "?"} 条指标`);
    if (r.stored === 0)
      toast(`「${it.name}」未识别到可入库的指标，可点「查看原文」核对识别结果`, true);
    if (!state.pending.length) {
      const n = state.batchCount; state.batchCount = 0;
      await afterBatchIngest(n);
    }
    return true;
  } catch (e) { toast(`「${it.name}」入库失败：${e.message}`, true); return false; }
}

$("#btnConfirmAll").addEventListener("click", async () => {
  const missing = state.pending.filter((p) => !p.date);
  if (missing.length)
    return toast(`还有 ${missing.length} 份未填写检查日期`, true);
  const btn = $("#btnConfirmAll");
  btn.disabled = true;
  let ok = 0;
  for (const it of [...state.pending]) {
    $("#parseStat").innerHTML =
      `<span class="spinner"></span> 正在入库：${esc(it.name)}…`;
    if (await confirmPending(it.id, true)) ok += 1;
  }
  $("#parseStat").textContent = "";
  btn.disabled = false;
  const sum = state.reports?.summary || {};
  toast(`批量入库完成：本次 ${ok} 份 · 累计 ${sum.n_reports ?? "?"} 份报告 / ${sum.n_stored_total ?? "?"} 条指标`);
});

/** 批量导入完成后的收尾（V3.1 用户反馈："先要趋势，然后自动跳转到评估"）：
    ① 跳到趋势页，让用户先看到按真实检查日期画出的指标曲线；
    ② 顶部横幅倒计时，自动进入评估页并运行预测（可点「留在本页」取消，
       任何手动切页也会取消 —— 自动流转不许和用户抢方向盘）。 */
async function afterBatchIngest(nNew) {
  cancelAutoJump();
  go("trend");
  await loadTrend(true);
  const sum = state.reports?.summary || {};
  const banner = $("#trendAutoBanner");
  let left = 8;
  const text = () =>
    `<b>✓ 本次入库 ${nNew} 份</b> · 累计 ${sum.n_reports ?? "—"} 份报告 / ` +
    `${sum.n_stored_total ?? "—"} 条指标。先看看各指标的历史趋势，` +
    `<b>${left}</b> 秒后自动进入风险评估。` +
    `<span class="tb-ops"><button class="btn sm" id="tbGoNow">立即评估</button>` +
    `<button class="btn ghost sm" id="tbStay">留在本页</button></span>`;
  banner.hidden = false;
  banner.innerHTML = text();
  const bind = () => {
    $("#tbGoNow").onclick = () => runAutoAssess();
    $("#tbStay").onclick = () => cancelAutoJump(true);
  };
  bind();
  state.autoTimer = setInterval(() => {
    left -= 1;
    if (left <= 0) return runAutoAssess();
    banner.innerHTML = text();
    bind();
  }, 1000);
}

function cancelAutoJump(byUser = false) {
  if (state.autoTimer) { clearInterval(state.autoTimer); state.autoTimer = null; }
  const b = $("#trendAutoBanner");
  if (b) {
    if (byUser) {
      b.innerHTML = `已留在趋势页。看完随时可去评估页点「运行纵向风险分析」。`;
      setTimeout(() => { b.hidden = true; }, 4000);
    } else b.hidden = true;
  }
}

async function runAutoAssess() {
  cancelAutoJump();
  state._autoNav = true;
  go("work");
  state._autoNav = false;
  $("#sec-predict").scrollIntoView({ behavior: "smooth", block: "start" });
  await doPredict({ fromAuto: true });
}

/* ---------------- 填充示例 ---------------- */
$("#btnSample").addEventListener("click", async () => {
  $("#repText").value = state.meta.sample_report;
  $("#detailsManualText").open = true;
  if (!$("#repDate").value) $("#repDate").value = new Date().toISOString().slice(0, 10);
  if (state.pid) {
    $("#parseStat").innerHTML = `<span class="spinner"></span> 解析中…`;
    try {
      const r = await parseAndIngestText(state.meta.sample_report, $("#repDate").value);
      $("#parseStat").textContent = "";
      if (r) toast(`示例已入库 ${r.stored} 条`);
    } catch (e) {
      $("#parseStat").textContent = "";
      toast(e.message, true);
    }
  }
});

/* ---------------- 手动点击解析 ---------------- */
$("#btnParse").addEventListener("click", async () => {
  const btn = $("#btnParse");
  const d = $("#repDate").value;
  if (!d) return toast("请先填写这份报告的检查日期", true);
  btn.disabled = true;
  $("#parseStat").innerHTML = `<span class="spinner"></span> 解析中…`;
  try {
    const r = await parseAndIngestText($("#repText").value, d);
    $("#parseStat").textContent = "";
    if (r) toast(`入库 ${r.stored} 条（检查日期 ${d}）`);
  } catch (e) {
    $("#parseStat").textContent = "";
    toast(e.message, true);
  } finally { btn.disabled = false; }
});

/* ---------------- 我的历史检查资料（逐份可见 · 可管理） ---------------- */
async function loadReports() {
  if (!state.pid) return;
  state.reports = await api(`/patients/${encodeURIComponent(state.pid)}/reports`);
  renderReports();
}

const ym = (d) => d ? String(d).slice(0, 7).replace("-", ".") : "—";

function renderReports() {
  const box = $("#reportTable"), sumEl = $("#reportsSummary");
  const d = state.reports;
  if (!d || !d.reports.length) {
    sumEl.innerHTML = `还没有上传过报告`;
    box.innerHTML = "";
    return;
  }
  const s = d.summary;
  sumEl.innerHTML =
    `<b>已上传 ${s.n_reports} 份报告</b>｜${ym(s.first_date)}—${ym(s.last_date)}｜识别 ${s.n_stored_total} 条指标`;
  box.innerHTML = `
    <div class="rep-row rep-head">
      <span>检查日期</span><span>报告</span><span class="num">识别结果</span><span>状态</span><span></span>
    </div>` +
    d.reports.map((r) => `
    <div class="rep-row" data-rep="${r.id}">
      <span class="rep-date"><input type="date" value="${esc(String(r.measured_at).slice(0, 10))}"
            title="修改检查日期后自动保存"></span>
      <span class="rep-name mono">#${r.id}</span>
      <span class="num">${r.n_stored} 项</span>
      <span>${r.n_stored ? `<span class="tag t1">✓ 已入库</span>` : `<span class="tag t2">无有效指标</span>`}</span>
      <span class="rep-ops">
        <button class="mini" data-view="${r.id}">查看原文</button>
        <button class="mini" data-reparse="${r.id}">重新识别</button>
        <button class="mini danger" data-del="${r.id}">删除</button>
      </span>
    </div>`).join("");

  $$("#reportTable .rep-date input").forEach((inp) =>
    inp.addEventListener("change", async () => {
      const id = inp.closest(".rep-row").dataset.rep;
      if (!inp.value) return;
      try {
        await api(`/reports/${id}`, { method: "PATCH", body: { measured_at: inp.value } });
        toast(`报告 #${id} 检查日期已改为 ${inp.value}，趋势与时间轴同步更新`);
        state.trend = null; state.riskTimeline = null;
        await loadRecords(); await loadReports(); await loadTimeline();
      } catch (e) { toast(e.message, true); await loadReports(); }
    }));
  $$("#reportTable [data-view]").forEach((b) =>
    b.addEventListener("click", async () => {
      try {
        const r = await api(`/reports/${b.dataset.view}`);
        $("#dlgReportMeta").textContent =
          `报告 #${r.id} · 检查日期 ${String(r.measured_at).slice(0, 10)} · 上传于 ${String(r.created_at).slice(0, 10)}`;
        $("#dlgReportText").textContent = r.raw_text;
        $("#dlgReport").showModal();
      } catch (e) { toast(e.message, true); }
    }));
  $$("#reportTable [data-reparse]").forEach((b) =>
    b.addEventListener("click", async () => {
      b.disabled = true;
      try {
        const r = await api(`/reports/${b.dataset.reparse}/reparse`, { method: "POST", body: {} });
        toast(`报告 #${b.dataset.reparse} 已重新识别，入库 ${r.stored} 项`);
        state.trend = null; state.riskTimeline = null;
        await loadRecords(); await loadReports(); await loadTimeline();
      } catch (e) { toast(e.message, true); } finally { b.disabled = false; }
    }));
  $$("#reportTable [data-del]").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!confirm(`删除报告 #${b.dataset.del} 及其全部指标记录？此操作不可恢复。`)) return;
      try {
        await api(`/reports/${b.dataset.del}`, { method: "DELETE" });
        toast(`报告 #${b.dataset.del} 已删除`);
        state.trend = null; state.riskTimeline = null;
        await refreshPatients();
        state.patient = state.patients.find((p) => p.patient_id === state.pid);
        renderCtx();
        await loadRecords(); await loadReports(); await loadTimeline();
        updateSteps();
      } catch (e) { toast(e.message, true); }
    }));
}
$("#dlgReportClose").addEventListener("click", () => $("#dlgReport").close());

/* ---------------- 数据确认 · 健康时间轴 ---------------- */
async function loadTimeline() {
  if (!state.pid) return;
  state.timeline = await api(`/patients/${encodeURIComponent(state.pid)}/timeline`);
  renderTimeline();
  updateSteps();
}

function renderTimeline() {
  const t = state.timeline, box = $("#timelineBox");
  if (!t || !t.n_records) {
    box.innerHTML = `<div class="empty"><b>还没有可确认的数据</b>先在上面导入至少一份历史报告</div>`;
    return;
  }
  const groups = t.groups.map((g) => {
    const ok = g.n_timepoints >= 2;
    return `<div class="tlg ${ok ? "" : "warn"}">
      <span class="tlg-ic">${ok ? "✓" : "⚠"}</span>
      <span>${esc(g.group)}</span>
      <span class="muted">${ok ? `${g.n_timepoints} 个时间点` : `仅 ${g.n_timepoints} 个时间点`}</span>
    </div>`;
  }).join("");
  box.innerHTML = `
    <div class="tl-done"><b>已建立个人健康时间轴</b></div>
    <div class="stat-row">
      <div class="stat"><div class="v">${esc(t.span_label)}</div><div class="k">数据跨度</div></div>
      <div class="stat"><div class="v">${t.n_reports}</div><div class="k">检查报告</div></div>
      <div class="stat"><div class="v">${t.n_records}</div><div class="k">有效记录</div></div>
      <div class="stat"><div class="v">${t.n_comparable_indicators}</div><div class="k">可连续比较指标</div></div>
    </div>
    <div class="tlg-list">${groups}</div>
    <div class="divider"></div>
    ${t.longitudinal_ready
      ? `<p class="ready-line">✓ 数据已具备纵向趋势分析条件（${esc(t.first_date)} — ${esc(t.last_date)}）</p>
         <button class="btn block" id="btnGoPredict">开始风险预测</button>`
      : `<p class="ready-line warn">⚠ 目前只有 ${t.n_dates} 个检查时间点；补充不同日期的报告后才能做纵向趋势分析（单次数据仍可预测，但没有趋势特征）。</p>
         <button class="btn ghost block" id="btnGoPredict">仍要基于单次数据预测</button>`}`;
  $("#btnGoPredict").addEventListener("click", () =>
    $("#sec-predict").scrollIntoView({ behavior: "smooth", block: "start" }));
}

function renderParse(r) {
  $("#parseOut").hidden = false;
  // 过滤只展示成功匹配并入库的指标，不展示失败或噪声行
  const validRows = (r.rows || []).filter((row) => row.indicator_code && row.value != null);

  $("#parseStats").innerHTML = `
    <div class="stat"><div class="v" style="color:var(--primary)">${validRows.length}</div><div class="k">已成功入库指标</div></div>
    <div class="stat"><div class="v">${r.parse.n_lines || validRows.length}</div><div class="k">扫描文本行数</div></div>
  `;

  if (validRows.length === 0) {
    $("#parseRows").innerHTML = `<div class="muted" style="padding:12px">未识别到标准检验指标，请核对单据内容或重新拍摄。</div>`;
    return;
  }

  $("#parseRows").innerHTML = validRows.map((row) => {
    return `
      <div class="rowitem static">
        <span class="grow">
          <span class="t1line">
            <b style="font-size:14px;color:var(--text)">${esc(row.indicator_code)}</b>
            <span class="mono" style="font-size:14px;font-weight:600;margin-left:6px">${num(row.value)} ${esc(row.unit || "")}</span>
            <span class="tag t1" style="margin-left:8px">已入库</span>
          </span>
          <span class="raw" title="${esc(row.raw_line)}">${esc(row.raw_line)}</span>
        </span>
        <span class="num muted">${Math.round(row.confidence * 100)}%</span>
      </div>`;
  }).join("");
}

/* ---------------- 预测 ---------------- */
/** 概率 → 分层刻度上的位置。四层等宽、层内线性，切点用真实值标注。 */
function tierPos(p, cuts) {
  const n = cuts.length + 1;
  let i = 0;
  while (i < cuts.length && p >= cuts[i]) i++;
  const lo = i === 0 ? 0 : cuts[i - 1];
  const hi = i === cuts.length ? 1 : cuts[i];
  const inner = hi > lo ? (p - lo) / (hi - lo) : 0;
  return { idx: i, pct: ((i + Math.max(0, Math.min(1, inner))) / n) * 100 };
}

/* 本次最关注的问题（改动 2）：单选 chip，随预测请求提交 */
$$("#concernChips .chip").forEach((b) => b.addEventListener("click", () => {
  state.concern = b.dataset.concern;
  $$("#concernChips .chip").forEach((x) => x.classList.toggle("on", x === b));
}));

/** 取"上一次预测"的 3 年风险概率（用于本次预测后的自动比较）。
    来源优先级：本会话里刚做过的预测 → 审计走势的最后一个点。 */
async function getPrevRiskProb() {
  const fromLast = state.lastPredict?.results?.find((x) => x.horizon === "3y");
  if (fromLast) return fromLast.probability;
  if (!state.patient?.last_predicted_at) return null;
  try {
    const t = state.trend ||
      (await api(`/patients/${encodeURIComponent(state.pid)}/trend`));
    state.trend = t;
    const pts = t.risk_trajectories?.["3y"]?.points;
    if (pts && pts.length) return pts[pts.length - 1].probability;
  } catch { /* 拿不到就不比较，预测本身照常 */ }
  return null;
}

/** 运行一次风险预测（改动 V3.1：手动点击与"趋势页倒计时自动评估"共用）。
    fromAuto 只影响文案；比较逻辑：只要存在上一次预测就自动给出 改善/稳定/升高。 */
async function doPredict({ fromAuto = false } = {}) {
  if (!state.pid) { toast("先在「患者」里选一位", true); return; }
  const btn = $("#btnPredict");
  btn.disabled = true;
  $("#predStat").innerHTML = `<span class="spinner"></span> 读取全部历史数据 → 时序特征管线 → 多时程推理中…`;
  try {
    const prevProb = await getPrevRiskProb();   // 必须在预测前取，否则比到自己
    const r = await api("/predict", {
      method: "POST", body: { patient_id: state.pid, concern: state.concern },
    });
    $("#predStat").textContent = "";
    renderPredict(r);
    state.predicted = true;
    state.riskTimeline = null;
    await loadTraces();
    await refreshPatients();
    state.patient = state.patients.find((p) => p.patient_id === state.pid);
    if (prevProb != null) renderAutoCompare(prevProb, r);
    renderFollowupPlain();
    renderLifestyle().catch(() => {});   // ④ 摘要异步补上，不阻塞预测结果展示
    updateSteps();
    if (fromAuto) toast("已基于全部历史数据完成本次风险评估");
  } catch (e) {
    $("#predStat").textContent = "";
    toast(e.message, true);
  } finally { btn.disabled = false; }
}

$("#btnPredict").addEventListener("click", () => doPredict());

const RANK_TONE = { 1: "t4", 2: "t3", 3: "t2" };

function renderOverview(r) {
  // ① 先回答用户主动关注的问题
  const ca = r.concern_answer, caBox = $("#concernAnswerBox");
  if (ca) {
    caBox.hidden = false;
    caBox.className = "concern-answer " + (ca.status === "abnormal" ? "warn" : "ok");
    caBox.innerHTML = `<b>你关注的「${esc(r.concern_label)}」：</b>${esc(ca.text)}`;
  } else caBox.hidden = true;

  // ② 风险优先级排序（改动 3）：首要/第二/第三关注 + 为什么排这里
  const ov = r.risk_overview || { items: [], stable_groups: [] };
  const items = ov.items.map((it) => {
    const inds = it.indicators.map((x) => {
      const trend = x.trend
        ? `<span class="tag ${x.worsened ? "t3" : "line"}" style="margin-left:6px">${esc(x.trend)}</span>`
        : "";
      return `<li>${esc(x.name_cn)} <b class="mono">${num(x.value)}${esc(x.unit)}</b>
        ${gradeTag(x.grade)}${trend}</li>`;
    }).join("");
    return `<div class="ov-item p${Math.min(it.rank, 4)}">
      <div class="ov-head">
        <span class="tag ${RANK_TONE[it.rank] || "line"}">${esc(it.rank_label)}</span>
        <b>${esc(it.group)}相关异常</b>
        <span class="muted">→ ${esc(it.department)} · ${esc(it.priority_label)}</span>
      </div>
      <div class="muted" style="margin:2px 0 6px">为什么排在这里：${esc(it.why)}</div>
      <ul class="ov-inds">${inds}</ul>
      ${it.direction ? `<div class="ov-dir">若长期未干预，可能向<b>${esc(it.direction)}</b>方向发展
        <span class="muted">（风险提示，非诊断）</span></div>` : ""}
    </div>`;
  }).join("");
  const stable = (ov.stable_groups && ov.stable_groups.length)
    ? `<div class="ov-stable">目前相对稳定：${ov.stable_groups.map(esc).join("、")}</div>` : "";
  $("#riskOverviewBox").innerHTML = items ||
    `<div class="empty"><b>本次未发现需要重点关注的异常</b>保持定期体检即可</div>`;
  $("#riskOverviewBox").innerHTML += stable;
}

function selectHorizon(h) {
  const r = state.lastPredict;
  if (!r) return;
  const main = r.results.find((x) => x.horizon === h) || r.results[r.results.length - 1];
  state.selHorizon = main.horizon;
  const tiers = (state.meta.tiers || {})[main.horizon];
  const cuts = tiers ? tiers.cutpoints : [0.05, 0.15, 0.4];
  const names = tiers ? tiers.names : ["低危", "中危", "高危", "极高危"];
  const pos = tierPos(main.probability, cuts);
  const color = tierColor(main.risk_tier);

  const hCn = H_LABEL[main.horizon] || main.horizon;
  $("#heroHorizon").textContent = `未来 ${hCn}内 · ${state.lastPredict.prediction_context?.endpoint_label || "综合健康风险"}`;
  $("#heroProb").innerHTML = `${(main.probability * 100).toFixed(1)}<span>%</span>`;
  $("#heroProb").style.color = color;
  $("#heroTier").innerHTML =
    `<span class="tag t${TIER_IDX[main.risk_tier] || 1}">${esc(main.risk_tier)}</span>
     <span class="muted mono">${esc(main.trace_id.slice(0, 12))}…</span>
     ${main.degraded ? `<span class="tag t2">降级输出</span>` : ""}`;

  // 分层刻度：段色只点亮到当前层，切点写真实数值
  $("#tierScale").innerHTML = `
    <div class="rail2"><span class="pin" style="left:${pos.pct}%">${pct1(main.probability)}</span></div>
    <div class="segs">${names.map((_, i) =>
      `<span class="seg ${i <= pos.idx ? "on" + (i + 1) : ""}"></span>`).join("")}</div>
    <div class="cuts">${names.map((nm, i) =>
      `<span>${esc(nm)}${i < cuts.length ? `<br>&lt; ${(cuts[i] * 100).toFixed(0)}%` : ""}</span>`).join("")}</div>`;

  $("#modelJudge").innerHTML =
    `模型判断：未来 <b>${esc(hCn)}</b>内属于「<b style="color:${color}">${esc(main.risk_tier)}</b>」风险区间`;

  // V3.1 用户反馈"60.2 看不懂"：把概率翻译成频数表述，认知负担最低。
  const n = Math.round(main.probability * 100);
  $("#heroPlain").innerHTML =
    `怎么理解这个数：100 位指标情况与你相近的人里，模型估计约 <b>${n} 位</b>` +
    `会在未来${esc(hCn)}内出现上述目标风险事件，约 ${100 - n} 位不会。` +
    `这是概率估计，不是对某个人的确定结论。`;

  // 三个时间窗口做成可点行（改动 4）
  $("#horizonMini").innerHTML = r.results.map((x) => {
    const c = (state.meta.tiers || {})[x.horizon];
    const p2 = tierPos(x.probability, c ? c.cutpoints : cuts);
    return `<div class="hmini sel ${x.horizon === main.horizon ? "on" : ""}" data-h="${esc(x.horizon)}"
                 title="点击查看该时间窗口的风险因素与解释">
      <span class="hl">未来${esc(H_LABEL[x.horizon] || x.horizon)}</span>
      <span class="bar"><i style="width:${p2.pct}%;background:${tierColor(x.risk_tier)}"></i></span>
      <span class="pv" style="color:${tierColor(x.risk_tier)}">${pct1(x.probability)}</span>
      <span class="tag t${TIER_IDX[x.risk_tier] || 1}">${esc(x.risk_tier)}</span>
    </div>`;
  }).join("");
  $$("#horizonMini .hmini").forEach((el) =>
    el.addEventListener("click", () => selectHorizon(el.dataset.h)));

  $("#factorHorizonNote").textContent =
    `当前展示：未来 ${hCn}窗口的指标贡献（点击上方时间窗口可切换）`;
  renderFactors(main);
  $("#narrative").textContent = main.narrative;
}

function renderPredict(r) {
  state.lastPredict = r;
  $("#predictOut").hidden = false;

  // 演示模型标识（核查项 4：不能包装成临床验证的疾病发生概率）
  const ctx = r.prediction_context || {};
  const banner = $("#demoBanner");
  if (ctx.development_only) {
    banner.hidden = false;
    banner.textContent = "⚠ " + (ctx.development_note ||
      "当前为演示模型：概率为统计演示值，未经过真实临床队列验证。");
  } else banner.hidden = true;

  renderOverview(r);

  $("#endpointLabel").innerHTML =
    `预测目标：<b>${esc(ctx.endpoint_label || "综合健康风险进展")}</b>` +
    (ctx.endpoint_detail ? `<span class="muted" style="display:block;margin-top:2px">${esc(ctx.endpoint_detail)}</span>` : "");

  $("#predBasis").textContent = ctx.n_reports != null
    ? `预测依据：${ym(ctx.first_date)}—${ym(ctx.last_date)} 期间 ${ctx.n_reports} 份检查报告 · ` +
      `${ctx.n_dates} 个检查时间点 · 共 ${ctx.n_comparable_indicators} 项可连续比较指标（有效记录 ${ctx.n_records} 条）`
    : "";

  selectHorizon(state.selHorizon || "3y");

  $("#servedBy").textContent =
    `服务版本 ${r.model_version}（${r.arm === "canary" ? "灰度臂" : "全量臂"}）· ` +
    `每个时程一条独立 trace，已写入全链路日志`;

  renderReferral(r.referral);
  $("#monotonicNote").textContent = r.monotonic_note || "";
}

function renderFactors(main) {
  const box = $("#sec-factors");
  const fac = main.top_factors || [];
  if (!fac.length) { box.hidden = true; return; }
  box.hidden = false;
  const max = Math.max(...fac.map((f) => f.magnitude || 0), 1e-9);
  $("#factorList").innerHTML = fac.map((f) => {
    const w = Math.max(2, (f.magnitude / max) * 46);
    let bar;
    if (f.is_missing) bar = `<i class="na" style="width:24px"></i>`;
    else if (f.direction >= 0) bar = `<i class="up" style="width:${w}%"></i>`;
    else bar = `<i class="down" style="width:${w}%"></i>`;
    return `<div class="factor">
      <div class="fn">${esc(f.display)}${f.is_missing ? "<em>本次未检查</em>" : ""}</div>
      <div class="fbar">${bar}</div>
    </div>`;
  }).join("");
}

function renderReferral(ref) {
  const sec = $("#sec-referral"), box = $("#referralBox");
  sec.hidden = false;
  if (!ref.items.length && !ref.general_note) {
    box.innerHTML = `<div class="empty"><b>各项指标都在参考区间内</b>保持定期体检即可</div>`;
    return;
  }
  box.innerHTML = ref.items.map((it) => `
    <div class="advice p${it.priority}">
      <div><span class="dept">${esc(it.department)}</span>
        <span class="tag">${esc(it.priority_label)}</span>
        <span class="tag line">${esc(it.group)}</span></div>
      <ul>${it.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <div class="checkups">建议检查：${it.checkups.map(esc).join("、")}</div>
    </div>`).join("") +
    (ref.general_note ? `<div class="prose">${esc(ref.general_note)}</div>` : "");
}

/* ---------------- 随访回流 ---------------- */
async function loadTraces() {
  const sel = $("#fbTrace");
  if (!state.pid) { sel.innerHTML = `<option value="">（先选择患者）</option>`; return; }
  state.trend = await api(`/patients/${encodeURIComponent(state.pid)}/trend`);
  const opts = [];
  for (const [h, traj] of Object.entries(state.trend.risk_trajectories)) {
    for (const p of traj.points) {
      opts.push(`<option value="${esc(p.trace_id)}">${H_LABEL[h] || h} · ${
        esc(p.at.slice(0, 16))} · ${pct1(p.probability)} ${esc(p.risk_tier)}</option>`);
    }
  }
  sel.innerHTML = opts.length ? opts.reverse().join("") : `<option value="">（暂无预测记录）</option>`;
}

$("#btnFeedback").addEventListener("click", async () => {
  const trace = $("#fbTrace").value;
  if (!trace) return toast("这位患者还没有可回填的预测", true);
  if (!$("#fbConsent").checked) return toast("未获授权的随访数据不能入库（规范 1.3）", true);
  try {
    await api("/feedback", { method: "POST", body: {
      trace_id: trace,
      event_occurred: $("#fbEvent").value === "1",
      days_since_prediction: +$("#fbDays").value,
      consented: true,
    } });
    state.followedUp = true;
    updateSteps();
    toast("随访结局已回流到样本库");
  } catch (e) { toast(e.message, true); }
});

/* ---------------- 后续健康跟踪（普通用户人话版，改动 8） ---------------- */
const RECHECK_BY_TIER = {
  "极高危": "建议 1 个月内复查", "高危": "建议 1~3 个月内复查",
  "中危": "建议 3~6 个月内复查", "低危": "建议 6~12 个月复查",
};

function renderFollowupPlain() {
  const box = $("#followupPlain");
  const lastAt = state.patient?.last_predicted_at;
  const r = state.lastPredict;
  if (!lastAt && !r) {
    box.innerHTML = `<div class="empty"><b>先完成一次风险预测</b>之后这里会给出建议复查时间与项目</div>`;
    return;
  }
  const main = r ? (r.results.find((x) => x.horizon === "3y") || r.results[0]) : null;
  const tier = main ? main.risk_tier : null;
  const checkups = r
    ? [...new Set((r.referral?.items || []).flatMap((it) => it.checkups))].slice(0, 8)
    : [];
  box.innerHTML = `
    <div class="fup-grid">
      <div class="fup-row"><span class="k">上次评估</span>
        <span class="v">${lastAt ? esc(String(lastAt).slice(0, 10)) : "本次会话"}${
          tier ? ` · <span class="tag t${TIER_IDX[tier] || 1}">${esc(tier)}</span>` : ""}</span></div>
      <div class="fup-row"><span class="k">建议复查时间</span>
        <span class="v">${tier ? esc(RECHECK_BY_TIER[tier] || "建议 6~12 个月复查") : "完成一次预测后给出"}</span></div>
      <div class="fup-row"><span class="k">需要复查的项目</span>
        <span class="v">${checkups.length ? checkups.map(esc).join("、") : "以「就医建议」卡片为准"}</span></div>
    </div>
    <p class="muted" style="margin:10px 0 0">到时候把新报告传进来，系统会自动和这次比较，告诉你是
      <b>改善 / 稳定 / 升高</b>。</p>`;
}

/** 上传新报告后自动比较（改动 8）：跑一次新预测，与上一次的 3 年风险比。 */
/** 本次 vs 上次评估的自动比较横幅。±2 个百分点内视为稳定。
    prevProb 由 doPredict 在发起预测【之前】捕获，避免"和自己比"。 */
function renderAutoCompare(prevProb, r) {
  const curr = r.results.find((x) => x.horizon === "3y") || r.results[0];
  const dpp = (curr.probability - prevProb) * 100;
  let verdict, cls;
  if (dpp <= -2) { verdict = "改善"; cls = "good"; }
  else if (dpp >= 2) { verdict = "升高"; cls = "bad"; }
  else { verdict = "稳定"; cls = "flat"; }
  const box = $("#autoCompareBox");
  box.hidden = false;
  box.className = "autocompare " + cls;
  box.innerHTML =
    `自动比较：3 年风险 ${pct1(prevProb)}（上次评估）` +
    ` → ${pct1(curr.probability)}（本次），<b>${verdict}</b>` +
    `（变化 ${dpp > 0 ? "+" : ""}${dpp.toFixed(1)} 个百分点）`;
  toast(`与上次评估相比：风险${verdict}`);
}

/** ④ 生活方式建议摘要（V3.1 用户反馈："除了就医建议还应有运动、饮食"）。
    内容取自趋势页 AI 深度分析的饮食/运动章节 —— 同一份数据只生成一次，
    预测页给摘要，完整方案仍在「趋势 → AI 深度分析」。AI 只解释，不造概率。 */
async function renderLifestyle() {
  const box = $("#lifestyleBox");
  if (!box || !state.pid) return;
  let t = state.trend;
  if (!t) {
    try {
      t = await api(`/patients/${encodeURIComponent(state.pid)}/trend`);
      state.trend = t;
    } catch { box.innerHTML = ""; return; }
  }
  const ai = t.ai_analysis;
  const pick = (arr, nGroups, nItems) => (arr || []).slice(0, nGroups)
    .map((g) => ({ title: g.title, items: (g.items || []).slice(0, nItems) }))
    .filter((g) => g.items.length);
  const diet = pick(ai?.diet_interventions, 2, 2);
  const life = pick(ai?.lifestyle_interventions, 1, 3);
  const cell = (icon, title, groups) => !groups.length ? "" : `
    <div class="ls-cell">
      <div class="ls-t">${icon} ${title}</div>
      ${groups.map((g) => `<div class="ls-g"><b>${esc(g.title)}</b>
        <ul>${g.items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul></div>`).join("")}
    </div>`;
  const inner = cell("🥗", "饮食要点", diet) + cell("🏃", "运动与生活方式", life);
  box.innerHTML = `
    <h3 class="blocktitle" style="margin-top:0">生活方式建议（摘要）</h3>
    ${inner ? `<div class="ls-grid">${inner}</div>`
            : `<p class="muted">累计更多检查数据后，这里会给出针对性的饮食与运动要点。</p>`}
    <button class="btn ghost sm" id="btnFullAI" style="margin-top:10px">
      查看完整 AI 深度分析（机制 / 膳食 / 运动 / 随访日程）→</button>`;
  $("#btnFullAI").onclick = () => {
    go("trend");
    setTimeout(() => $("#sec-trend-interventions")
      .scrollIntoView({ behavior: "smooth", block: "start" }), 80);
  };
}

/* ---------------- 步骤状态（4 步） ---------------- */
function updateSteps() {
  const hasRec = (state.patient?.n_records || 0) > 0;
  const tlReady = !!state.timeline?.longitudinal_ready;
  const hasPred = state.predicted || !!state.patient?.last_predicted_at;
  const set = (k, v) => {
    const el = $(`.step[data-step="${k}"]`);
    if (el) el.dataset.state = v;
  };
  set("ingest", hasRec ? "done" : "now");
  set("timeline", tlReady ? "done" : hasRec ? "now" : "idle");
  set("predict", hasPred ? "done" : hasRec ? "now" : "idle");
  set("followup", state.followedUp ? "done" : hasPred ? "now" : "idle");
}

/* ---------------- 趋势 ---------------- */
async function loadTrend(force = false) {
  if (!state.pid) return;
  if (force) { state.trend = null; state.riskTimeline = null; }
  const t = (!force && state.trend) ? state.trend : (await api(`/patients/${encodeURIComponent(state.pid)}/trend`));
  state.trend = t;

  await drawRiskTimeline();

  // 本次 vs 上次：标题写明确日期（改动 5），并画出整段检查时间轴
  const dates = [...new Set(state.records.map((r) => String(r.measured_at).slice(0, 10)))].sort();
  const dot = (d) => d ? d.replaceAll("-", ".") : "—";
  $("#compareDates").textContent = dates.length >= 2
    ? `本次 ${dot(dates[dates.length - 1])} vs 上次 ${dot(dates[dates.length - 2])}`
    : "超过临床变化阈值才判为真实变化";
  renderExamStrip(dates);

  const cbox = $("#compareList");
  $("#compareEmpty").hidden = !!t.comparisons.length;
  cbox.innerHTML = t.comparisons.map((c) => {
    const ref = state.refMap[c.code] || {};
    const verdict = c.is_real_change
      ? `<span class="tag ${c.worsened ? "t3" : "cool"}">${esc(c.direction)}${c.worsened ? " · 加重" : ""}</span>`
      : `<span class="tag line">RCV 内 · 视为平稳</span>`;
    const d = `<span class="mono" style="margin-left:6px">${c.delta > 0 ? "+" : ""}${
      num(c.delta)}${c.delta_pct != null ? `（${(c.delta_pct * 100).toFixed(0)}%）` : ""}</span>`;
    const pair = `<span class="muted" style="margin-left:6px;font-size:11px">${
      esc(String(c.prev_at).slice(0, 10))} → ${esc(String(c.curr_at).slice(0, 10))}</span>`;
    return bandRow({
      name: c.name_cn, code: c.code, unit: c.unit, value: c.curr_value, prev: c.prev_value,
      low: ref.low, high: ref.high, grade: c.curr_grade, extra: " " + verdict + d + pair,
    });
  }).join("");

  // 指标历史趋势：全部真实检查时间点 + 时间范围筛选
  const sel = $("#seriesSel");
  sel.innerHTML = t.series.map((s, i) => `<option value="${i}">${esc(s.name_cn)}</option>`).join("");
  sel.onchange = () => drawSeries(t.series[+sel.value]);
  if (t.series.length) drawSeries(t.series[0]);
  $("#trendText").textContent = t.rendered_text;
  renderAITop3(t);
  renderAIAdvisor(t);
}

/* 风险走势（按真实检查日期回溯）：历史实线 + 未来预测虚线（改动 5/6）。
   审计走势（按预测发生时间）保留给管理台复盘，不再作为用户侧曲线 ——
   它在同一天连传 8 份报告时只会画出一条竖线/平线，正是本次要修的问题。 */
async function drawRiskTimeline() {
  if (!state.riskTimeline) {
    try {
      state.riskTimeline = await api(`/patients/${encodeURIComponent(state.pid)}/risk-timeline`);
    } catch (e) { toast(e.message, true); return; }
  }
  const rt = state.riskTimeline;
  const pts = rt.points || [];
  const has = pts.length > 0;
  $("#riskEmpty").hidden = has;
  $("#riskChart").style.display = has ? "" : "none";
  $("#riskBasisNote").textContent = has
    ? `实线：各检查日期当时可得数据的回溯风险（${rt.model_version || ""}）；` +
      `虚线：自最近一次检查（${pts[pts.length - 1].at}）起，未来 1/3/5 年内的累计风险预测。预测≠确定结果。`
    : "";
  if (!has) return;

  const h = state.riskHorizon;
  const histData = pts
    .filter((p) => p.horizons[h])
    .map((p) => [p.at, +(p.horizons[h].probability * 100).toFixed(2), p.horizons[h].risk_tier]);

  // 未来累计风险：从最近检查日出发，P(≤+0)=0 起笔，经 +1y/+3y/+5y 三点
  const last = pts[pts.length - 1];
  const lastMs = new Date(last.at).getTime();
  const DAY = 86400000;
  const futureData = [[last.at, 0, null]];
  [["1y", 365], ["3y", 1095], ["5y", 1825]].forEach(([hz, days]) => {
    if (last.horizons[hz]) {
      futureData.push([
        new Date(lastMs + days * DAY).toISOString().slice(0, 10),
        +(last.horizons[hz].probability * 100).toFixed(2),
        `未来${H_LABEL[hz] || hz}`,
      ]);
    }
  });

  const base = baseOption();
  chart("riskChart").setOption({
    ...base,
    color: [token("--ink"), token("--ink-3")],
    legend: { top: 0, right: 0, icon: "roundRect", itemWidth: 12, itemHeight: 3,
              textStyle: { color: token("--ink-2"), fontSize: 11 } },
    tooltip: {
      ...base.tooltip,
      formatter: (ps) => ps.map((p) => {
        const tag = p.data[2] ? `（${p.data[2]}）` : "";
        return `${p.marker}${p.seriesName} ${String(p.data[0]).slice(0, 10)}：<b>${p.data[1]}%</b>${tag}`;
      }).join("<br>"),
    },
    xAxis: { ...base.xAxis, axisLabel: { ...base.xAxis.axisLabel,
      formatter: (v) => { const d = new Date(v); return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, "0")}`; } } },
    yAxis: { ...base.yAxis, name: "风险概率 %", min: 0, max: 100 },
    series: [
      {
        name: `历史 · ${H_LABEL[h] || h}风险（回溯）`, type: "line",
        smooth: 0.2, symbolSize: 9,
        lineStyle: { width: 2, color: token("--ink") },
        itemStyle: { color: (p) => tierColor(histData[p.dataIndex]?.[2]) },
        data: histData,
        markLine: {
          silent: true, symbol: "none",
          lineStyle: { color: token("--ink-3"), type: "dotted" },
          label: { formatter: "最近检查", color: token("--ink-3"), fontSize: 10 },
          data: [{ xAxis: last.at }],
        },
      },
      {
        name: "未来累计风险预测", type: "line",
        smooth: 0.2, symbolSize: 8, symbol: "emptyCircle",
        lineStyle: { width: 2, color: token("--ink-3"), type: "dashed" },
        itemStyle: { color: token("--ink-3") },
        data: futureData,
      },
    ],
  }, true);
}

$$("#riskHorizonChips .chip").forEach((b) => b.addEventListener("click", () => {
  state.riskHorizon = b.dataset.h;
  $$("#riskHorizonChips .chip").forEach((x) => x.classList.toggle("on", x === b));
  drawRiskTimeline();
}));

/** 检查时间轴条：从初始点到终止点，把每一次检查按真实日期标在轴上。 */
function renderExamStrip(dates) {
  const el = $("#examStrip");
  if (!dates || dates.length < 2) { el.hidden = true; el.innerHTML = ""; return; }
  el.hidden = false;
  const t0 = new Date(dates[0]).getTime(), t1 = new Date(dates[dates.length - 1]).getTime();
  const span = Math.max(t1 - t0, 1);
  const dot = (d) => d.replaceAll("-", ".");
  el.innerHTML = `
    <span class="es-lab">${dot(dates[0])}</span>
    <span class="es-track">${dates.map((d, i) => {
      const x = ((new Date(d).getTime() - t0) / span) * 100;
      const cur = i === dates.length - 1;
      return `<i class="${cur ? "cur" : ""}" style="left:${x}%" title="${dot(d)}"></i>`;
    }).join("")}</span>
    <span class="es-lab">${dot(dates[dates.length - 1])}</span>
    <span class="muted" style="margin-left:8px;white-space:nowrap">${dates.length} 次检查</span>`;
}

$$("#seriesRangeChips .chip").forEach((b) => b.addEventListener("click", () => {
  state.seriesRange = b.dataset.range;
  $$("#seriesRangeChips .chip").forEach((x) => x.classList.toggle("on", x === b));
  const t = state.trend;
  if (t && t.series.length) drawSeries(t.series[+$("#seriesSel").value || 0]);
}));

/** AI 分析置顶：最需要关注的 3 个问题（改动 7）。内容来自模型/规则结果，
    按"为什么重要 → 历史变化 → 当前风险 → 建议关注"组织，AI 不创造概率。 */
function renderAITop3(t) {
  const box = $("#aiTop3");
  const ivs = (t.interventions || []).filter((x) => x.level !== "平稳维持");
  const order = { "重点关注": 0, "积极改善": 1 };
  ivs.sort((a, b) => (order[a.level] ?? 9) - (order[b.level] ?? 9));
  const top = ivs.slice(0, 3);
  if (!top.length) { box.innerHTML = ""; return; }
  const compByName = {};
  (t.comparisons || []).forEach((c) => { compByName[c.name_cn] = c; });
  box.innerHTML = `<div class="top3-title">最需要关注的 ${top.length} 个问题</div>` +
    top.map((iv, i) => {
      const hist = iv.target_indicators.map((ti) => {
        const nm = ti.split(" (")[0];
        const c = compByName[nm];
        if (!c) return null;
        return c.is_real_change
          ? `${nm}较上次${c.direction}${c.worsened ? "且加重" : ""}`
          : `${nm}较上次平稳`;
      }).filter(Boolean).slice(0, 3);
      return `<div class="top3-item">
        <div class="top3-head"><span class="top3-n">${i + 1}</span><b>${esc(iv.system)}</b>
          <span class="tag ${iv.level === "重点关注" ? "t3" : "t2"}">${esc(iv.level)}</span></div>
        <div class="top3-body">
          <p><b>为什么重要：</b>${iv.target_indicators.slice(0, 4).map(esc).join("、")}</p>
          <p><b>历史变化：</b>${hist.length ? hist.map(esc).join("；") : "累计两次以上记录后可见变化方向"}</p>
          <p><b>当前风险：</b>该系统当前${iv.level === "重点关注" ? "存在恶化中的异常，属于优先处理项" : "存在异常但未见加重，建议积极改善"}</p>
          <p><b>建议关注：</b>${esc(iv.followup_cycle)}</p>
        </div>
      </div>`;
    }).join("");
}

function renderAIAdvisor(t) {
  const box = $("#trendInterventionBox");
  const ai = t.ai_analysis;
  const srcLbl = $("#aiModelSource");
  if (srcLbl) {
    if (ai && ai.source === "AI_ONLINE_LLM") {
      srcLbl.textContent = "AI 在线临床大模型（深度分析完成）";
    } else {
      srcLbl.textContent = "AI 临床专家模型（深度分析完成）";
    }
  }

  if (!ai && (!t.interventions || !t.interventions.length)) {
    box.innerHTML = `<div class="empty"><b>暂无干预数据</b>请先为该患者记录化验数据并运行预测</div>`;
    return;
  }

  // 改动 7：结论排序在上（renderAITop3），长文本一律进展开区域
  let html = "";

  // 1. 若有 LLM 自由生成的叙述文本，优先展示
  if (ai && ai.llm_narrative_text) {
    html += `
      <div class="ai-chapter-card">
        <div class="ai-ch-title"><span class="ch-badge">LLM</span> <strong>🤖 大模型综合临床见解</strong></div>
        <div class="prose ai-prose">${esc(ai.llm_narrative_text)}</div>
      </div>
    `;
  }

  // 2. 第一章：时序恶化机制深度剖析
  if (ai && ai.pathology_mechanism && ai.pathology_mechanism.length) {
    html += `
      <div class="ai-chapter-card">
        <div class="ai-ch-title"><span class="ch-badge ch-1">01</span> <strong>📊 检验指标时序波动与病理机制剖析</strong></div>
        <div class="ai-ch-body">
          ${ai.pathology_mechanism.map((p) => `<p class="ai-para">${esc(p)}</p>`).join("")}
        </div>
      </div>
    `;
  }

  // 3. 第二章：个性化精准膳食营养处方
  if (ai && ai.diet_interventions && ai.diet_interventions.length) {
    html += `
      <div class="ai-chapter-card">
        <div class="ai-ch-title"><span class="ch-badge ch-2">02</span> <strong>🥗 个性化精准膳食与营养干预处方</strong></div>
        <div class="ai-diet-grid">
          ${ai.diet_interventions.map((d) => `
            <div class="ai-block">
              <div class="ai-block-title">${esc(d.title)}</div>
              <ul>${d.items.map((it) => `<li>${esc(it)}</li>`).join("")}</ul>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }

  // 4. 第三章：生活方式、睡眠节律与分级运动管理
  if (ai && ai.lifestyle_interventions && ai.lifestyle_interventions.length) {
    html += `
      <div class="ai-chapter-card">
        <div class="ai-ch-title"><span class="ch-badge ch-3">03</span> <strong>🏃 生活方式、睡眠节律与分级运动处方</strong></div>
        <div class="ai-diet-grid">
          ${ai.lifestyle_interventions.map((l) => `
            <div class="ai-block">
              <div class="ai-block-title">${esc(l.title)}</div>
              <ul>${l.items.map((it) => `<li>${esc(it)}</li>`).join("")}</ul>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }

  // 5. 第四章：靶向专科随访日程与动态监测路径
  if (ai && ai.followup_plan) {
    const fp = ai.followup_plan;
    html += `
      <div class="ai-chapter-card">
        <div class="ai-ch-title"><span class="ch-badge ch-4">04</span> <strong>🩺 靶向专科随访日程与动态监测路径</strong></div>
        <div class="ai-timeline">
          <div class="tl-step">
            <div class="tl-badge">${esc(fp.cycle_short)}</div>
            <div class="tl-content">
              <strong>近期监测：</strong>${fp.cycle_short_items.map(esc).join("、")}
            </div>
          </div>
          <div class="tl-step">
            <div class="tl-badge highlight">${esc(fp.cycle_medium)}</div>
            <div class="tl-content">
              <strong>中期复查：</strong>${fp.cycle_medium_items.map(esc).join("、")}
            </div>
          </div>
          <div class="tl-step">
            <div class="tl-badge">${esc(fp.cycle_long)}</div>
            <div class="tl-content">
              <strong>长期评估：</strong>${fp.cycle_long_items.map(esc).join("、")}
            </div>
          </div>
        </div>
        <div class="tl-dept"><strong>推荐就医科室：</strong>${esc(fp.recommend_dept)}</div>
      </div>
    `;
  }

  // 6. 第五章：危险信号预警与就医红线
  if (ai && ai.red_flags && ai.red_flags.length) {
    html += `
      <div class="ai-chapter-card alert-red-card">
        <div class="ai-ch-title red"><span class="ch-badge ch-red">05</span> <strong>🚨 危险信号预警与就医红线</strong></div>
        <ul class="red-flag-list">
          ${ai.red_flags.map((rf) => `<li>${esc(rf)}</li>`).join("")}
        </ul>
      </div>
    `;
  }

  box.innerHTML = html
    ? `<details class="fold ai-fold"><summary>展开完整专业分析（机制剖析 / 膳食与运动处方 / 随访日程）</summary>
       <div style="margin-top:12px">${html}</div></details>`
    : "";
}

function drawSeries(s) {
  if (!s) return;
  // 时间范围筛选（全部/近1年/近3年/近5年）——从最近一次检查日期倒推
  let points = s.points;
  if (state.seriesRange !== "all" && points.length) {
    const lastMs = new Date(points[points.length - 1].at).getTime();
    const cutoff = lastMs - (+state.seriesRange) * 365.25 * 86400000;
    const filtered = points.filter((p) => new Date(p.at).getTime() >= cutoff);
    if (filtered.length >= 1) points = filtered;
  }
  s = { ...s, points };
  const ref = state.refMap[s.code] || {};
  const markArea = (ref.low != null || ref.high != null) ? {
    silent: true,
    itemStyle: { color: token("--paper-2"), opacity: .75 },
    label: { show: true, position: "insideTopLeft", color: token("--ink-3"),
             fontSize: 10, formatter: "参考区间" },
    data: [[{ yAxis: ref.low ?? 0 }, { yAxis: ref.high ?? "max" }]],
  } : undefined;
  const base = baseOption();
  chart("seriesChart").setOption({
    ...base,
    xAxis: { ...base.xAxis, axisLabel: { ...base.xAxis.axisLabel,
      formatter: (v) => { const d = new Date(v); return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, "0")}`; } } },
    yAxis: { ...base.yAxis, name: s.unit, scale: true },
    series: [{
      name: s.name_cn, type: "line", smooth: 0.25, symbolSize: 9,
      lineStyle: { width: 2, color: token("--ink-2") },
      itemStyle: { color: (p) => {
        const g = Math.abs(s.points[p.dataIndex].grade || 0);
        return token(g >= 3 ? "--t4" : g === 2 ? "--t3" : g === 1 ? "--t2" : "--t1");
      } },
      data: s.points.map((p) => [p.at, p.value]),
      markArea,
    }],
  }, true);
}

/* ---------------- 管理台 ---------------- */
async function refreshAdmin() {
  try {
    const d = await api("/admin/versions");
    const rows = Object.values(d.versions).sort((a, b) => a.created_at.localeCompare(b.created_at));
    const stTag = { ACTIVE: "t1", CANARY: "t2", STAGING: "line", RETIRED: "" };
    $("#versionList").innerHTML = rows.map((v) => {
      const auc = v.headline_auc && v.headline_auc["3y"] != null ? v.headline_auc["3y"].toFixed(3) : "—";
      const traffic = v.status === "CANARY" ? v.traffic_pct : v.status === "ACTIVE" ? 100 : 0;
      const ops = [];
      if (v.status === "STAGING" || v.status === "CANARY")
        ops.push(`<button class="btn sm" data-promote="${esc(v.version)}">晋升全量</button>`);
      if (v.status === "STAGING")
        ops.push(`<button class="btn ghost sm" data-canary="${esc(v.version)}">设灰度</button>`);
      return `<div class="rowitem static">
        <span class="grow">
          <span class="t1line"><span class="mono-id">${esc(v.version)}</span>
            <span class="tag ${stTag[v.status] || ""}">${esc(v.status)}</span></span>
          <span class="sub">AUC(3y) ${auc} · 承接流量 ${traffic}%${
            v.notes ? " · " + esc(v.notes) : ""}</span>
        </span>
        <span style="display:flex;gap:6px">${ops.join("")}</span>
      </div>`;
    }).join("") || `<div class="empty"><b>注册表为空</b>先完成一次自举</div>`;

    $$("#versionList [data-promote]").forEach((b) =>
      b.addEventListener("click", () => promote(b.dataset.promote)));
    $$("#versionList [data-canary]").forEach((b) =>
      b.addEventListener("click", () => setCanary(b.dataset.canary)));
  } catch (e) { toast(e.message, true); }
}

async function promote(v) {
  if (!confirm(`把 ${v} 晋升为全量版本？下一次预测起立即生效。`)) return;
  try {
    await api("/admin/promote", { method: "POST", body: { version: v } });
    toast(`${v} 已全量上线`);
    await refreshAdmin(); await boot();
  } catch (e) { toast(e.message, true); }
}
async function setCanary(v) {
  const p = prompt(`${v} 的灰度流量百分比 (0-100]`, "10");
  if (p == null) return;
  try {
    await api("/admin/canary", { method: "POST", body: { version: v, traffic_pct: +p } });
    toast(`${v} 灰度 ${p}% 生效`);
    await refreshAdmin(); await boot();
  } catch (e) { toast(e.message, true); }
}
$("#btnRefreshAdmin").addEventListener("click", refreshAdmin);
$("#btnRollback").addEventListener("click", async () => {
  if (!confirm("回滚到最近一个已退役版本？")) return;
  try {
    const info = await api("/admin/rollback", { method: "POST", body: {} });
    toast(`已回滚至 ${info.version}`);
    await refreshAdmin(); await boot();
  } catch (e) { toast(e.message, true); }
});

$("#btnDrift").addEventListener("click", async () => {
  const box = $("#driftBox");
  box.innerHTML = `<div class="empty"><span class="spinner"></span> 计算 PSI…</div>`;
  try {
    const r = await api(`/admin/drift?horizon=${encodeURIComponent($("#driftHorizon").value)}`);
    const lvTag = { OK: "t1", WATCH: "t2", ALERT: "t4", INSUFFICIENT: "line" };
    const feats = (r.features || []).filter((f) => f.level !== "OK").slice(0, 8);
    const maxPsi = Math.max(...feats.map((f) => f.psi || 0), 0.25);
    box.innerHTML = `
      <div class="t1line" style="display:flex;align-items:center;gap:8px">
        <span class="tag ${lvTag[r.level] || ""}">${esc(r.level)}</span>
        <span class="muted">线上样本 ${r.n_online} · 加权 PSI ${(r.weighted_psi ?? 0).toFixed(3)}
          · 最大 PSI ${(r.max_psi ?? 0).toFixed(3)}</span>
      </div>
      ${(r.messages || []).map((m) => `<p class="muted" style="margin:6px 0 0">· ${esc(m)}</p>`).join("")}
      ${feats.length ? `<div style="margin-top:12px">${feats.map((f) => `
        <div class="hmini">
          <span class="hl" style="width:auto;flex:0 0 34%;overflow:hidden;text-overflow:ellipsis"
                title="${esc(f.name)}">${esc(f.name)}</span>
          <span class="bar"><i style="width:${Math.min(100, (f.psi || 0) / maxPsi * 100)}%;
            background:var(--${f.level === "ALERT" ? "t4" : "t2"})"></i></span>
          <span class="pv">${(f.psi ?? 0).toFixed(3)}</span>
        </div>`).join("")}</div>` : `<p class="muted" style="margin-top:10px">没有超阈值的特征。</p>`}`;
  } catch (e) { box.innerHTML = ""; toast(e.message, true); }
});

$("#btnReview").addEventListener("click", async () => {
  try {
    const q = await api("/admin/review-queue");
    $("#reviewSummary").textContent = q.summary;
    const catTag = { false_negative: "t4", false_positive: "t2", confirmed_positive: "t1" };
    const catCn = { false_negative: "漏诊 FN", false_positive: "过度预警 FP", confirmed_positive: "判对 TP" };
    $("#reviewList").innerHTML = q.cases.map((c) => `
      <div class="rowitem static">
        <span class="num" style="width:26px;color:var(--ink-3)">${c.priority_rank + 1}</span>
        <span class="grow">
          <span class="t1line">
            <span class="tag ${catTag[c.category] || ""}">${catCn[c.category] || esc(c.category)}</span>
            <span class="tag t${TIER_IDX[c.risk_tier] || 1}">${esc(c.risk_tier)}</span>
          </span>
          <span class="sub mono">${esc(c.trace_id.slice(0, 14))}… · 结局${
            c.outcome_event ? "发生" : "未发生"}</span>
        </span>
        <span class="num">${pct1(c.probability)}</span>
      </div>`).join("") || `<div class="empty"><b>队列是空的</b>还没有带随访结局的样本</div>`;
  } catch (e) { toast(e.message, true); }
});

$("#btnAB").addEventListener("click", async () => {
  const c = $("#abChampion").value.trim(), g = $("#abChallenger").value.trim();
  if (!c || !g) return toast("两个版本号都要填", true);
  const h = $("#abHorizon").value;
  try {
    const r = await api(`/admin/ab?champion=${encodeURIComponent(c)}&challenger=${
      encodeURIComponent(g)}${h ? "&horizon=" + encodeURIComponent(h) : ""}`);
    $("#abBox").textContent = r.summary;
  } catch (e) { toast(e.message, true); }
});

$("#btnRefreshAI").addEventListener("click", async () => {
  if (!state.pid) return toast("请先选择患者", true);
  toast("正在调用 AI 临床大模型生成深度解读与干预方案...");
  try {
    delete state.trend;
    await loadTrend();
    toast("AI 临床干预方案已更新！");
  } catch (e) {
    toast(e.message, true);
  }
});

/* ---------------- go ---------------- */
boot().catch((e) => toast("初始化失败：" + e.message, true));
