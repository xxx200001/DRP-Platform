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
  const map = { ingest: "#sec-ingest", predict: "#sec-predict", followup: "#sec-followup" };
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
  state.pid = pid;
  state.patient = state.patients.find((p) => p.patient_id === pid) || null;
  state.trend = null; state.predicted = false; state.followedUp = false;
  store.set("drp.pid", pid);

  renderCtx();
  $("#predictOut").hidden = true;
  $("#sec-factors").hidden = true;
  $("#sec-referral").hidden = true;
  $("#parseOut").hidden = true;

  await loadRecords();
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

/* ---------------- 报告解析与自动入库 ---------------- */
async function parseAndIngestText(text) {
  if (!state.pid) {
    toast("先在「患者」里选一位", true);
    return null;
  }
  const measured_at = $("#repDate").value || new Date().toISOString().slice(0, 10);
  const r = await api("/reports/parse", {
    method: "POST",
    body: { patient_id: state.pid, text: text, measured_at: measured_at },
  });
  renderParse(r);
  await refreshPatients();
  state.patient = state.patients.find((p) => p.patient_id === state.pid);
  renderCtx();
  await loadRecords();
  state.trend = null;
  updateSteps();
  return r;
}

/* ---------------- 拍照/上传图片识别 ---------------- */
$("#btnOCR").addEventListener("click", () => { $("#repImage").click(); });
$("#repImage").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (!state.pid) { toast("先在「患者」里选一位", true); return; }

  const btn = $("#btnOCR");
  btn.disabled = true;
  $("#parseStat").innerHTML = `<span class="spinner"></span> 正在识别并自动入库…`;

  try {
    const b64 = await new Promise((ok, fail) => {
      const reader = new FileReader();
      reader.onload = () => ok(reader.result);
      reader.onerror = fail;
      reader.readAsDataURL(file);
    });

    const ocrRes = await api("/ocr", { method: "POST", body: { image: b64 } });
    if (!ocrRes.text || !ocrRes.text.trim()) {
      $("#parseStat").textContent = "";
      toast("未识别到文字，请检查图片清晰度", true);
      return;
    }

    $("#repText").value = ocrRes.text;

    // 自动直接执行解析入库
    const parseRes = await parseAndIngestText(ocrRes.text);
    $("#parseStat").textContent = "";
    if (parseRes) {
      toast(`已自动识别并入库 ${parseRes.stored} 项指标`);
    }
  } catch (err) {
    $("#parseStat").textContent = "";
    toast(err.message || "识别/入库失败", true);
  } finally {
    btn.disabled = false;
    $("#repImage").value = "";
  }
});

/* ---------------- 填充示例 ---------------- */
$("#btnSample").addEventListener("click", async () => {
  $("#repText").value = state.meta.sample_report;
  if (state.pid) {
    $("#parseStat").innerHTML = `<span class="spinner"></span> 解析中…`;
    try {
      const r = await parseAndIngestText(state.meta.sample_report);
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
  btn.disabled = true;
  $("#parseStat").innerHTML = `<span class="spinner"></span> 解析中…`;
  try {
    const r = await parseAndIngestText($("#repText").value);
    $("#parseStat").textContent = "";
    if (r) toast(`入库 ${r.stored} 条`);
  } catch (e) {
    $("#parseStat").textContent = "";
    toast(e.message, true);
  } finally { btn.disabled = false; }
});

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

$("#btnPredict").addEventListener("click", async () => {
  if (!state.pid) return toast("先在「患者」里选一位", true);
  const btn = $("#btnPredict");
  btn.disabled = true;
  $("#predStat").innerHTML = `<span class="spinner"></span> 特征管线 + 多时程推理中…`;
  try {
    const r = await api("/predict", { method: "POST", body: { patient_id: state.pid } });
    $("#predStat").textContent = "";
    renderPredict(r);
    state.predicted = true;
    state.trend = null;
    await loadTraces();
    await refreshPatients();
    updateSteps();
  } catch (e) {
    $("#predStat").textContent = "";
    toast(e.message, true);
  } finally { btn.disabled = false; }
});

function renderPredict(r) {
  $("#predictOut").hidden = false;
  const main = r.results.find((x) => x.horizon === "3y") || r.results[r.results.length - 1];
  const tiers = (state.meta.tiers || {})[main.horizon];
  const cuts = tiers ? tiers.cutpoints : [0.05, 0.15, 0.4];
  const names = tiers ? tiers.names : ["低危", "中危", "高危", "极高危"];
  const pos = tierPos(main.probability, cuts);
  const color = tierColor(main.risk_tier);

  $("#heroHorizon").textContent = (main.horizon || "").toUpperCase() + " 进展风险";
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

  // 其余时程
  $("#horizonMini").innerHTML = r.results.map((x) => {
    const c = (state.meta.tiers || {})[x.horizon];
    const p2 = tierPos(x.probability, c ? c.cutpoints : cuts);
    return `<div class="hmini">
      <span class="hl">${(x.horizon || "").toUpperCase()}</span>
      <span class="bar"><i style="width:${p2.pct}%;background:${tierColor(x.risk_tier)}"></i></span>
      <span class="pv" style="color:${tierColor(x.risk_tier)}">${pct1(x.probability)}</span>
      <span class="tag t${TIER_IDX[x.risk_tier] || 1}">${esc(x.risk_tier)}</span>
    </div>`;
  }).join("");

  $("#servedBy").textContent =
    `服务版本 ${r.model_version}（${r.arm === "canary" ? "灰度臂" : "全量臂"}）· ` +
    `每个时程一条独立 trace，已写入全链路日志`;

  renderFactors(main);
  renderReferral(r.referral);
  $("#narrative").textContent = main.narrative;
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

/* ---------------- 步骤状态 ---------------- */
function updateSteps() {
  const hasRec = (state.patient?.n_records || 0) > 0;
  const hasPred = state.predicted || !!state.patient?.last_predicted_at;
  const set = (k, v) => $(`.step[data-step="${k}"]`).dataset.state = v;
  set("ingest", hasRec ? "done" : "now");
  set("predict", hasPred ? "done" : hasRec ? "now" : "idle");
  set("followup", state.followedUp ? "done" : hasPred ? "now" : "idle");
}

/* ---------------- 趋势 ---------------- */
async function loadTrend() {
  if (!state.pid) return;
  const t = state.trend || (await api(`/patients/${encodeURIComponent(state.pid)}/trend`));
  state.trend = t;

  // 风险走势：线色中性表示"哪个时程"，点色表示"当时哪一层"——
  // 保持"饱和色只表示风险等级"这条全站规则。
  const inks = [token("--ink"), token("--ink-2"), token("--ink-3")];
  // 顶层 color 决定图例图标色；不设它 echarts 会退回默认蓝绿黄调色板。
  const series = Object.entries(t.risk_trajectories).map(([h, traj], i) => ({
    name: H_LABEL[h] || h, type: "line", smooth: 0.25, symbolSize: 9,
    lineStyle: { width: 2, color: inks[i % 3] },
    itemStyle: { color: (p) => tierColor(traj.points[p.dataIndex].risk_tier) },
    data: traj.points.map((p) => [p.at, +(p.probability * 100).toFixed(2)]),
  }));
  const has = series.some((s) => s.data.length);
  $("#riskEmpty").hidden = has;
  $("#riskChart").style.display = has ? "" : "none";
  if (has) {
    chart("riskChart").setOption({
      ...baseOption(),
      color: inks,
      legend: { top: 0, right: 0, icon: "roundRect", itemWidth: 10, itemHeight: 3,
                textStyle: { color: token("--ink-2"), fontSize: 11 } },
      tooltip: { ...baseOption().tooltip, valueFormatter: (v) => v + "%" },
      yAxis: { ...baseOption().yAxis, name: "风险概率 %", min: 0 },
      series,
    }, true);
  }

  // 本次 vs 上次：复用参考区间带，把"上次→本次"的位移画出来
  const cbox = $("#compareList");
  $("#compareEmpty").hidden = !!t.comparisons.length;
  cbox.innerHTML = t.comparisons.map((c) => {
    const ref = state.refMap[c.code] || {};
    const verdict = c.is_real_change
      ? `<span class="tag ${c.worsened ? "t3" : "cool"}">${esc(c.direction)}${c.worsened ? " · 加重" : ""}</span>`
      : `<span class="tag line">RCV 内 · 视为平稳</span>`;
    const d = `<span class="mono" style="margin-left:6px">${c.delta > 0 ? "+" : ""}${
      num(c.delta)}${c.delta_pct != null ? `（${(c.delta_pct * 100).toFixed(0)}%）` : ""}</span>`;
    return bandRow({
      name: c.name_cn, code: c.code, unit: c.unit, value: c.curr_value, prev: c.prev_value,
      low: ref.low, high: ref.high, grade: c.curr_grade, extra: " " + verdict + d,
    });
  }).join("");

  // 指标曲线：参考区间画成背景带，比在图例里写"正常范围"直观
  const sel = $("#seriesSel");
  sel.innerHTML = t.series.map((s, i) => `<option value="${i}">${esc(s.name_cn)}</option>`).join("");
  sel.onchange = () => drawSeries(t.series[+sel.value]);
  if (t.series.length) drawSeries(t.series[0]);
  $("#trendText").textContent = t.rendered_text;

  // 渲染干预建议与应对方案卡片
  const ivBox = $("#trendInterventionBox");
  if (t.interventions && t.interventions.length) {
    ivBox.innerHTML = t.interventions.map((it) => `
      <div class="intervention-card ${it.level === '重点关注' ? 'alert-high' : 'alert-normal'}">
        <div class="iv-header">
          <div class="iv-title"><span class="iv-icon">${esc(it.icon)}</span> <strong>${esc(it.system)}</strong></div>
          <span class="tag ${it.level === '重点关注' ? 't3' : 't2'}">${esc(it.level)}</span>
        </div>
        ${it.target_indicators && it.target_indicators.length ? `
          <div class="iv-targets">
            <span class="iv-lbl">关联指标：</span>
            ${it.target_indicators.map((tg) => `<span class="iv-badge">${esc(tg)}</span>`).join("")}
          </div>` : ""}
        <div class="iv-grid">
          <div class="iv-sec">
            <div class="iv-sub">🥗 膳食调理办法</div>
            <ul>${it.diet_advice.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>
          </div>
          <div class="iv-sec">
            <div class="iv-sub">🏃 运动与作息管理</div>
            <ul>${it.lifestyle_advice.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
          </div>
        </div>
        <div class="iv-footer">
          <div class="iv-cycle"><strong>🩺 复查跟踪周期：</strong>${esc(it.followup_cycle)}</div>
          ${it.red_flags && it.red_flags.length ? `
            <div class="iv-red"><strong>🚨 就医预警指征：</strong>${it.red_flags.map(esc).join("；")}</div>` : ""}
        </div>
      </div>
    `).join("");
  } else {
    ivBox.innerHTML = `<div class="empty"><b>暂无干预建议</b>各项指标均在平稳健康范围</div>`;
  }
}

function drawSeries(s) {
  if (!s) return;
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

/* ---------------- go ---------------- */
boot().catch((e) => toast("初始化失败：" + e.message, true));
