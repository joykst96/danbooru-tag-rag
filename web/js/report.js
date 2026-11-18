// 생성 리포트 모달: 방금 끝난 생성 1건이 어떤 과정으로 만들어졌는지 보여준다.
// (운영자용 log_viewer.html 의 상세 화면에서 목록·검색·빌드 정보·원본 JSON 을 뺀 것)
//
// 리포트는 생성 스트림의 마지막 줄로만 온다(core/api.py _report_event). 다시 받아올 API 가 없다.
// 그래서 state.report(메모리)에만 두고, localStorage·IndexedDB·URL 어디에도 넣지 않는다.
//   - 새로고침하면 사라진다.
//   - 새 생성을 시작하면 generate.js 가 clearReport() 로 지운다.
import { state, $, escapeHtml as esc } from "./state.js";
import { tagInfo, ac } from "./tag-data.js";
import { doCopy } from "./clipboard.js";
import { hideTooltip } from "./tooltip.js";

const STATUS_LABEL = { ok: "완료", error: "오류" };
const MODE_LABEL = { basic: "기본", split: "고급 (인물 분할)" };
const TONE_LABEL = { rich: "묘사적", plain: "담백", phrase: "단어형+" };
const CAT_LABEL = { 0: "일반", 1: "작가", 3: "작품", 4: "캐릭터", 5: "메타" };
// 서버 단계 키 → 화면 이름. 분해는 한국어·영어 조각을 한 번에 내므로 english 는 길이 0 이라 표시되지 않는다.
const STAGE_LABEL = { korean: "문장 분해", english: "영어 변환", final: "태그 검색·선택", nl: "자연어 작성", background: "배경" };
const stageLabel = (k) => STAGE_LABEL[k] || k.replace(/^character(\d+)$/, "인물 $1");

const sec = (ms) => (ms == null ? "–" : ms < 1000 ? ms + "ms" : (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + "초");
// 태그 표시: 언더스코어→공백. ^_^ 같은 짧은 이모티콘 태그는 그대로.
const tagLabel = (t) => (t.length > 3 ? t.replace(/_/g, " ") : t);
const NONE = '<span class="rp-none">없음</span>';

// ── 태그 칩 ──
// 태그 사전(/danbooru-tags.csv → tagInfo)에 있는 태그는 호버하면 설명이 뜬다(tooltip.js 가 data-tooltip 을 읽는다).
function chip(tag, cls = "") {
  tag = String(tag);
  if (tag.startsWith("phrase:")) {
    return `<span class="rp-chip phrase" data-tooltip="맞는 Danbooru 태그가 없어 영어 구 그대로 넣었습니다.">${esc(tag.slice(7))}</span>`;
  }
  const info = tagInfo.get(tag);
  const tip = info ? ` data-tooltip="${esc(info.desc || "설명 없음")}"` : "";
  return `<span class="rp-chip ${cls}"${tip}>${esc(tagLabel(tag))}</span>`;
}
const chips = (arr, cls = () => "") =>
  arr && arr.length ? `<div class="rp-chips">${arr.map((t) => chip(t, cls(t))).join("")}</div>` : NONE;
// 태그가 아닌 글(분해 조각 등): 사전 조회·표기 변환 없이 그대로
const plainChips = (arr, cls = "") =>
  arr && arr.length ? `<div class="rp-chips">${arr.map((s) => `<span class="rp-chip ${cls}">${esc(s)}</span>`).join("")}</div>` : NONE;

// ── 복사 버튼 ── (렌더할 때마다 버퍼를 새로 채운다)
let copyBuf = [];
function copyBtn(text) {
  copyBuf.push(String(text ?? ""));
  return `<button type="button" class="copy-btn rp-copy" data-rp-copy="${copyBuf.length - 1}">복사</button>`;
}

const section = (title, body, copy = "") =>
  `<section class="rp-section"><div class="rp-section-head"><h3>${title}</h3>${copy}</div>${body}</section>`;
const kv = (rows) =>
  `<dl class="rp-kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;

function renderSettings(r) {
  const q = r.request || {};
  const rows = [
    ["자연어 톤", esc(TONE_LABEL[q.nl_tone] || TONE_LABEL.rich)],
    ["자연어 온도", q.nl_temperature == null ? "톤 권장값" : esc(q.nl_temperature)],
  ];
  if (r.mode !== "split") {
    const cats = q.search_categories && q.search_categories.length ? q.search_categories : [0];
    rows.push(["태그 검색 범위", esc(cats.map((c) => CAT_LABEL[c] ?? c).join(", "))]);
  }
  rows.push(["맞는 태그가 없을 때 영어 구", q.include_phrases === false ? "넣지 않음" : "넣음"]);
  return section("설정", `<div class="rp-box">${kv(rows)}</div>`);
}

function renderTiming(tm) {
  if (!tm) return "";
  // stages 는 차례가 온 뒤부터의 누적 시각 → 앞 단계와의 차이가 그 단계에 걸린 시간
  const stages = Object.entries(tm.stages || {}).sort((a, b) => a[1] - b[1]);
  const segs = [];
  if (tm.queue) segs.push(["대기", tm.queue, "wait"]);
  let prev = 0;
  stages.forEach(([k, v]) => {
    const d = Math.max(0, v - prev);
    prev = v;
    if (d > 0) segs.push([stageLabel(k), d, "s" + (segs.filter((s) => s[2] !== "wait").length % 5 + 1)]);
  });
  if (!segs.length) return "";
  const sum = segs.reduce((a, [, v]) => a + v, 0);
  const bar = segs
    .map(([k, v, c]) => `<span class="rp-seg ${c}" style="width:${(v / sum) * 100}%" data-tooltip="${esc(k)} ${sec(v)}">${v / sum > 0.16 ? esc(k) : ""}</span>`)
    .join("");
  const legend = segs.map(([k, v, c]) => `<span><i class="rp-seg ${c}"></i>${esc(k)} <b>${sec(v)}</b></span>`).join("");
  return section(`걸린 시간 · 총 ${sec(tm.total)}`, `<div class="rp-box"><div class="rp-bar">${bar}</div><div class="rp-legend">${legend}</div></div>`);
}

function renderBasic(r) {
  const t = r.trace || {};
  let h = "";
  const people = t.people || [], droppedPeople = t.dropped_people || [];
  if (people.length || droppedPeople.length) {
    let body = chips(people);
    if (droppedPeople.length) {
      body += `<div class="rp-sub-row"><span class="rp-sub">인원수로 센 표현</span>${plainChips(droppedPeople, "del")}</div>`;
    }
    h += section("인원", `<div class="rp-box">${body}</div>`);
  }
  if (t.units && t.units.length) {
    let anyWarn = false, anyPhrase = false;
    const rows = t.units.map((u) => {
      const picks = u.picks || [], cands = u.cands || [];
      const pset = new Set(picks), cset = new Set(cands);
      // 후보 밖에서 고른 태그는 검색이 놓쳤거나 선택이 지어낸 것이라 따로 표시한다
      const pickCls = (p) => {
        if (String(p).startsWith("phrase:")) { anyPhrase = true; return ""; }
        if (cset.has(p)) return "pick";
        anyWarn = true;
        return "warn";
      };
      return `<tr><td class="rp-ko">${esc(u.ko)}${u.kind === "character" ? '<div class="rp-sub">캐릭터</div>' : ""}</td>
        <td>${esc(u.en)}</td>
        <td>${chips(cands, (c) => (pset.has(c) ? "pick" : ""))}</td>
        <td>${picks.length ? chips(picks, pickCls) : '<span class="rp-none">선택 없음</span>'}</td></tr>`;
    }).join("");
    const legend = ['<span><i class="rp-chip pick"></i>선택된 태그</span>']
      .concat(anyPhrase ? ['<span><i class="rp-chip phrase"></i>태그 대신 넣은 영어 구</span>'] : [])
      .concat(anyWarn ? ['<span><i class="rp-chip warn"></i>후보 밖에서 고른 태그</span>'] : [])
      .concat(ac.ready ? ["<span>태그에 마우스를 올리면 설명이 나옵니다</span>"] : ["<span>태그 설명을 불러오는 중…</span>"]);
    h += section("조각별 분해 → 후보 → 선택",
      `<div class="rp-table-wrap"><table class="rp-table"><thead><tr><th>한국어</th><th>영어</th><th>후보</th><th>선택</th></tr></thead><tbody>${rows}</tbody></table></div>
       <div class="rp-legend">${legend.join("")}</div>`);
  } else if ((t.ko_units || []).length || (t.en_units || []).length) {
    // 오류로 분해 단계까지만 진행된 경우
    h += section("분해 (여기까지 진행)", `<div class="rp-box">${kv([["한국어", plainChips(t.ko_units)], ["영어", plainChips(t.en_units)]])}</div>`);
  }
  if ((t.select_notes || []).length) {
    h += section("선택 메모", `<div class="rp-box"><ul class="rp-notes">${t.select_notes.map((n) => `<li>${esc(typeof n === "string" ? n : JSON.stringify(n))}</li>`).join("")}</ul></div>`);
  }
  return h;
}

function renderSplit(r) {
  const t = r.trace || {};
  const reqChars = (r.request && r.request.characters) || [];
  const done = new Map((t.characters || []).map((c) => [c.index, c]));
  const n = Math.max(reqChars.length, ...[...done.keys()].map(Number), 0);
  let cards = "";
  for (let i = 1; i <= n; i++) {
    const q = reqChars[i - 1] || {}, c = done.get(i);
    const flags = [];
    if (q.is_original) flags.push('<span class="rp-badge">오리지널</span>');
    if (q.is_passthrough || (c && c.is_passthrough)) flags.push('<span class="rp-badge">패스스루</span>');
    if (c && c.name_unmatched) flags.push('<span class="rp-badge warn">DB에 없는 이름</span>');
    const name = [q.name || (c && c.name), q.series || (c && c.series)].filter(Boolean).join(" · ");
    cards += `<div class="rp-box rp-char">
      <div class="rp-char-head"><span class="rp-sub">#${i}</span><b>${esc(name) || '<span class="rp-none">(이름 없음)</span>'}</b>${flags.join("")}</div>
      ${q.desc ? `<div class="rp-pre">${esc(q.desc)}</div>` : ""}
      ${c ? chips(c.tags) : '<span class="rp-none">처리 전에 끝남</span>'}</div>`;
  }
  let h = section("인물칸", cards || `<div class="rp-box">${NONE}</div>`);
  const bg = (r.request && r.request.background) || "";
  if (bg || t.background_tags) {
    h += section("배경", `<div class="rp-box rp-char">${bg ? `<div class="rp-pre">${esc(bg)}</div>` : ""}${t.background_tags ? chips(t.background_tags) : '<span class="rp-none">처리 전에 끝남</span>'}</div>`);
  }
  return h;
}

function render() {
  const r = state.report;
  if (!r) return;
  copyBuf = [];
  const when = r.ts ? new Date(r.ts).toLocaleString("ko-KR") : "";
  $("report_meta").innerHTML =
    `<span class="rp-badge ${r.status === "ok" ? "ok" : "error"}">${esc(STATUS_LABEL[r.status] || r.status)}</span>` +
    `<span class="rp-badge">${esc(MODE_LABEL[r.mode] || r.mode)}</span>` +
    `<span>${esc(when)}</span><span>요청 번호 #${esc(r.id)}</span>`;

  let h = "";
  if (r.status !== "ok") {
    h += `<div class="rp-box rp-error">이 요청은 오류로 끝났습니다${r.error && r.error.type ? ` (${esc(r.error.type)})` : ""}. 아래는 오류 전까지 진행된 내용입니다.</div>`;
  }
  h += section("입력", `<div class="rp-box rp-pre">${esc(r.input) || '<span class="rp-none">(빈 입력)</span>'}</div>`, copyBtn(r.input));

  const tags = r.final_tags_raw || [];
  const dropped = (r.trace && r.trace.hallucinated) || [];
  // 사전이 준비됐는데 사전에 없는 최종 태그 = 태그 대신 들어간 영어 구
  const finalCls = (t) => (ac.ready && !tagInfo.has(t) ? "phrase" : "");
  let tagBody = chips(tags, finalCls);
  if (dropped.length) tagBody += `<div class="rp-sub-row"><span class="rp-sub">사전에 없어 버린 태그</span>${chips(dropped, () => "del")}</div>`;
  h += section(`최종 태그 (${tags.length})`, `<div class="rp-box">${tagBody}</div>`, tags.length ? copyBtn(tags.map(tagLabel).join(", ")) : "");

  if (r.nl_prompt || r.status === "ok") {
    h += section("자연어", `<div class="rp-box rp-pre">${esc(r.nl_prompt) || NONE}</div>`, r.nl_prompt ? copyBtn(r.nl_prompt) : "");
  }
  h += r.mode === "split" ? renderSplit(r) : renderBasic(r);
  h += renderTiming(r.timing_ms);
  h += renderSettings(r);
  h += '<p class="help">이 리포트는 지금 열려 있는 화면에만 있습니다. 새로고침하거나 다시 생성하면 사라지고, 다시 받아올 수 없습니다.</p>';
  $("report_body").innerHTML = h;
}

// ── 외부에서 쓰는 것 ──
export function setReport(report) {
  state.report = report;
  $("report_bar").hidden = false;
}

export function clearReport() {
  closeReport();
  state.report = null;
  $("report_bar").hidden = true;
  $("report_body").innerHTML = "";
  $("report_meta").innerHTML = "";
}

// 태그 사전이 리포트보다 늦게 준비됐을 때: 열려 있으면 설명을 채워 다시 그린다(스크롤 위치 유지).
export function refreshReport() {
  if ($("report_modal").hidden || !state.report) return;
  const body = $("report_body");
  const top = body.scrollTop;
  render();
  body.scrollTop = top;
}

let lastFocus = null;
function openReport() {
  if (!state.report) return;
  render();
  lastFocus = document.activeElement;
  $("report_modal").hidden = false;
  document.body.classList.add("report-open");
  $("report_body").scrollTop = 0;
  $("report_close_btn").focus();
}

function closeReport() {
  const modal = $("report_modal");
  if (modal.hidden) return;
  modal.hidden = true;
  document.body.classList.remove("report-open");
  hideTooltip();   // 호버 중이던 칩이 사라지면 mouseout 이 오지 않는다
  if (lastFocus && document.contains(lastFocus)) lastFocus.focus();
  lastFocus = null;
}

export function initReport() {
  const modal = $("report_modal");
  $("report_open_btn").addEventListener("click", openReport);
  $("report_close_btn").addEventListener("click", closeReport);
  // 바깥(어두운 영역)을 누르면 닫는다
  modal.addEventListener("mousedown", (e) => { if (e.target === modal) closeReport(); });
  modal.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-rp-copy]");
    if (!btn) return;
    doCopy(copyBuf[+btn.dataset.rpCopy], () => {
      btn.textContent = "복사됨";
      btn.classList.add("is-copied");
      setTimeout(() => { btn.textContent = "복사"; btn.classList.remove("is-copied"); }, 1200);
    });
  });
  document.addEventListener("keydown", (e) => {
    if (modal.hidden) return;
    if (e.key === "Escape") { e.preventDefault(); closeReport(); return; }
    if (e.key !== "Tab") return;
    // 모달이 열린 동안 Tab 이 뒤 화면으로 나가지 않게 한다
    const items = [...modal.querySelectorAll("button, [href], [tabindex]:not([tabindex='-1'])")].filter((el) => !el.disabled);
    if (!items.length) return;
    const first = items[0], last = items[items.length - 1];
    if (!modal.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
    else if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  });
}
