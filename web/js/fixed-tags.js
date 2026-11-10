// 고정 태그(퀄리티·작가)와 패스스루 이름: LLM/DB 를 거치지 않고 프론트에서 결과 태그 앞뒤에 붙인다.
// 최종 순서: [앞 슬롯(우선순위순)] + [패스스루 + danbooru] + [뒤 슬롯(우선순위순)]
import { state, $, setActiveIn } from "./state.js";
import {
  parseTagWeight, formatTagWeight, bumpToken, readFormatOptions, formatTagText,
} from "./tag-format.js";
import { displayResults } from "./results.js";

// ── 위치·우선순위·접두사 세그먼트 ──
export function setArtistPrefix(btn) {
  state.artistPrefix = btn.dataset.prefix;
  setActiveIn("#artist_prefix_group .seg-btn", (b) => b === btn);
  refreshFixedTags();
}
export function setFixedPos(which, btn) {
  state.fixedPos[which] = btn.dataset.pos;
  const grp = which === "quality" ? "#quality_pos_group" : "#artist_pos_group";
  setActiveIn(grp + " .seg-btn", (b) => b === btn);
  refreshFixedTags();
}
export function setFixedPriority(btn) {
  state.fixedPriority = btn.dataset.priority;
  setActiveIn("#fixed_priority_group .seg-btn", (b) => b === btn);
  refreshFixedTags();
}

// 고정 태그 입력이 바뀌면 결과 영역 다시 렌더
export function refreshFixedTags() {
  if (state.result && state.result.final_prompt) displayResults();
}

// ── 퀄리티 ──
// 퀄리티 input 의 원시 토큰 배열(가중치 표기 포함, 로컬 (tag:w)). 진실원천.
function qualityTokens() {
  const raw = ($("fixed_quality")?.value || "").trim();
  if (!raw) return [];
  return raw.split(",").map(t => t.trim()).filter(Boolean);
}
// 결과/복사용: 각 토큰에 현재 표기(로컬/NAI) 적용. (고정태그는 치환/이스케이프 미적용)
function parseFixedQuality() {
  return qualityTokens().map((tok) => {
    const { tag, weight } = parseTagWeight(tok);
    return formatTagWeight(tag, weight);
  });
}
// 퀄리티 input 변경 → 뱃지 재렌더 + 결과 갱신
export function onQualityInput() {
  renderQualityBadges();
  refreshFixedTags();
}
// 퀄리티 태그를 뱃지로 렌더(휠 가중치). 가중치는 input 텍스트에 로컬 표기로 되쓴다.
export function renderQualityBadges() {
  const c = $("quality_badges");
  if (!c) return;
  c.innerHTML = "";
  const toks = qualityTokens();
  toks.forEach((token, idx) => {
    const el = document.createElement("span");
    el.className = "tag deletable nl-phrase-badge";
    const { tag, weight } = parseTagWeight(token);
    el.textContent = formatTagWeight(tag, weight);  // 표시도 로컬/NAI 반영
    if (Math.abs(weight - 1.0) > 1e-9) el.classList.add("weighted");
    el.setAttribute("data-tooltip", "휠: 가중치 ±0.1 · 클릭: 삭제");
    el.onclick = () => {
      const t = qualityTokens();
      t.splice(idx, 1);
      $("fixed_quality").value = t.join(", ");
      onQualityInput();
    };
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      bumpQualityWeight(idx, e.deltaY < 0 ? 0.1 : -0.1);
    }, { passive: false });
    c.appendChild(el);
  });
}
function bumpQualityWeight(idx, delta) {
  const toks = qualityTokens();
  if (idx < 0 || idx >= toks.length) return;
  toks[idx] = bumpToken(toks[idx], delta);   // input 에는 항상 로컬 표기 저장
  $("fixed_quality").value = toks.join(", ");
  renderQualityBadges();
  refreshFixedTags();
}

// ── 작가 ──
let artistRowSeq = 0;
export function addArtistRow(name = "", weight = "1.0") {
  artistRowSeq += 1;
  const n = artistRowSeq;
  const row = document.createElement("div");
  row.className = "artist-row";
  row.dataset.idx = n;
  row.innerHTML = `
    <input type="text" class="cb-input artist-name" placeholder="작가명 (예: ratatatat74)"
      value="${name.replace(/"/g, "&quot;")}" />
    <input type="number" class="cb-input artist-weight" step="0.1" min="${state.weightSyntax === 'nai' ? '-10' : '0'}" value="${weight}"
      title="가중치 (1.0 = 기본)" />
    <button type="button" class="cb-remove artist-remove" title="삭제">×</button>
  `;
  row.querySelector(".artist-name").addEventListener("input", refreshFixedTags);
  row.querySelector(".artist-weight").addEventListener("input", refreshFixedTags);
  row.querySelector(".artist-remove").addEventListener("click", () => removeArtistRow(n));
  $("artist_rows").appendChild(row);
}
function removeArtistRow(n) {
  const el = document.querySelector(`.artist-row[data-idx="${n}"]`);
  if (el) el.remove();
  refreshFixedTags();
}
function parseFixedArtist() {
  const prefix = state.artistPrefix || "";   // "" / "@" / "artist:"
  const rows = document.querySelectorAll("#artist_rows .artist-row");
  const out = [];
  rows.forEach((row) => {
    let name = row.querySelector(".artist-name").value.trim();
    if (!name) return;
    name = name.replace(/^(@|artist:)/, "");       // 입력에 기존 접두사 있으면 일단 떼고
    if (prefix) name = prefix + name;              // 선택한 접두사 붙임(가중치 안으로)
    const wRaw = row.querySelector(".artist-weight").value.trim();
    const w = parseFloat(wRaw);
    if (!isNaN(w) && Math.abs(w - 1.0) > 1e-9) {
      // 1.0이 아니면 표기별 형식(로컬 (name:w) / NAI w::name::). formatTagWeight 로 통일.
      out.push(formatTagWeight(name, w));
    } else {
      out.push(name);                              // 1.0이면 그대로
    }
  });
  return out;
}
// 가중치 표기를 바꾸면 작가 weight 입력 하한도 맞춘다(NAI 음수 허용)
export function syncArtistWeightMin() {
  document.querySelectorAll("#artist_rows .artist-weight").forEach((inp) => {
    inp.min = (state.weightSyntax === "nai") ? "-10" : "0";
  });
}

// ── 패스스루(신규 캐릭터) ──
// DB를 못 뚫는 신규 캐릭터라 RAG 태그로는 못 나가므로, 사용자가 적은 캐릭터명을
// 프론트에서 태그 슬롯 맨 앞(퀄리티 고정태그 뒤)에 직접 끼운다.
// 원본(언더스코어 포함) 문자열을 반환 — 치환/이스케이프는 호출측에서 DB 태그와 동일하게 처리.
export function parsePassthroughNames() {
  if (!state.result || !Array.isArray(state.result.characters)) return [];
  const out = [];
  state.result.characters.forEach((ch) => {
    if (ch && ch.is_passthrough && ch.name && ch.name.trim()) {
      const nm = ch.name.trim();
      if (!out.includes(nm)) out.push(nm);
    }
  });
  return out;
}

// ── 슬롯 구성 ──
// 위치(앞/뒤)와 우선순위에 따라 고정태그(퀄리티/작가) 슬롯을 구성한다.
export function fixedSlots() {
  const q = parseFixedQuality();
  const a = parseFixedArtist();
  // 우선순위: 같은 위치일 때 먼저 올 것을 앞에 둔다.
  const order = state.fixedPriority === "artist"
    ? [["artist", a], ["quality", q]]
    : [["quality", q], ["artist", a]];
  const front = [], back = [];
  order.forEach(([which, tags]) => {
    (state.fixedPos[which] === "back" ? back : front).push(...tags);
  });
  return { front, back };
}
// DB 결과 태그 문자열에 고정 태그를 앞뒤로 붙여 최종 문자열 생성(복사용).
// 패스스루 이름에는 DB 태그와 동일한 치환/이스케이프를 적용(태그 슬롯에 들어가므로).
export function withFixedTags(dbTagString) {
  const { front, back } = fixedSlots();
  const opts = readFormatOptions();
  const pt = parsePassthroughNames().map((nm) => formatTagText(nm, opts));   // danbooru 블록 맨앞 붙박이
  const mid = (dbTagString || "").split(", ").map(t => t.trim()).filter(Boolean);
  return [...front, ...pt, ...mid, ...back].join(", ");
}
