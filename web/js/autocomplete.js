// 태그 자동완성 드롭다운 (anima-webgpu 포팅, textarea 1개).
// 상태/뱃지 동기화는 호출측(onChoose)이 맡고, 여기서는 후보 표시와 삽입만 한다.
import { ac, acEnsureLoaded, acSearch } from "./tag-data.js";

function acFmtCount(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}
function acEscape(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}
function acHighlight(text, q) {
  const esc = acEscape(text);
  if (!q) return esc;
  const idx = text.toLowerCase().indexOf(q.toLowerCase());
  if (idx < 0) return esc;
  return acEscape(text.slice(0, idx)) + "<mark>" +
         acEscape(text.slice(idx, idx + q.length)) + "</mark>" +
         acEscape(text.slice(idx + q.length));
}
// 커서 기준 현재 토큰 [경계, 단어부 오프셋, 검색어]. 가중치/괄호 문법 제외.
function acCurrentToken(text, caret) {
  let start = text.lastIndexOf(",", caret - 1) + 1;
  let end = text.indexOf(",", caret);
  if (end < 0) end = text.length;
  let inner = text.slice(start, end);
  let lead = 0;
  const mLead = inner.match(/^[\s(]*/);
  if (mLead) { lead = mLead[0].length; inner = inner.slice(lead); }
  const mTail = inner.match(/(:[+-]?\d*\.?\d*)?[\s)]*$/);
  if (mTail && mTail[0]) inner = inner.slice(0, inner.length - mTail[0].length);
  const wordStart = start + lead;
  return { tokenStart: start, tokenEnd: end, wordStart, wordEnd: wordStart + inner.length, query: inner };
}

export function makeAutocomplete(ta, pop, { onChoose } = {}) {
  if (!ta || !pop) return;
  let items = [];
  let active = -1;
  let token = null;
  let debounce = 0;

  function close() {
    pop.innerHTML = "";
    pop.style.display = "none";
    items = []; active = -1; token = null;
  }
  function render() {
    if (!items.length) { close(); return; }
    const q = token ? token.query : "";
    pop.innerHTML = items.map((r, i) => {
      const aliasHtml = r.alias ? `<span class="ac-alias">${acHighlight(r.alias, q)}</span>` : "";
      const catHtml = r.cat ? `<span class="ac-badge ac-badge-cat">${acEscape(r.cat)}</span>` : "";
      return (
        `<div class="ac-item${i === active ? " active" : ""}" data-i="${i}">` +
        `<div class="ac-row1">` +
        `<span class="ac-tag">${acHighlight(r.tag.replace(/_/g, " "), q)}</span>` +
        `<span class="ac-count">${acFmtCount(r.pc)}</span>` +
        `</div>` +
        `<div class="ac-row2">${catHtml}${aliasHtml}</div>` +
        `</div>`
      );
    }).join("");
    pop.style.display = "block";
    [...pop.children].forEach((el) => {
      el.onmousedown = (e) => { e.preventDefault(); choose(+el.dataset.i); };
    });
    scrollActiveIntoView();
  }
  function scrollActiveIntoView() {
    if (active < 0) return;
    const el = pop.children[active];
    if (el) el.scrollIntoView({ block: "nearest" });
  }
  // 선택 → 영어 태그를 현재 토큰 단어부에 치환(언더스코어 유지). 뱃지 동기화.
  function choose(i) {
    const r = items[i];
    if (!r || !token) return close();
    const before = ta.value.slice(0, token.wordStart);
    const after = ta.value.slice(token.wordEnd);
    const tagOut = r.tag;   // danbooru 태그는 언더스코어 유지(뱃지가 표기 설정 반영)
    const trimmedAfter = after.replace(/^\s*/, "");
    const needsComma = !trimmedAfter.startsWith(",");
    const insert = tagOut + (needsComma ? ", " : "");
    const newCaret = before.length + insert.length;
    ta.value = before + insert + after;
    ta.setSelectionRange(newCaret, newCaret);
    ta.focus();
    close();
    if (onChoose) onChoose();
  }
  function update() {
    if (!ac.ready) { acEnsureLoaded(); return; }
    const caret = ta.selectionStart;
    token = acCurrentToken(ta.value, caret);
    const q = token.query;
    if (!q || q.length < 1) { close(); return; }
    items = acSearch(q);
    active = items.length ? 0 : -1;
    render();
  }

  ta.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(update, 150);
  });
  ta.addEventListener("keydown", (e) => {
    if (!items.length) return;
    if (e.key === "ArrowDown") { e.preventDefault(); active = (active + 1) % items.length; render(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); }
    else if (e.key === "Enter" || e.key === "Tab") { if (active >= 0) { e.preventDefault(); choose(active); } }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
  });
  ta.addEventListener("blur", () => setTimeout(close, 120));
  ta.addEventListener("click", () => { clearTimeout(debounce); debounce = setTimeout(update, 150); });
}
