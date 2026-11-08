// 결과 패널: Danbooru 태그 뱃지(휠 가중치·클릭 삭제), 편집칸 동기화, 자연어/키워드/인물 그리드, 복사.
import { state, $, escapeHtml, setActiveIn } from "./state.js";
import {
  parseTagWeight, formatTagWeight, bumpToken, readFormatOptions, formatTagText,
  applyTagFormatting, splitTokens,
} from "./tag-format.js";
import { tagInfo } from "./tag-data.js";
import {
  fixedSlots, withFixedTags, parsePassthroughNames, renderQualityBadges, syncArtistWeightMin,
} from "./fixed-tags.js";
import { renderDirectDBResults } from "./direct-search.js";
import { doCopy } from "./clipboard.js";

const EMPTY_HTML = (text) => `<span style="color: var(--dim); font-size: 13px;">${text}</span>`;

// 가중치 표기 세그먼트(로컬/NAI)
export function setWeightSyntax(btn) {
  state.weightSyntax = btn.dataset.syntax;
  setActiveIn("#weight_syntax_group .seg-btn", (b) => b === btn);
  syncArtistWeightMin();
  renderQualityBadges();   // 퀄리티 뱃지 표기 갱신
  if (state.result && state.result.final_prompt) displayResults();
  else if (state.nlPhraseString) renderNlPhraseBadges(state.nlPhraseString);
}

// 출력 형식 체크박스 변경
export function updatePromptDisplay() {
  if (state.result) displayResults();
  if (state.directResults) renderDirectDBResults();
}

export function displayResults() {
  if (!state.result) return;
  const data = state.result;
  const opts = readFormatOptions();

  $("empty_state").style.display = "none";
  $("results_container").style.display = "flex";

  // 1. 자연어 프롬프트 처리
  const nlPromptBox = $("nl_prompt_box");
  if (data.english_prompt) {
    nlPromptBox.style.display = "block";
    // 단어형+(phrase)는 danbooru 그리드처럼 뱃지 + 휠 가중치로 렌더.
    // 그 외 톤(rich/plain)은 기존대로 평문 문장.
    if (state.nlTone === "phrase") {
      renderNlPhraseBadges(data.english_prompt);
    } else {
      state.nlPhraseString = "";
      const nlEl = $("res_nl_prompt");
      nlEl.classList.remove("tag-list");
      nlEl.textContent = data.english_prompt;
    }
  } else {
    nlPromptBox.style.display = "none";
  }

  // 2. 최종 단보루 프롬프트 처리
  const finalPromptBox = $("res_final_prompt").closest(".result-box");
  if (data.final_prompt) {
    finalPromptBox.style.display = "block";
    const finalContainer = $("res_final_prompt");
    finalContainer.innerHTML = "";

    // 고정 뱃지 (클릭 삭제 없음, 색상 토큰 1개)
    const addFixedBadge = (txt) => {
      const b = document.createElement("span");
      b.className = "tag fixed-badge";
      // 고정 태그는 언더스코어 치환/괄호 이스케이프를 받지 않고 입력 그대로 출력
      b.textContent = txt;
      b.setAttribute("data-tooltip", "고정 태그 (LLM/DB 미경유, 치환·이스케이프 미적용)");
      finalContainer.appendChild(b);
    };
    // 앞 슬롯: 위치=맨앞으로 지정된 고정태그(우선순위순)
    const { front: fixedFront, back: fixedBack } = fixedSlots();
    fixedFront.forEach(addFixedBadge);

    // 퀄리티 뒤, DB 태그 앞: 패스스루(신규 캐릭터) 이름 배지.
    // DB를 못 뚫는 신규 캐릭터라 RAG 태그로는 안 나가므로 프론트에서 직접 끼운다.
    parsePassthroughNames().forEach((txt) => {
      const b = document.createElement("span");
      b.className = "tag passthrough-badge";
      b.textContent = formatTagText(txt, opts);
      b.setAttribute("data-tooltip", "패스스루 (신규 캐릭터 — DB 미경유, 사용자 입력 이름)");
      finalContainer.appendChild(b);
    });

    const tagsArray = splitTokens(data.final_prompt);

    tagsArray.forEach((token, idx) => {
      const el = document.createElement("span");
      el.className = "tag deletable db-tag";
      // 토큰은 "tag" 또는 "(tag:1.2)". 가중치를 분리해 코어 태그로 설명/색상 매칭.
      const { tag: coreTag, weight } = parseTagWeight(token);
      let tooltip;
      const cleanTag = coreTag.trim();

      if (tagInfo.has(cleanTag)) {
        const info = tagInfo.get(cleanTag);
        const descText = info.desc;
        tooltip = (descText ? descText : "설명 없음") + "\n\n(휠: 가중치 ±0.1 · 클릭: 삭제)";
        el.classList.add("tag-cat-" + info.category);
      } else {
        tooltip = "설명을 찾을 수 없습니다.\n\n(휠: 가중치 ±0.1 · 클릭: 삭제)";
      }
      el.setAttribute("data-tooltip", tooltip);

      // 표시: 코어 태그에만 치환/이스케이프 적용, 가중치 표기는 그대로 감싼다.
      el.textContent = formatTagWeight(formatTagText(coreTag, opts), weight);
      if (Math.abs(weight - 1.0) > 1e-9) el.classList.add("weighted");

      // 클릭 삭제 — 위치(idx) 기반으로 정확히 그 토큰만 제거(중복 태그 안전).
      el.onclick = () => {
        const toks = splitTokens(state.result.final_prompt);
        toks.splice(idx, 1);
        state.result.final_prompt = toks.join(", ");
        syncFinalEditFromData();
        displayResults();
      };

      // 휠 가중치 조절(호버 중). DB 태그 뱃지에만 건다(고정/패스스루/그리드 제외).
      el.addEventListener("wheel", (e) => {
        e.preventDefault();          // 페이지 스크롤 방지
        const delta = e.deltaY < 0 ? 0.1 : -0.1;   // 위로 굴리면 +, 아래로 -
        bumpTagWeight(idx, delta);
      }, { passive: false });

      finalContainer.appendChild(el);
    });

    // 뒤 슬롯: 위치=맨뒤로 지정된 고정태그(우선순위순)
    fixedBack.forEach(addFixedBadge);

    if (tagsArray.length === 0 && fixedFront.length === 0 && fixedBack.length === 0) {
      finalContainer.innerHTML = EMPTY_HTML("매칭된 태그가 없습니다.");
    }

    // textarea(편집칸)를 데이터와 동기화. 단 포커스 중이면 덮어쓰지 않음.
    syncFinalEditFromData();
  } else {
    finalPromptBox.style.display = "none";
  }

  // 3. 키워드 처리 (한국어 + 영어)
  const kwBox = $("keywords_box");
  const hasKo = data.ko_units && data.ko_units.length > 0;
  const hasEn = data.keywords && data.keywords.length > 0;
  if (hasKo || hasEn) {
    kwBox.style.display = "block";

    const koContainer = $("res_keywords_ko");
    koContainer.innerHTML = "";
    if (hasKo) {
      data.ko_units.forEach((kw) => {
        const el = document.createElement("span");
        el.className = "tag";
        el.textContent = kw;   // 한국어는 치환/이스케이프 불필요
        koContainer.appendChild(el);
      });
    }

    const enContainer = $("res_keywords_en");
    enContainer.innerHTML = "";
    if (hasEn) {
      data.keywords.forEach((kw) => {
        const el = document.createElement("span");
        el.className = "tag";
        el.textContent = formatTagText(kw, opts);
        enContainer.appendChild(el);
      });
    }
  } else {
    kwBox.style.display = "none";
  }

  // 4. 인물 그리드 (분할 모드 전용)
  renderCharGrid(data, opts);
}

function renderCharGrid(data, opts) {
  const gridBox = $("char_grid_box");
  const chars = data.characters || [];
  const bgTags = data.background_tags || [];
  if (!(chars.length > 0 || bgTags.length > 0)) {
    gridBox.style.display = "none";
    return;
  }
  gridBox.style.display = "block";
  const grid = $("res_char_grid");
  grid.innerHTML = "";
  const tagHtml = (t) => '<span class="cc-tag">' + escapeHtml(formatTagText(t, opts)) + "</span>";

  chars.forEach((ch, i) => {
    const card = document.createElement("div");
    card.className = "char-card";

    const head = ch.name
      ? escapeHtml(ch.name)
      : '<span style="color:var(--dim);">인물 ' + (ch.index || i + 1) + "</span>";
    const series = ch.series
      ? ' <span class="cc-series">[' + escapeHtml(ch.series) + "]</span>"
      : "";
    const miss = ch.name_unmatched
      ? ' <span class="cc-miss" title="DB에 없는 캐릭터 — 이름은 자연어에만 사용">DB 미검색</span>'
      : "";
    const pt = ch.is_passthrough
      ? ' <span class="cc-pass" title="패스스루 — DB 미경유 신규 캐릭터, 이름을 태그 맨 앞+자연어에 사용">패스스루</span>'
      : "";

    const tags = (ch.tags || []).map(tagHtml).join("");

    card.innerHTML =
      '<div class="cc-head">' + head + series + miss + pt + "</div>" +
      '<div class="cc-tags">' +
      (tags || '<span style="color:var(--dim);font-size:12px;">태그 없음</span>') +
      "</div>";
    grid.appendChild(card);
  });

  // 배경/공통 카드 (있을 때만)
  if (bgTags.length > 0) {
    const bg = document.createElement("div");
    bg.className = "char-card char-card-bg";
    bg.innerHTML =
      '<div class="cc-head">\u{1F310} 배경 / 공통</div>' +
      '<div class="cc-tags">' + bgTags.map(tagHtml).join("") + "</div>";
    grid.appendChild(bg);
  }
}

// 휠로 특정 토큰의 가중치를 delta(±0.1)만큼 조절하고 final_prompt 갱신.
// idx 는 data.final_prompt 분해 배열에서의 위치(중복 태그 안전하게 위치 기반).
function bumpTagWeight(idx, delta) {
  if (!state.result || !state.result.final_prompt) return;
  const tokens = splitTokens(state.result.final_prompt);
  if (idx < 0 || idx >= tokens.length) return;
  // 내부 상태는 항상 로컬 표기로 저장(NAI 변환은 표시/복사 시 formatTagWeight 가).
  tokens[idx] = bumpToken(tokens[idx], delta);
  state.result.final_prompt = tokens.join(", ");
  syncFinalEditFromData();        // textarea 동기화
  displayResults();               // 뱃지 재렌더(가중치 표시 갱신)
}

// 단어형+(phrase) 출력을 danbooru 그리드처럼 뱃지로 렌더한다.
// danbooru 와 차이: 카테고리 색(tag-cat-N) 미적용(키워드 뱃지 스타일), 가중치
// 상태는 state.nlPhraseString 에만 저장(final_prompt 와 분리).
function renderNlPhraseBadges(phraseString) {
  state.nlPhraseString = phraseString || "";
  const c = $("res_nl_prompt");
  c.classList.add("tag-list");
  c.innerHTML = "";
  const tokens = splitTokens(state.nlPhraseString);
  if (tokens.length === 0) {
    c.innerHTML = EMPTY_HTML("생성된 구가 없습니다.");
    return;
  }
  tokens.forEach((token, idx) => {
    const el = document.createElement("span");
    el.className = "tag deletable nl-phrase-badge";
    const { tag: core, weight } = parseTagWeight(token);
    el.textContent = formatTagWeight(core, weight);  // 구는 치환/이스케이프 미적용(자연어)
    if (Math.abs(weight - 1.0) > 1e-9) el.classList.add("weighted");
    el.setAttribute("data-tooltip", "휠: 가중치 ±0.1 · 클릭: 삭제");
    // 클릭 삭제 (위치 기반)
    el.onclick = () => {
      const toks = splitTokens(state.nlPhraseString);
      toks.splice(idx, 1);
      renderNlPhraseBadges(toks.join(", "));
    };
    // 휠 가중치 (호버 중)
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      bumpNlPhraseWeight(idx, e.deltaY < 0 ? 0.1 : -0.1);
    }, { passive: false });
    c.appendChild(el);
  });
}

// 단어형+ 뱃지의 가중치를 delta(±0.1)만큼 조절. final_prompt 와 무관하게
// state.nlPhraseString 만 갱신 후 재렌더.
function bumpNlPhraseWeight(idx, delta) {
  const tokens = splitTokens(state.nlPhraseString);
  if (idx < 0 || idx >= tokens.length) return;
  tokens[idx] = bumpToken(tokens[idx], delta);
  renderNlPhraseBadges(tokens.join(", "));
}

// final_prompt(데이터) → textarea 표시. textarea 는 DB 태그만(고정태그 제외).
export function syncFinalEditFromData() {
  const ta = $("final_prompt_edit");
  if (!ta) return;
  // textarea 가 포커스 중이면 사용자가 타이핑 중일 수 있으니 덮어쓰지 않음.
  if (document.activeElement === ta) return;
  ta.value = (state.result && state.result.final_prompt) || "";
}

// textarea 입력 → final_prompt(데이터) 갱신. 입력 중에는 뱃지 재렌더 안 함(커서 보존).
export function onFinalEditInput() {
  const ta = $("final_prompt_edit");
  if (!ta || !state.result) return;
  // 쉼표 구분 그대로 저장(가중치 표기 (tag:1.2) 포함 가능). 빈 토큰 정리.
  const tokens = ta.value.split(",").map(t => t.trim()).filter(Boolean);
  state.result.final_prompt = tokens.join(", ");
}

// textarea blur → 정규화된 값으로 뱃지 재렌더 + textarea 재동기화.
export function onFinalEditBlur() {
  // 드롭다운 항목 클릭(mousedown)이 blur보다 먼저 처리되도록 약간 지연 후 닫기.
  setTimeout(hideAc, 150);
  onFinalEditInput();
  displayResults();
}

// ── 복사 ──
// 복사 성공 표시: 잠깐 강조색으로 바꿨다가 되돌린다.
function flashCopied(btn, text, restoreText, ms) {
  const originalText = restoreText ?? btn.textContent;
  const originalBg = btn.style.background;
  const originalColor = btn.style.color;

  btn.textContent = text;
  btn.style.color = "#0b0e14";
  btn.style.borderColor = "var(--primary)";
  btn.style.background = "var(--primary)";

  setTimeout(() => {
    btn.textContent = originalText;
    btn.style.color = originalColor;
    btn.style.borderColor = "";
    btn.style.background = originalBg;
  }, ms);
}

// 결과의 Danbooru 태그 최종 문자열(복사용): DB 태그에만 치환/이스케이프(가중치 표기 보존),
// 고정 태그는 원형 유지.
function finalTagsForCopy() {
  return withFixedTags(applyTagFormatting(state.result.final_prompt || "", readFormatOptions()));
}

export function copyToClipboard(elementId) {
  let text = "";
  if (elementId === "res_final_prompt") {
    text = finalTagsForCopy();
  } else if (elementId === "res_nl_prompt" && state.nlTone === "phrase" && state.nlPhraseString) {
    // 단어형+ 뱃지 모드: textContent(뱃지 텍스트)가 아니라 가중치 문자열을 복사.
    text = state.nlPhraseString;
  } else {
    text = $(elementId).textContent;
  }

  doCopy(text, () => {
    const btn = $(elementId).closest(".result-box").querySelector(".copy-btn");
    flashCopied(btn, "복사 완료!", undefined, 2000);
  });
}

export function copyAll() {
  const finalTags = finalTagsForCopy();
  // 단어형+ 뱃지 모드면 가중치 문자열, 아니면 평문 textContent.
  const nlPrompt = (state.nlTone === "phrase" && state.nlPhraseString)
    ? state.nlPhraseString.trim()
    : $("res_nl_prompt").textContent.trim();

  let combined = "";
  if (finalTags && finalTags !== "매칭된 태그가 없습니다." && nlPrompt) {
    combined = finalTags + ". " + nlPrompt;
  } else if (finalTags && finalTags !== "매칭된 태그가 없습니다.") {
    combined = finalTags;
  } else if (nlPrompt) {
    combined = nlPrompt;
  }

  if (!combined) return;

  doCopy(combined, () => {
    flashCopied($("copy_all_btn"), "모두 복사 완료!", "모두 복사하기 (단보루 태그 + 자연어)", 2000);
  });
}

export { flashCopied };
