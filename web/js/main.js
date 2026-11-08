// 진입점: 마크업의 버튼·입력에 동작을 연결하고 초기 상태를 만든다.
// (예전 인라인 onclick/oninput 속성을 여기로 옮겼다. 마크업은 index.html 에만 있다.)
import { state, $ } from "./state.js";
import { initTooltip } from "./tooltip.js";
import { acEnsureLoaded, setOnTagDataBuilt } from "./tag-data.js";
import { makeAutocomplete } from "./autocomplete.js";
import {
  setMode, addCharBlock, setNlTone, togglePoolCat, bindSliderValue,
} from "./input-form.js";
import {
  setArtistPrefix, setFixedPos, setFixedPriority, onQualityInput, renderQualityBadges, addArtistRow,
} from "./fixed-tags.js";
import {
  displayResults, setWeightSyntax, updatePromptDisplay, onFinalEditInput, onFinalEditBlur,
  copyToClipboard, copyAll,
} from "./results.js";
import { toggleCat, setDirectClickMode, searchDirectDB } from "./direct-search.js";
import { generateTags, closeDuplicateModal } from "./generate.js";

const on = (el, type, fn) => el.addEventListener(type, fn);
const each = (selector, fn) => document.querySelectorAll(selector).forEach(fn);

initTooltip();

// ── 입력 패널 ──
on($("mode_basic_btn"), "click", () => setMode("basic"));
on($("mode_split_btn"), "click", () => setMode("split"));
on($("add_char_btn"), "click", () => addCharBlock());
each("#nl_tone_group .tone-btn", (b) => on(b, "click", () => setNlTone(b.dataset.tone)));
each("#search_pool_categories .cat-toggle", (b) => on(b, "click", () => togglePoolCat(b)));
on($("replace_underscores"), "change", updatePromptDisplay);
on($("escape_parentheses"), "change", updatePromptDisplay);
each("#weight_syntax_group .seg-btn", (b) => on(b, "click", () => setWeightSyntax(b)));

// ── 고정 태그 ──
each("#quality_pos_group .seg-btn", (b) => on(b, "click", () => setFixedPos("quality", b)));
each("#artist_pos_group .seg-btn", (b) => on(b, "click", () => setFixedPos("artist", b)));
each("#artist_prefix_group .seg-btn", (b) => on(b, "click", () => setArtistPrefix(b)));
each("#fixed_priority_group .seg-btn", (b) => on(b, "click", () => setFixedPriority(b)));
on($("fixed_quality"), "input", onQualityInput);
on($("add_artist_btn"), "click", () => addArtistRow());

// ── 슬라이더 값 표시 ──
bindSliderValue("threshold", "threshold_val", 2);
bindSliderValue("kw_temp", "kw_temp_val", 1);
bindSliderValue("nl_temp", "nl_temp_val", 1);
bindSliderValue("direct_threshold", "direct_threshold_val", 2);

// ── 생성 ──
on($("generate_btn"), "click", generateTags);
on(document.querySelector("#dup_modal .dup-modal-btn"), "click", closeDuplicateModal);

// ── 직접조회 ──
each("#direct_categories .cat-toggle", (b) => on(b, "click", () => toggleCat(b)));
each("#direct_click_mode .mode-btn", (b) => on(b, "click", () => setDirectClickMode(b.dataset.clickmode)));
on($("direct_search_btn"), "click", searchDirectDB);

// ── 결과 패널 ──
// 편집칸: 상태 동기화 리스너를 자동완성보다 먼저 건다(예전 인라인 속성과 같은 순서).
const finalEdit = $("final_prompt_edit");
on(finalEdit, "input", onFinalEditInput);
on(finalEdit, "blur", onFinalEditBlur);
makeAutocomplete(finalEdit, $("ac_dropdown"), { onChoose: onFinalEditInput });
// 첫 포커스 시 자동완성 데이터 로드(페이지 로드를 막지 않음)
finalEdit.addEventListener("focus", acEnsureLoaded, { once: true });
// 데이터를 새로 빌드했고 결과가 이미 화면에 있으면 뱃지 카테고리 색 갱신
setOnTagDataBuilt(() => {
  if (state.result && state.result.final_prompt) displayResults();
});

each(".copy-btn[data-copy]", (b) => on(b, "click", () => copyToClipboard(b.dataset.copy)));
on($("copy_all_btn"), "click", copyAll);

// ── 초기 상태 ──
addArtistRow();
renderQualityBadges();
