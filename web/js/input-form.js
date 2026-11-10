// 입력 패널: 모드(기본/인물 분할), 인물칸, 자연어 톤, 검색 범위, 슬라이더 값 표시.
import { state, $ } from "./state.js";

// 자연어 톤 프리셋 선택. 톤을 고르면 nl_temp 슬라이더를 그 톤의 권장온도로
// 점프시킨다(프리셋 동작). 이후 사용자가 슬라이더로 미세조정 가능.
export function setNlTone(tone) {
  state.nlTone = tone;
  document.querySelectorAll("#nl_tone_group .tone-btn").forEach((b) => {
    const on = b.dataset.tone === tone;
    b.classList.toggle("active", on);
    if (on) {
      // 권장온도로 슬라이더/표시값 동기화
      const t = b.dataset.temp;
      const slider = $("nl_temp");
      const valEl = $("nl_temp_val");
      if (slider) slider.value = t;
      if (valEl) valEl.textContent = t;
    }
  });
}

// ── 태그 검색 범위 (카테고리 다중선택, 기본 모드 전용) ──
export function togglePoolCat(btn) {
  btn.classList.toggle("active");
}
// 현재 선택된 검색 범위 카테고리. 아무것도 안 켜져 있으면 null(=백엔드에서 일반 cat0 폴백).
export function getSearchPoolCategories() {
  const cats = Array.from(
    document.querySelectorAll("#search_pool_categories .cat-toggle.active")
  ).map((b) => parseInt(b.dataset.pool, 10));
  return cats.length ? cats : null;
}

export function setMode(mode) {
  state.mode = mode;
  const isSplit = mode === "split";
  $("basic_input").hidden = isSplit;
  $("split_input").hidden = !isSplit;
  $("mode_basic_btn").classList.toggle("active", !isSplit);
  $("mode_split_btn").classList.toggle("active", isSplit);
  // 검색 범위는 기본 모드 전용(고급 모드는 백엔드가 4-step 본선/cat3·4 검색 사용).
  const srs = $("search_range_section");
  if (srs) srs.hidden = isSplit;
  // 고급 모드 첫 진입 시 인물칸 1개 자동 생성
  if (isSplit && document.querySelectorAll("#char_blocks .char-block").length === 0) {
    addCharBlock();
  }
}

// ── 인물칸 ──
let charBlockSeq = 0;

export function addCharBlock() {
  charBlockSeq += 1;
  const n = charBlockSeq;
  const div = document.createElement("div");
  div.className = "char-block";
  div.dataset.idx = n;
  div.innerHTML = `
    <button type="button" class="cb-remove" title="삭제">×</button>
    <label class="cb-label">캐릭터 출처</label>
    <div class="source-seg" data-source="db">
      <button type="button" class="src-btn active" data-src="db">DB에서 찾기</button>
      <button type="button" class="src-btn" data-src="passthrough">DB에 없는 캐릭터</button>
      <button type="button" class="src-btn" data-src="original">오리지널</button>
    </div>
    <p class="src-hint help">DB에 등록된 캐릭터를 검색해 태그를 가져옵니다.</p>
    <div class="cb-row">
      <div class="cb-field">
        <label class="cb-label">작품명 (옵션)</label>
        <input type="text" class="cb-input cb-series" placeholder="예: 원신 / Genshin Impact" />
      </div>
      <div class="cb-field">
        <label class="cb-label">캐릭터명 (옵션)</label>
        <input type="text" class="cb-input cb-name" placeholder="예: raiden shogun / 임시이름" />
      </div>
    </div>
    <label class="cb-label">캐릭터 묘사 (한국어)</label>
    <textarea class="cb-textarea cb-desc" rows="2" placeholder="예: 은발, 보라눈, 보라색 기모노, 우산을 든 포즈"></textarea>
  `;
  div.querySelector(".cb-remove").addEventListener("click", () => removeCharBlock(n));
  div.querySelectorAll(".src-btn").forEach((b) => {
    b.addEventListener("click", () => setCharSource(b, b.dataset.src));
  });
  $("char_blocks").appendChild(div);
}

// 캐릭터 출처 3택: db(기본) / passthrough(신규) / original(오리지널).
// is_original/is_passthrough 불리언은 이 단일 상태에서 payload 수집 시 파생한다(collectCharacters).
const SRC_HINT = {
  db: "DB에 등록된 캐릭터를 검색해 태그를 가져옵니다.",
  passthrough: "DB에 없는 신규 캐릭터입니다. 입력한 이름을 그대로 자연어와 태그 맨 앞에 사용합니다.",
  original: "오리지널 캐릭터입니다. 작품/캐릭터명 없이 묘사만 쓰며, 이름은 자동 생성됩니다.",
};
function setCharSource(btn, src) {
  const block = btn.closest(".char-block");
  const seg = block.querySelector(".source-seg");
  seg.dataset.source = src;
  seg.querySelectorAll(".src-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.src === src);
  });
  const hint = block.querySelector(".src-hint");
  if (hint) hint.textContent = SRC_HINT[src] || "";
  const series = block.querySelector(".cb-series");
  const name = block.querySelector(".cb-name");
  // 오리지널: 작품/캐릭터명 비활성+비움(이름 자동 생성, 흐림은 CSS :disabled). db/passthrough: 입력 살림.
  const lock = src === "original";
  [series, name].forEach((el) => {
    el.disabled = lock;
    if (lock) el.value = "";
  });
  // passthrough 는 시각적으로 점선 표시(DB 미경유 신규).
  block.classList.toggle("is-passthrough", src === "passthrough");
}

function removeCharBlock(n) {
  const blocks = document.querySelectorAll("#char_blocks .char-block");
  if (blocks.length <= 1) return; // 최소 1칸 유지
  const el = document.querySelector(`#char_blocks .char-block[data-idx="${n}"]`);
  if (el) el.remove();
}

// 분할 모드 요청의 characters 배열. 출처 세그먼트 상태에서 is_original/is_passthrough 를 파생.
export function collectCharacters() {
  const characters = [];
  document.querySelectorAll("#char_blocks .char-block").forEach((blk) => {
    const seg = blk.querySelector(".source-seg");
    const src = seg ? seg.dataset.source : "db";
    const isOriginal = src === "original";
    const isPassthrough = src === "passthrough";
    // 오리지널은 이름/작품을 비우지만, db/passthrough 는 살려둔다(이름을 NL 핸들로 사용).
    const name = isOriginal ? "" : blk.querySelector(".cb-name").value.trim();
    const series = isOriginal ? "" : blk.querySelector(".cb-series").value.trim();
    const desc = blk.querySelector(".cb-desc").value.trim();
    if (name || series || desc) {
      characters.push({ name, series, desc, is_original: isOriginal, is_passthrough: isPassthrough });
    }
  });
  return characters;
}

// ── 슬라이더: 값 표시 동기화 ──
export function bindSliderValue(sliderId, valueId, digits) {
  $(sliderId).addEventListener("input", (e) => {
    $(valueId).textContent = parseFloat(e.target.value).toFixed(digits);
  });
}
