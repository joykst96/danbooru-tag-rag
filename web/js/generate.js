// 태그 생성: 입력 수집 → /api/generate(_split) NDJSON 스트림 → 진행 상태·대기열 표시 → 결과 누적·렌더.
import { state, $, escapeHtml } from "./state.js";
import { streamNdjson } from "./api.js";
import { collectCharacters, getSearchPoolCategories } from "./input-form.js";
import { displayResults } from "./results.js";
import { acEnsureLoaded } from "./tag-data.js";

export function showDuplicateModal(msg) {
  const m = $("dup_modal");
  const t = $("dup_modal_msg");
  if (t) t.textContent = msg || "동일 IP로 처리 중인 작업이 있습니다.";
  if (m) m.hidden = false;
}
export function closeDuplicateModal() {
  const m = $("dup_modal");
  if (m) m.hidden = true;
}

function showError(msg) {
  const errorEl = $("error_msg");
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

function resetButton() {
  const btn = $("generate_btn");
  btn.disabled = false;
  $("spinner").hidden = true;
  btn.querySelector("span").textContent = "태그 생성하기";
}

// 요청 본문. 입력이 비었으면 오류 문구를 돌려준다.
function buildRequest() {
  if (state.mode === "split") {
    const characters = collectCharacters();
    const background = $("bg_desc").value.trim();
    if (characters.length === 0 && !background) {
      return { error: "인물 또는 배경을 하나 이상 입력해주세요." };
    }
    return {
      url: "/api/generate_split",
      payload: {
        characters,
        background,
        nl_temperature: parseFloat($("nl_temp").value),
        nl_tone: state.nlTone,
        include_phrases: $("include_phrases").checked,
      },
    };
  }
  const payload = {
    prompt: $("prompt").value.trim(),
    nl_temperature: parseFloat($("nl_temp").value),
    nl_tone: state.nlTone,
    search_categories: getSearchPoolCategories(),
    include_phrases: $("include_phrases").checked,
  };
  if (!payload.prompt) return { error: "이미지 묘사를 입력해주세요." };
  return { url: "/api/generate", payload };
}

// 대기 메시지는 결과 영역(빈 상태)에 크게 표시
function showQueueStatus(status) {
  const qm = $("queue_msg");
  if (!qm) return;
  if (status.indexOf("대기") !== -1) {
    qm.textContent = "⏳ " + status;
    qm.hidden = false;
  } else {
    qm.hidden = true;
  }
}

export async function generateTags() {
  const btn = $("generate_btn");
  const btnText = btn.querySelector("span");

  // 초기화
  $("error_msg").hidden = true;
  $("results_container").hidden = true;
  $("empty_state").hidden = false;
  btn.disabled = true;
  $("spinner").hidden = false;
  btnText.textContent = "생성 준비 중...";

  state.result = {};
  // 새 생성 시작: 편집 textarea 비움(이전 결과 잔류 방지)
  const fpEdit = $("final_prompt_edit");
  if (fpEdit) fpEdit.value = "";

  const splitUnmatchedNames = [];
  const note = $("unmatched_note");
  if (note) { note.hidden = true; note.innerHTML = ""; }
  const grid = $("char_grid_box");
  if (grid) grid.hidden = true;

  const req = buildRequest();
  if (req.error) {
    showError(req.error);
    resetButton();
    return;
  }
  // 결과 뱃지 호버 설명용 태그 데이터. 생성(수 초)과 함께 받기 시작하고, 기다리지 않는다.
  acEnsureLoaded();

  try {
    await streamNdjson(req.url, req.payload, (chunk) => {
      if (chunk.status === "rejected" || chunk.reason === "duplicate_ip") {
        showDuplicateModal(chunk.message || "동일 IP로 처리 중인 작업이 있습니다.");
        return false;   // 스트림 중단
      }
      if (chunk.status) {
        btnText.textContent = chunk.status;
        showQueueStatus(chunk.status);
      }
      if (chunk.data && Object.keys(chunk.data).length > 0) {
        // 분할모드: 인물별 character 이벤트는 배열로 누적(그리드 + 미스 표시)
        if (chunk.data.character) {
          const ch = chunk.data.character;
          if (!state.result) state.result = {};
          if (!state.result.characters) state.result.characters = [];
          state.result.characters.push(ch);
          if (ch.name_unmatched && ch.name) {
            splitUnmatchedNames.push(ch.name);
            note.hidden = false;
            note.innerHTML =
              '<span class="name-unmatched-note">⚠ DB에 없는 캐릭터(태그 미검색, 자연어에만 이름 사용): ' +
              splitUnmatchedNames.map(escapeHtml).join(", ") + "</span>";
          }
          delete chunk.data.character; // spread 대상에서 제외(이미 배열에 넣음)
        }
        state.result = { ...state.result, ...chunk.data };
        displayResults();
      }
    });
  } catch (err) {
    showError(err.message || "API 서버에 연결할 수 없습니다.");
  } finally {
    resetButton();
  }
}
