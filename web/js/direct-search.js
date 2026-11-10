// 벡터DB 단일태그 직접조회: LLM 없이 입력 단어로 /api/direct_search 를 부르고 결과 뱃지를 그린다.
import { state, $, setActiveIn } from "./state.js";
import { readFormatOptions, formatTagText } from "./tag-format.js";
import { tagInfo } from "./tag-data.js";
import { directSearch } from "./api.js";
import { doCopy } from "./clipboard.js";

export function toggleCat(btn) {
  btn.classList.toggle("active");
}

export function setDirectClickMode(mode) {
  state.directClickMode = mode;
  setActiveIn("#direct_click_mode .mode-btn", (b) => b.dataset.clickmode === mode);
  if (state.directResults) renderDirectDBResults(); // 이미 뜬 결과의 클릭 동작 갱신
}

export async function searchDirectDB() {
  const query = $("direct_query").value.trim();
  const threshold = parseFloat($("direct_threshold").value);

  const btn = $("direct_search_btn");
  const spinner = $("direct_spinner");
  const btnText = btn.querySelector("span");
  const errorMsg = $("direct_error_msg");
  const resultContainer = $("res_direct_db");

  errorMsg.hidden = true;
  resultContainer.hidden = true;

  if (!query) {
    errorMsg.textContent = "조회할 단어를 입력해주세요.";
    errorMsg.hidden = false;
    return;
  }

  btn.disabled = true;
  spinner.hidden = false;
  btnText.textContent = "조회 중...";

  const categories = Array.from(
    document.querySelectorAll(".cat-toggle[data-cat].active")
  ).map((b) => parseInt(b.dataset.cat, 10));

  try {
    const data = await directSearch({ query, threshold, categories });
    state.directResults = data.results || [];
    renderDirectDBResults();
    resultContainer.hidden = false;
  } catch (err) {
    errorMsg.textContent = err.message || "API 서버에 연결할 수 없습니다.";
    errorMsg.hidden = false;
  } finally {
    btn.disabled = false;
    spinner.hidden = true;
    btnText.textContent = "조회하기";
  }
}

// 복사 성공 표시: 폭을 고정한 채 "복사!"로 잠깐 바꿨다가 되돌린다(색은 CSS .is-copied).
function flashTagCopied(el) {
  const originalText = el.textContent;
  el.style.width = el.offsetWidth + "px";
  el.textContent = "복사!";
  el.classList.add("is-copied");

  setTimeout(() => {
    el.textContent = originalText;
    el.style.width = "";
    el.classList.remove("is-copied");
  }, 1000);
}

export function renderDirectDBResults() {
  if (!state.directResults) return;
  const resultContainer = $("res_direct_db");
  const opts = readFormatOptions();

  resultContainer.innerHTML = "";

  if (state.directResults.length === 0) {
    resultContainer.innerHTML = '<span class="empty-note">매칭된 태그가 없습니다.</span>';
    return;
  }
  state.directResults.forEach((item) => {
    const el = document.createElement("span");
    el.className = "tag candidate copyable";

    const tagText = formatTagText(item.tag, opts);
    el.textContent = `${tagText} (${item.score.toFixed(2)})`;

    let descText = "";
    const cleanTag = item.tag.trim();
    if (tagInfo.has(cleanTag)) {
      const info = tagInfo.get(cleanTag);
      descText = info.desc;
      el.classList.add("tag-cat-" + info.category);
    }

    const tooltipDesc = descText ? descText : item.tag + "\n\n(설명 없음)";
    const clickHint = state.directClickMode === "danbooru"
      ? "\n\n(클릭하여 Danbooru 열기)"
      : "\n\n(클릭하여 복사)";
    el.setAttribute("data-tooltip", tooltipDesc + clickHint);

    if (state.directClickMode === "danbooru") {
      el.onclick = () => {
        // Danbooru 태그 검색은 원본 언더스코어 형태 사용(공백/이스케이프 미적용)
        const url = "https://danbooru.donmai.us/posts?tags=" +
          encodeURIComponent(item.tag);
        window.open(url, "_blank", "noopener");
      };
    } else {
      el.onclick = () => doCopy(tagText, () => flashTagCopied(el));
    }

    resultContainer.appendChild(el);
  });
}
