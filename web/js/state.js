// 화면 전역 상태와 DOM 공용 헬퍼.
// 예전 인라인 스크립트의 최상위 let 변수들을 한 객체로 모았다(모듈끼리 같은 값을 공유).

export const state = {
  // 생성 결과(NDJSON data 이벤트를 누적). null 이면 아직 생성 전.
  result: null,
  // 직접조회 결과 배열. null 이면 조회 전.
  directResults: null,
  // 입력 모드: "basic"(한 문장) / "split"(인물 분할)
  mode: "basic",
  // 자연어 톤 프리셋: "rich" / "plain" / "phrase"
  nlTone: "rich",
  // 단어형+(phrase) 뱃지의 현재 가중치 문자열. danbooru final_prompt 와 분리해
  // NL 박스만 갱신한다(서로 오염 방지). 빈 문자열이면 뱃지 모드 아님.
  nlPhraseString: "",
  // 작가 태그 접두사: "" (없음) / "@" / "artist:"
  artistPrefix: "",
  // 고정태그 위치(앞/뒤) + 우선순위. 기본: 퀄리티=앞, 작가=뒤, 우선순위=퀄리티.
  fixedPos: { quality: "front", artist: "back" },
  fixedPriority: "quality", // 같은 위치일 때 누가 먼저 ("quality"/"artist")
  // 가중치 표기: "local" => (tag:weight) / "nai" => weight::tag::
  // 내부 상태(final_prompt, nlPhraseString)는 항상 로컬 표기로 저장하고,
  // 표시/복사 시 formatTagWeight 에서만 변환한다(단일 진실원천).
  weightSyntax: "local",
  // 직접조회 태그 클릭 동작: "copy" / "danbooru"
  directClickMode: "copy",
};

export const $ = (id) => document.getElementById(id);

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// 같은 그룹 버튼 중 하나만 active 로 표시한다.
export function setActiveIn(selector, isActive) {
  document.querySelectorAll(selector).forEach((b) => {
    b.classList.toggle("active", isActive(b));
  });
}
