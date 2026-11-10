// 태그 문자열 표기: 가중치 (tag:weight) / NAI weight::tag:: 변환, 언더스코어 치환·괄호 이스케이프.
import { state, $ } from "./state.js";

// final_prompt 문자열의 각 요소는 "tag" 또는 "(tag:1.2)" 형태일 수 있다.
// 가중치 1.0 은 괄호 없이 "tag" 로 표기(기본값).
const WEIGHT_RE = /^\((.+):\s*(-?\d+(?:\.\d+)?)\)$/;
// NAI 표기 역파싱: weight::tag:: (중첩 오염 복구 포함)
const NAI_WEIGHT_RE = /^(-?\d+(?:\.\d+)?)::([\s\S]+)::$/;

export function parseTagWeight(token) {
  // 반환: { tag, weight } — 가중치 없으면 weight=1.0
  const t = (token || "").trim();
  // 로컬 (tag:weight)
  let m = t.match(WEIGHT_RE);
  if (m) {
    const w = parseFloat(m[2]);
    return { tag: m[1].trim(), weight: isNaN(w) ? 1.0 : w };
  }
  // NAI weight::tag:: — 내부 상태에 NAI 표기가 섞여 들어와도(또는 과거 오염
  // 데이터) 역파싱해 휠 중첩 누적을 막는다. 최외곽 weight 채택, 중첩은 벗김.
  m = t.match(NAI_WEIGHT_RE);
  if (m) {
    const w = parseFloat(m[1]);
    let inner = m[2];
    let guard = 0, im;
    while ((im = inner.match(NAI_WEIGHT_RE)) && guard++ < 50) inner = im[2];
    inner = inner.replace(/^(?:-?\d+(?:\.\d+)?::)+/, "").replace(/(?:::)+$/, "").trim();
    return { tag: inner, weight: isNaN(w) ? 1.0 : w };
  }
  return { tag: t, weight: 1.0 };
}

// 저장 전용: 내부 상태는 표기 토글과 무관하게 항상 로컬 (tag:weight) 로 보관.
// 이래야 휠 재조절 시 parseTagWeight 가 안정적으로 역파싱한다(NAI 중첩 누적 방지).
export function localWeightToken(tag, weight) {
  const w = Math.round(weight * 10) / 10;
  if (Math.abs(w - 1.0) < 1e-9) return tag;
  return "(" + tag + ":" + w.toFixed(1) + ")";
}

export function formatTagWeight(tag, weight) {
  // 1.0 이면 tag 만(로컬·NAI 공통). 아니면 표기별 형식.
  const w = Math.round(weight * 10) / 10;
  if (Math.abs(w - 1.0) < 1e-9) return tag;
  const wStr = w.toFixed(1);
  if (state.weightSyntax === "nai") {
    // NAI 수치 강조: weight::tag:: (1.2::tag::, 음수 -1.0::tag:: 가능)
    return wStr + "::" + tag + "::";
  }
  // 로컬(SD 계열): (tag:weight)
  return "(" + tag + ":" + wStr + ")";
}

// 현재 표기의 가중치 하한(휠/입력 클램프용). NAI 는 음수 정식 지원.
export function weightFloor() {
  return (state.weightSyntax === "nai") ? -10 : 0.1;
}

// 토큰 가중치를 delta 만큼 옮긴 새 토큰(로컬 표기). 휠 조절 공용.
export function bumpToken(token, delta) {
  const { tag, weight } = parseTagWeight(token);
  let w = Math.round((weight + delta) * 10) / 10;
  if (w < weightFloor()) w = weightFloor();
  return localWeightToken(tag, w);
}

// 출력 형식 옵션(체크박스 2개)
export function readFormatOptions() {
  return {
    replaceUnderscores: $("replace_underscores").checked,
    escapeParentheses: $("escape_parentheses").checked,
  };
}

// 태그 한 개에 _→공백 / 괄호 이스케이프 적용
export function formatTagText(text, { replaceUnderscores, escapeParentheses }) {
  let t = text;
  if (replaceUnderscores) t = t.replace(/_/g, " ");
  if (escapeParentheses) t = t.replace(/\(/g, "\\(").replace(/\)/g, "\\)");
  return t;
}

// final_prompt 문자열에 치환/이스케이프를 적용하되 가중치 표기는 보존한다.
// 각 토큰의 코어 태그에만 _→공백 / 괄호 이스케이프를 걸고, (core:weight) 형태는
// 가중치 괄호를 깨지 않도록 다시 감싼다.
export function applyTagFormatting(dbTagString, opts) {
  const tokens = (dbTagString || "").split(", ").map(t => t.trim()).filter(Boolean);
  return tokens.map((tok) => {
    const { tag, weight } = parseTagWeight(tok);
    return formatTagWeight(formatTagText(tag, opts), weight);
  }).join(", ");
}

// ", " 로 이어진 토큰 문자열 → 빈 토큰 제외 배열
export function splitTokens(s) {
  return (s || "").split(", ").filter(t => t.trim() !== "");
}
