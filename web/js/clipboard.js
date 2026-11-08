// 복사 공통 헬퍼: HTTPS/localhost 가 아니면 navigator.clipboard 가 없으므로
// execCommand 폴백을 쓴다. 성공 시 onOk 콜백 실행.
export function doCopy(text, onOk) {
  if (!text) return;
  const fallback = () => {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      if (ok && onOk) onOk();
      else if (!ok) alert("복사 실패: 직접 선택해 복사해주세요.");
    } catch (e) {
      alert("복사 실패: 직접 선택해 복사해주세요.");
    }
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => onOk && onOk()).catch(fallback);
  } else {
    fallback();
  }
}
