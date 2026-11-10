// 글로벌 툴팁: data-tooltip 속성이 있는 요소에 마우스를 올리면 설명을 띄운다.
export function initTooltip() {
  const globalTooltip = document.createElement("div");
  globalTooltip.className = "custom-tooltip";
  document.body.appendChild(globalTooltip);

  document.addEventListener("mouseover", (e) => {
    const tag = e.target.closest("[data-tooltip]");
    if (tag) {
      globalTooltip.textContent = tag.getAttribute("data-tooltip");
      globalTooltip.style.display = "block";

      const rect = tag.getBoundingClientRect();
      let topPos = rect.top - 8;
      let transform = "translate(-50%, -100%)";

      // 화면 위쪽으로 잘릴 것 같으면 툴팁을 아래로 표시
      if (topPos < 50) {
        topPos = rect.bottom + 8;
        transform = "translate(-50%, 0)";
      }

      const leftPos = rect.left + (rect.width / 2);

      globalTooltip.style.left = leftPos + "px";
      globalTooltip.style.top = topPos + "px";
      globalTooltip.style.transform = transform;
    }
  });

  document.addEventListener("mouseout", (e) => {
    const tag = e.target.closest("[data-tooltip]");
    if (tag) {
      globalTooltip.style.display = "none";
    }
  });

  // 스크롤 시 툴팁 숨기기 (위치 어긋남 방지)
  document.addEventListener("scroll", () => {
    globalTooltip.style.display = "none";
  }, true);
}
