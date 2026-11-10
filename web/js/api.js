// 서버 API 호출. 요청 본문·응답 해석은 예전 인라인 스크립트와 같다.
//   POST /api/generate, /api/generate_split → NDJSON 스트림 (줄마다 {status?, data?, ...})
//   POST /api/direct_search → JSON { results: [{tag, score}, ...] }

async function errorFrom(response) {
  const errData = await response.json().catch(() => ({}));
  return new Error(errData.detail || `서버 에러가 발생했습니다: ${response.status}`);
}

// NDJSON 스트림을 읽어 줄마다 onEvent(chunk) 를 부른다.
// onEvent 가 false 를 돌려주면 스트림을 끊고 멈춘다(중복요청 거절 등).
// 마지막 줄바꿈 뒤에 남은 조각은 버린다(서버는 줄마다 \n 으로 끝낸다).
export async function streamNdjson(url, payload, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw await errorFrom(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (!line.trim()) continue;
      const chunk = JSON.parse(line);
      if (onEvent(chunk) === false) {
        await reader.cancel();
        return;
      }
    }
  }
}

export async function directSearch({ query, threshold, categories }) {
  const response = await fetch("/api/direct_search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: query,
      threshold: threshold,
      top_k: 20,
      categories: categories.length ? categories : null,
    }),
  });
  if (!response.ok) throw await errorFrom(response);
  return response.json();
}
