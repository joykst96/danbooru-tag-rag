// 태그 CSV 파싱 워커: CSV → {tag, pc, cat, alias, tagNorm, desc} + 한글 bigram 역인덱스 + 결과 뱃지용 tagInfo.
// tag-data.js 가 { url } 을 보내면 { type: "done", ... } 를 돌려준다.

// name,category,post_count,"[대분류 > 소분류] 정의문. 키워드: 별칭1, 별칭2..."
function parseLine(line) {
  const m = line.match(/^([^,]*),([^,]*),([^,]*),([\s\S]*)$/);
  if (!m) return null;
  let desc = m[4].trim();
  if (desc.startsWith('"') && desc.endsWith('"')) desc = desc.slice(1, -1);
  const pc = parseInt(m[3], 10);
  const cat2 = parseInt(m[2], 10);   // 2열 숫자 카테고리(결과 뱃지 색용)
  if (!m[1] || isNaN(pc)) return null;
  return { tag: m[1], pc, cat2: isNaN(cat2) ? 0 : cat2, desc };
}
// 한글 alias(표시용): "키워드:" 뒤 콤마 분리. (최신 CSV는 꺾쇠<>가 없음)
function extractAliases(desc) {
  const i = desc.indexOf('키워드:');
  if (i < 0) return [];
  return desc.slice(i + 4).split(',').map(s => s.trim()).filter(Boolean);
}
// 카테고리(표시용): 맨 앞 [대분류] 또는 [대분류 > 소분류].
function extractCategory(desc) {
  const mb = desc.match(/^\s*\[([^\]]*)\]/);
  return mb ? mb[1].trim() : null;
}
self.onmessage = async (ev) => {
  try {
    const { url } = ev.data;
    const res = await fetch(url);
    if (!res.ok) throw new Error('CSV HTTP ' + res.status);
    const text = await res.text();
    self.postMessage({ type: 'progress', stage: '파싱' });
    const lines = text.split('\n');
    const tags = [];        // 자동완성용 {tag, pc, desc, alias, cat, tagNorm}
    const infoKeys = [];    // 결과 뱃지용 tagInfo: tag → {desc, category}
    const infoVals = [];
    for (let i = 0; i < lines.length; i++) {
      const ln = lines[i];
      if (!ln.trim()) continue;
      if (i === 0 && /^name\s*,/.test(ln)) continue;   // 헤더 스킵
      const r = parseLine(ln);
      if (!r) continue;
      tags.push({
        tag: r.tag, pc: r.pc, desc: r.desc,
        alias: extractAliases(r.desc).join(', '),
        cat: extractCategory(r.desc),
        tagNorm: r.tag.toLowerCase().replace(/_/g, ' '),
      });
      infoKeys.push(r.tag);
      infoVals.push({ desc: r.desc, category: r.cat2 });
    }
    tags.sort((a, b) => b.pc - a.pc);   // post_count 내림차순
    self.postMessage({ type: 'progress', stage: '인덱스' });
    // 한글 bigram 역인덱스: desc 연속 2글자(공백 제외) → 행 인덱스
    const bi = new Map();
    for (let i = 0; i < tags.length; i++) {
      const s = tags[i].desc;
      const seen = new Set();
      for (let p = 0; p < s.length - 1; p++) {
        const a = s[p], b = s[p + 1];
        if (a === ' ' || b === ' ') continue;
        const bg = a + b;
        if (seen.has(bg)) continue;
        seen.add(bg);
        let arr = bi.get(bg);
        if (!arr) { arr = []; bi.set(bg, arr); }
        arr.push(i);
      }
    }
    const bigramKeys = [], bigramVals = [];
    for (const [k, v] of bi) { bigramKeys.push(k); bigramVals.push(v); }
    self.postMessage({ type: 'done', tags, bigramKeys, bigramVals, infoKeys, infoVals });
  } catch (e) {
    self.postMessage({ type: 'error', message: e.message });
  }
};
