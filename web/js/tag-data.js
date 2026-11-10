// 태그 데이터(/danbooru-tags.csv): 자동완성 인덱스 + 결과 뱃지 호버 설명(tagInfo).
// anima-webgpu 자동완성 포팅: Web Worker CSV 파싱 + IndexedDB 캐시 + bigram 역인덱스.
// 프롬프트는 서버로 안 나감(검색 전부 로컬).
//   영어: prefix 매칭. 한글: 2글자↑ bigram 역인덱스 → desc substring. post_count 내림차순.
// CSV 한 번만 파싱해 자동완성용(ac.tags)과 결과 뱃지용(tagInfo) 둘 다 채운다.

const AC_CSV_URL = "/danbooru-tags.csv";
const AC_DB_NAME = "dtr-tagac";
const AC_DB_VER = 1;
const AC_STORE = "index";

// 결과 뱃지용: tag → { desc, category }
export const tagInfo = new Map();
export const ac = { ready: false, loading: false, tags: null, bigram: null };

// 워커로 새로 빌드했을 때 호출(이미 떠 있는 결과의 뱃지 색 갱신용). main.js 가 정한다.
let onBuilt = null;
export function setOnTagDataBuilt(fn) {
  onBuilt = fn;
}

// ── IndexedDB: 빌드 결과 저장/로드 (재방문 시 파싱 스킵) ──
function acIdbOpen() {
  return new Promise((resolve, reject) => {
    const rq = indexedDB.open(AC_DB_NAME, AC_DB_VER);
    rq.onupgradeneeded = () => {
      const db = rq.result;
      if (db.objectStoreNames.contains(AC_STORE)) db.deleteObjectStore(AC_STORE);
      db.createObjectStore(AC_STORE);
    };
    rq.onsuccess = () => resolve(rq.result);
    rq.onerror = () => reject(rq.error);
  });
}
async function acIdbGet(key) {
  const db = await acIdbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(AC_STORE, "readonly");
    const rq = tx.objectStore(AC_STORE).get(key);
    rq.onsuccess = () => resolve(rq.result);
    rq.onerror = () => reject(rq.error);
  });
}
async function acIdbSet(key, val) {
  const db = await acIdbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(AC_STORE, "readwrite");
    tx.objectStore(AC_STORE).put(val, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function fillTagInfo(infoKeys, infoVals) {
  tagInfo.clear();
  for (let i = 0; i < infoKeys.length; i++) tagInfo.set(infoKeys[i], infoVals[i]);
}
function rebuildBigram(keys, vals) {
  ac.bigram = new Map();
  for (let i = 0; i < keys.length; i++) ac.bigram.set(keys[i], vals[i]);
}

export async function acEnsureLoaded() {
  if (ac.ready || ac.loading) return;
  ac.loading = true;
  try {
    // 1) IndexedDB 캐시 우선
    const cached = await acIdbGet("data");
    if (cached && cached.tags && cached.bigramKeys) {
      ac.tags = cached.tags;
      rebuildBigram(cached.bigramKeys, cached.bigramVals);
      if (cached.infoKeys) fillTagInfo(cached.infoKeys, cached.infoVals);
      ac.ready = true; ac.loading = false;
      console.log("태그 자동완성 준비됨 (캐시 " + ac.tags.length + "개)");
      return;
    }
    // 2) Worker 로 CSV 파싱 + 인덱스 빌드
    console.log("태그 자동완성 데이터 빌드 중... (최초 1회)");
    const wk = new Worker(new URL("./tag-worker.js", import.meta.url), { type: "module" });
    const result = await new Promise((resolve, reject) => {
      wk.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === "done") resolve(m);
        else if (m.type === "error") reject(new Error(m.message));
      };
      wk.onerror = (e) => reject(new Error(e.message || "worker 오류"));
      wk.postMessage({ url: new URL(AC_CSV_URL, location.href).href });
    });
    wk.terminate();
    ac.tags = result.tags;
    rebuildBigram(result.bigramKeys, result.bigramVals);
    fillTagInfo(result.infoKeys, result.infoVals);
    ac.ready = true;
    console.log("태그 자동완성 준비됨 (" + ac.tags.length + "개)");
    // 3) IndexedDB 저장 (다음 방문 즉시)
    acIdbSet("data", {
      tags: result.tags,
      bigramKeys: result.bigramKeys, bigramVals: result.bigramVals,
      infoKeys: result.infoKeys, infoVals: result.infoVals,
    }).catch(() => {});
    // 결과가 이미 화면에 있으면 뱃지 카테고리 색 갱신
    if (onBuilt) {
      try { onBuilt(); } catch (e) {}
    }
  } catch (e) {
    console.warn("태그 자동완성 로드 실패:", e);
  } finally {
    ac.loading = false;
  }
}

// ── 검색 ──
const AC_LIMIT = 50;
const HANGUL_RE = /[가-힣]/;
export function acSearch(qRaw) {
  if (!ac.ready) return [];
  const q = qRaw.trim();
  if (!q) return [];
  const out = [];
  if (HANGUL_RE.test(q)) {
    if (q.length < 2) return [];   // 한글 1글자 스킵
    const cand = ac.bigram.get(q.slice(0, 2));
    if (!cand) return [];
    for (let i = 0; i < cand.length; i++) {
      const r = ac.tags[cand[i]];
      if (r.desc.includes(q)) { out.push(r); if (out.length >= AC_LIMIT) break; }
    }
  } else {
    // 영어 prefix(언더스코어/공백 양쪽 정규화)
    const ql = q.toLowerCase().replace(/_/g, " ");
    for (let i = 0; i < ac.tags.length; i++) {
      if (ac.tags[i].tagNorm.startsWith(ql)) {
        out.push(ac.tags[i]); if (out.length >= AC_LIMIT) break;
      }
    }
  }
  return out;
}
