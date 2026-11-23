# 사내 GitHub 배포 — 약식 화면 HTML을 사내 링크로 공유

- 목적: 사외 GitHub 연결이 안 되는 사내 회의실에서도 약식 화면(`viewer/*-lite.html`)을 사내 링크로 열기
- 방법: 약식 화면 HTML만 공개 저장소 `Sparse_Hybrid_Attention`으로 내보냄 → 사내 PC에서 받음 → 사내 GitHub에 push → 사내 GitHub Pages 링크로 공유
- 내보내는 것: `viewer/<model>-lite.html` 6개 · 이 문서 · `README.md` · `.nojekyll`
- 안 내보내는 것: 보고서 · 모델 yaml · 빌드 도구 · 템플릿 · claude.ai artifact용 본문 (사내에서 안 씀)

## 흐름

```
[개인 PC]  위키 llm-arch-lineage-wiki/arch_insight/dataflow-claude   ← 원본. 고치는 곳은 여기뿐
    │  python tools/export_repo.py ../../../Sparse_Hybrid_Attention --commit
    ▼
[github.com]  KimHyeon-Ji/Sparse_Hybrid_Attention   (공개 · HTML만)
    │  사내 PC에서 git clone (A)  또는  ZIP 반입 (B)
    ▼
[사내 PC]  ── git push ──▶  [사내 GitHub] <org>/Sparse_Hybrid_Attention  ── Pages ──▶  사내 링크
```

## 내보낼 때 바뀌는 것

| 항목 | 위키 (원본) | 내보낸 저장소 |
|---|---|---|
| 뷰어 위 모델 이동 칩 링크 | 공개 Pages 주소 (`kimhyeon-ji.github.io/...`) | 같은 폴더 상대 경로 — 사내 Pages · 파일로 열기 모두 사외망 없이 동작 |
| 그 밖의 뷰어 내용 | — | 그대로 |
| `README.md` | 위키 폴더 설명 | 모델 6개 파일 표 + 원본 위키 커밋 번호 (새로 만듦) |

---

## 1. 개인 PC — 내보내기

### 처음 한 번 (완료)

- `https://github.com/KimHyeon-Ji/Sparse_Hybrid_Attention` — 공개 저장소
  - 공개 이유: 사내 PC에서 개인 계정 로그인 없이 clone · ZIP 받기. 내용은 이미 공개 위키에 있는 것과 같음
- 위키 옆 폴더에 clone: `C:/Users/99ktx/OneDrive/바탕 화면/Sparse_Hybrid_Attention`

### 뷰어가 바뀔 때마다

위키에서 뷰어를 다시 빌드하고 위키 main에 커밋한 뒤:

```bash
cd "C:/Users/99ktx/OneDrive/바탕 화면/llm-arch-lineage-wiki/arch_insight/dataflow-claude"
python tools/export_repo.py ../../../Sparse_Hybrid_Attention --commit   # 복사 + 링크 변환 + 커밋 (push 안 함)
cd ../../../Sparse_Hybrid_Attention
git push origin main
```

- 내보낸 폴더는 `.git`만 남기고 매번 새로 채움 → 그 폴더에서 직접 고친 내용은 사라짐
- 위키에 커밋 안 된 변경이 있으면 경고 + 커밋 메시지에 `+dirty`

---

## 2. 사내 PC — 처음 한 번

### 2-1. 받기 (A 또는 B)

**A. 사내 PC에서 github.com 접속 가능**

```bash
git clone https://github.com/KimHyeon-Ji/Sparse_Hybrid_Attention.git
cd Sparse_Hybrid_Attention
git remote rename origin github      # 공개 저장소 = github, 사내 저장소 = origin
```

**B. 접속 불가 — ZIP 반입**

1. 개인 PC에서 `https://github.com/KimHyeon-Ji/Sparse_Hybrid_Attention` → **Code → Download ZIP**
2. 사내 반입 절차대로 옮기고 압축 풀기 (폴더 이름 `Sparse_Hybrid_Attention-main`)
3. git 저장소로 만들기

   ```bash
   cd Sparse_Hybrid_Attention-main
   git init -b main
   git add -A
   git commit -m "import Sparse_Hybrid_Attention"
   ```

### 2-2. 사내 GitHub에 빈 저장소 만들기

- 이름 `Sparse_Hybrid_Attention` (다른 이름도 됨 — 링크의 저장소 이름만 바뀜)
- README · .gitignore 체크 해제
- 공개 범위(Internal · Private)는 사내 규정대로

### 2-3. push

```bash
git remote add origin https://<사내호스트>/<org>/Sparse_Hybrid_Attention.git
git push -u origin main
```

### 2-4. Pages 켜기

- 사내 저장소 **Settings → Pages → Source: Deploy from a branch → Branch `main` / `(root)` → Save**
- 1~2분 뒤 같은 화면 위에 사이트 주소가 뜸. 사내 GitHub(Enterprise Server)는 보통 둘 중 하나
  - `https://pages.<사내호스트>/<org>/Sparse_Hybrid_Attention/`
  - `https://<사내호스트>/pages/<org>/Sparse_Hybrid_Attention/`
- Settings에 Pages 메뉴가 없으면 → 사내 GitHub 관리자가 Pages를 꺼 둔 것. 관리자에게 요청

### 2-5. 공유할 링크

`<사이트 주소>` = 2-4에서 뜬 주소

| 모델 | 링크 |
|---|---|
| 기준선 · DeepSeek-V3.1 (MLA) | `<사이트 주소>/viewer/mla-v31-lite.html` |
| Sparse · DeepSeek-V3.2 (DSA) | `<사이트 주소>/viewer/dsa-v32-lite.html` |
| Sparse · DeepSeek-V4-Flash (CSA) | `<사이트 주소>/viewer/csa-v4flash-lite.html` |
| Hybrid · Kimi K3.1 Pro (KDA + DSA) | `<사이트 주소>/viewer/kda-k3-lite.html` |
| Hybrid · Qwen3.8-Flash-Next (GDN + QSA) | `<사이트 주소>/viewer/qsa-qwen38-lite.html` |
| Hybrid · Qwen3.6-27B (GDN 층) | `<사이트 주소>/viewer/gdn-qwen36-lite.html` |

- 화면 위 모델 이동 칩으로 나머지 모델에 이동 → 링크 하나만 공유해도 됨
- `<사이트 주소>/` 자체는 빈 페이지(404) — 위 파일 주소로 공유

---

## 3. 사내 PC — 업데이트

개인 PC에서 1의 "뷰어가 바뀔 때마다"를 마친 뒤.

**A. git으로 받은 경우**

```bash
git pull github main
git push origin main
```

**B. ZIP으로 받은 경우** — 새 ZIP을 풀고 기존 사내 저장소 폴더에 덮어쓰기 (PowerShell)

```powershell
robocopy "<새 ZIP 푼 폴더>" "<사내 저장소 폴더>" /MIR /XD .git
cd "<사내 저장소 폴더>"
git add -A
git commit -m "sync <위키 커밋>"     # 위키 커밋 번호 = README.md 맨 아래 "원본" 줄
git push origin main
```

- `/MIR` = 원본에서 지운 파일도 지움 · `/XD .git` = 사내 저장소의 `.git`은 유지
- robocopy 종료 코드 0 ~ 7은 정상 (8 이상이 오류)

push 후 1~2분 뒤 같은 사내 링크에 반영.

---

## 4. 사내 Claude에게 줄 프롬프트

사내 PC에서 받은 폴더를 연 상태로, `< >` 부분만 채워서 붙여 넣기.

### 처음 한 번

```text
이 폴더(Sparse_Hybrid_Attention)를 사내 GitHub에 올리고 GitHub Pages로 HTML 링크를 공유하려 한다.
DEPLOY-INTERNAL.md의 "2. 사내 PC — 처음 한 번"을 따라 진행해줘.

- 받은 방식: <A: git clone | B: ZIP>
- 사내 저장소 주소: https://<사내호스트>/<org>/Sparse_Hybrid_Attention.git  (웹에서 빈 저장소로 만들어 둠)

지켜줄 것:
- git 명령은 실행 전에 보여주고, push 전에 한 번 더 확인받기
- 파일 내용은 고치지 않기 (원본은 개인 PC 위키에 있음)
- Settings → Pages는 내가 웹에서 켜니 메뉴 경로만 알려주기

끝나면:
- viewer/*-lite.html 6개 안의 DATA.pages가 ""(상대 경로)이고 kimhyeon-ji.github.io 주소가 없는지 확인
- 내가 알려주는 Pages 사이트 주소로 6개 모델 링크 표 만들기 (DEPLOY-INTERNAL.md 2-5 형식)
```

### 업데이트

```text
DEPLOY-INTERNAL.md의 "3. 사내 PC — 업데이트"를 따라 이 저장소를 최신으로 맞추고 사내 GitHub에 push해줘.

- 받은 방식: <A: git pull | B: 새 ZIP 푼 폴더 = <경로>>

push 전에 바뀐 파일 요약(git status · README.md의 위키 커밋 번호)을 보여주고 확인받아줘. 파일 내용은 고치지 마.
```

---

## 5. 확인 · 안 될 때

| 증상 | 원인 · 조치 |
|---|---|
| 모델 이동 칩이 `kimhyeon-ji.github.io`로 감 | `export_repo.py`를 거치지 않은 파일 (위키 원본을 직접 복사) → 이 저장소 파일 쓰기 |
| GitHub 파일 화면 · raw 링크로 열면 코드만 보임 | 정상 — HTML은 Pages 주소로 열어야 화면이 뜸 |
| Pages 주소 404 | 배포 1~2분 대기 · Settings → Pages가 `main` / `(root)`인지 · 경로 대소문자 확인 · 주소 끝에 `/viewer/<model>-lite.html` 붙였는지 |
| Pages 주소가 로그인을 요구 | 사내 설정에 따라 사내 GitHub 로그인이 필요할 수 있음 → 회의실 PC 브라우저에서 사내 계정 로그인 |
| Settings에 Pages 없음 | 사내에서 Pages 꺼짐 → 관리자 요청. 그동안은 `viewer/` 폴더를 통째로 복사해 파일로 열기 (상대 경로라 모델 이동도 됨) |
| 글꼴이 다르게 보임 | 사내망에서 Google Fonts 차단 → 시스템 글꼴로 대체. 내용 · 그림은 그대로 |
