"""약식 화면 HTML만 따로 저장소(Sparse_Hybrid_Attention)로 내보낸다 — 사내 GitHub로 옮기기 전 단계.

    python tools/export_repo.py <내보낼 폴더> [--commit]

- 내보내는 것: viewer/<model>-lite.html 6개 · DEPLOY-INTERNAL.md · README.md(생성) · .nojekyll
- 안 내보내는 것: 보고서 · 모델 yaml · 빌드 도구 · 템플릿 · viewer/artifact/(claude.ai 게시용)
- 뷰어 위 모델 이동 링크 → 같은 폴더 상대 경로 (사내 Pages · 파일로 열기 모두 사외망 없이 동작)
- 내보낼 폴더는 .git만 남기고 비운 뒤 다시 채운다. push는 하지 않는다
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WIKI = "https://github.com/KimHyeon-Ji/llm-arch-lineage-wiki"
DATA_LINE = re.compile(r"^const DATA = (\{.*\});\s*$", re.M)


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout.decode("utf-8")


def read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s.replace("\r\n", "\n"))


def clear(out):
    """내보낼 폴더를 .git만 남기고 비운다. git 저장소가 아닌 폴더는 건드리지 않는다."""
    if os.path.exists(out) and os.listdir(out) and not os.path.isdir(os.path.join(out, ".git")):
        sys.exit(f"{out}: 비어 있지 않은데 git 저장소가 아님 — 잘못 지정한 폴더일 수 있어 멈춤")
    os.makedirs(out, exist_ok=True)
    if not os.path.isdir(os.path.join(out, ".git")):
        git("init", "-b", "main", cwd=out)
    for name in os.listdir(out):
        if name == ".git":
            continue
        p = os.path.join(out, name)
        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)


def readme(nav, sha, dirty):
    rows = "\n".join(f"| {n['label']} | [viewer/{n['file']}-lite.html](viewer/{n['file']}-lite.html) |" for n in nav)
    src = f"`{sha}`" + (" + 커밋 안 된 변경" if dirty else "")
    return f"""# Sparse · Hybrid Attention 데이터 이동 — 약식 화면

모델별 attention 층 1개 · decode 1스텝에서 데이터가 HBM · 칩 안 메모리 · 연산기 사이를 어떻게 오가는지 커널 단위로 보는 HTML.

- 파일 하나에 다 들어 있음 — 서버 없이 브라우저로 바로 열림
- 화면 위 모델 이동 칩 = 같은 폴더 상대 경로 → GitHub Pages · 파일로 열기 모두 사외망 없이 동작 (글꼴만 Google Fonts)

| 모델 | 파일 |
|---|---|
{rows}

- 사내 GitHub로 옮겨 Pages 링크로 공유하는 법: [DEPLOY-INTERNAL.md](DEPLOY-INTERNAL.md)
- 원본: [llm-arch-lineage-wiki]({WIKI}) `arch_insight/dataflow-claude` (위키 커밋 {src})
- 이 저장소는 원본의 `tools/export_repo.py`가 매번 새로 채움 → 여기서 직접 고치지 않음
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="내보낼 폴더 (Sparse_Hybrid_Attention clone, 또는 새로 만들 빈 폴더)")
    ap.add_argument("--commit", action="store_true", help="복사 뒤 내보낸 폴더에서 git commit (push는 안 함)")
    a = ap.parse_args()
    if not sys.stdout.isatty():          # 파이프로 받을 때 Windows 기본(cp949) 대신 UTF-8
        sys.stdout.reconfigure(encoding="utf-8")

    out = os.path.abspath(a.out)
    top = os.path.normpath(git("rev-parse", "--show-toplevel", cwd=SRC).strip())
    nc = os.path.normcase
    if nc(out) == nc(top) or nc(out).startswith(nc(top + os.sep)):
        sys.exit(f"{out}: 위키 저장소 안쪽 — 위키 밖 폴더를 지정해야 함")

    sha = git("rev-parse", "--short", "HEAD", cwd=SRC).strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", "viewer", "DEPLOY-INTERNAL.md", cwd=SRC).strip())

    # 모델 목록 = 뷰어에 들어 있는 모델 이동 줄 (buildlib.nav_list와 같은 순서 · 이름)
    first = read(os.path.join(SRC, "viewer", "mla-v31-lite.html"))
    nav = json.loads(DATA_LINE.search(first).group(1))["nav"]

    clear(out)
    os.makedirs(os.path.join(out, "viewer"))
    for n in nav:
        rel = f"viewer/{n['file']}-lite.html"
        html = read(os.path.join(SRC, rel))
        m = DATA_LINE.search(html)
        if not m:
            sys.exit(f"{rel}: const DATA 줄을 못 찾음")
        data = json.loads(m.group(1))
        data["pages"] = ""          # 모델 이동 링크 = 같은 폴더 상대 경로
        js = json.dumps(data, ensure_ascii=False)
        html = html[:m.start(1)] + js + html[m.end(1):]
        if "github.io" in html:
            sys.exit(f"{rel}: 바꾼 뒤에도 github.io 주소가 남음 — 확인 필요")
        write(os.path.join(out, rel), html)
    write(os.path.join(out, "DEPLOY-INTERNAL.md"), read(os.path.join(SRC, "DEPLOY-INTERNAL.md")))
    write(os.path.join(out, "README.md"), readme(nav, sha, dirty))
    write(os.path.join(out, ".nojekyll"), "")

    print(f"{out}: 뷰어 {len(nav)}개 (모델 이동 링크 상대 경로로) · DEPLOY-INTERNAL.md · README.md · .nojekyll")
    if dirty:
        print("주의: 위키 viewer/ · DEPLOY-INTERNAL.md에 커밋 안 된 변경이 있음 — 지금 작업 트리 내용이 복사됨")

    if a.commit:
        git("add", "-A", cwd=out)
        if not git("status", "--porcelain", cwd=out).strip():
            print("변경 없음 — 커밋 안 함")
            return
        msg = f"sync: llm-arch-lineage-wiki {sha}{'+dirty' if dirty else ''} · dataflow-claude 약식 화면"
        git("commit", "-m", msg, cwd=out)
        print(f"커밋: {msg}\n다음: cd \"{out}\" ; git push origin main")
    else:
        print(f"다음: cd \"{out}\" ; git add -A ; git commit -m \"sync {sha}\" ; git push origin main")


if __name__ == "__main__":
    main()
