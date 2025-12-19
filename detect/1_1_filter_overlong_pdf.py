#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, shutil, sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

ROOT_DIR = "/mnt/parallel_ssd/home/zdhs0006/mlrbench/download/downloads/ICML_2025_oral"
MAX_PAGES = 30
DEST_DIR = "/mnt/parallel_ssd/home/zdhs0006/mlrbench/download/downloads/ICML_30"


# --- 优先用 pypdf；没有则退回 PyPDF2 ---
def _import_pdf_reader():
    try:
        from pypdf import PdfReader  # type: ignore
        return PdfReader, "pypdf"
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            return PdfReader, "PyPDF2"
        except Exception:
            print("请先安装 pypdf 或 PyPDF2：pip install pypdf 或 pip install PyPDF2", file=sys.stderr)
            sys.exit(1)


PdfReader, _PDF_LIB = _import_pdf_reader()


def count_pdf_pages(pdf_path: Path) -> int:
    """返回 PDF 页数；若无法读取则返回 -1。"""
    try:
        with pdf_path.open("rb") as f:
            reader = PdfReader(f)
            # pypdf / PyPDF2 都兼容 len(reader.pages)
            return len(reader.pages)
    except Exception:
        return -1


def find_papers(root: Path) -> List[Tuple[Path, Path]]:
    """
    在 root 的第一层子目录中查找 paper.pdf
    返回 [(paper_dir, pdf_path), ...]
    """
    out = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        pdf_path = sub / "paper.pdf"
        if pdf_path.is_file():
            out.append((sub, pdf_path))
    return out


def make_bins_by5(max_pages: int) -> List[Tuple[int, int]]:
    """
    生成 [ (1,5), (6,10), ... ] 直到覆盖到 max_pages（向上取整到5的倍数）
    """
    top = int(math.ceil(max(1, max_pages) / 5.0) * 5)
    bins = []
    start = 1
    while start <= top:
        end = start + 4
        bins.append((start, end))
        start = end + 1
    return bins


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def build_histogram(page_counts: List[int]) -> Dict[str, int]:
    """按 5 页一个区间统计直方图（忽略 <=0 的计数）。"""
    valid = [p for p in page_counts if p > 0]
    if not valid:
        return {}
    bins = make_bins_by5(max(valid))
    hist = {bin_label(lo, hi): 0 for (lo, hi) in bins}
    for p in valid:
        # 找到对应区间
        idx = (int(math.ceil(p / 5.0)) - 1)
        lo, hi = bins[idx]
        hist[bin_label(lo, hi)] += 1
    return hist


def safe_copy_dir(src: Path, dst_dir: Path) -> Path:
    """
    将 src 目录复制到 dst_dir 中；若重名则追加序号避免覆盖。
    返回目标路径。原目录不会被修改。
    """
    base = src.name
    target = dst_dir / base
    if not target.exists():
        shutil.copytree(str(src), str(target))
        return target
    # 处理重名：base__1, base__2, ...
    i = 1
    while True:
        cand = dst_dir / f"{base}__{i}"
        if not cand.exists():
            shutil.copytree(str(src), str(cand))
            return cand
        i += 1


def save_reports(root: Path, rows: List[Dict], hist: Dict[str, int]) -> None:
    # CSV
    csv_path = root / "pdf_page_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "pdf_path", "pages"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "folder": r["folder"],
                "pdf_path": r["pdf_path"],
                "pages": r["pages"],
            })
    # JSON
    json_path = root / "pdf_page_report.json"
    report = {
        "summary": {
            "total_papers": len(rows),
            "readable_papers": sum(1 for r in rows if r["pages"] > 0),
            "unreadable_papers": sum(1 for r in rows if r["pages"] <= 0),
        },
        "histogram_by5": hist,
        "items": rows,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="统计并按页数过滤 PDF（每个子文件夹里有 paper.pdf）")
    ap.add_argument("--root", default=ROOT_DIR,
                    help="根目录路径（包含若干论文子文件夹，每个夹内有 paper.pdf）")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES,
                    help="阈值（例如 25）。若提供，则把小于阈值(<N)的论文子文件夹复制到新目录")
    ap.add_argument("--dest", default=DEST_DIR,
                    help="过滤结果的目标目录（默认：<root>__LT_<N>）")
    ap.add_argument("--dry-run", action="store_true", help="只展示将要复制的子文件夹，不实际复制")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[错误] root 不存在或不是目录：{root}", file=sys.stderr)
        sys.exit(2)

    # 1) 收集 paper.pdf
    pairs = find_papers(root)
    if not pairs:
        print(f"[提示] 未在 {root} 下找到任何包含 paper.pdf 的子文件夹。")
        sys.exit(0)

    # 2) 统计页数
    rows = []
    page_values = []
    unreadable = 0
    for folder, pdf in pairs:
        pages = count_pdf_pages(pdf)
        rows.append({"folder": str(folder.name), "pdf_path": str(pdf), "pages": pages})
        if pages > 0:
            page_values.append(pages)
        else:
            unreadable += 1

    # 3) 直方图（每 5 页）
    hist = build_histogram(page_values)
    save_reports(root, rows, hist)

    # 4) 终端输出统计
    total = len(rows)
    ok = len(page_values)
    print(f"\n====== 统计结果（库：{_PDF_LIB}）======")
    print(f"总论文数：{total}")
    print(f"可读取：{ok} ；无法读取：{unreadable}")
    if ok > 0:
        print(f"最小页数：{min(page_values)}  最大页数：{max(page_values)}")
        print(f"平均值：{mean(page_values):.2f}  中位数：{median(page_values):.2f}")
    print("\n【每 5 页区间直方图】")
    if hist:
        # 保证按区间序排列输出
        def _key(k):
            lo = int(k.split("-")[0])
            return lo
        for k in sorted(hist.keys(), key=_key):
            print(f"{k:>7}: {hist[k]}")
    else:
        print("(无可用页数)")

    # 5) 若提供了阈值，则进行筛选并复制
    if args.max_pages is not None:
        threshold = int(args.max_pages)
        # 只复制 0 < pages < threshold 的论文
        to_copy = [r for r in rows if 0 < r["pages"] < threshold]
        keep = [r for r in rows if r["pages"] >= threshold]

        print(f"\n阈值：{threshold} 页")
        print(f"需复制（<{threshold} 页）：{len(to_copy)} 篇")
        print(f"保留（≥{threshold} 页，或者无法读取）：{len(keep) + unreadable}")

        if to_copy:
            dest = Path(args.dest).expanduser().resolve() if args.dest else (
                root.parent / f"{root.name}__LT_{threshold}"
            )
            print(f"目标目录：{dest}")
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                # 执行复制（原目录不动）
                for r in to_copy:
                    src = root / r["folder"]
                    if src.exists() and src.is_dir():
                        safe_copy_dir(src, dest)
                print("\n✅ 已复制完成。")
            else:
                # 仅展示将要复制的列表
                print("\n[Dry-Run] 将要复制以下子文件夹：")
                for r in to_copy:
                    print(" -", r["folder"])

        # 提醒导出文件
        print(f"\n报告文件：")
        print(f" - {root / 'pdf_page_report.csv'}")
        print(f" - {root / 'pdf_page_report.json'}")
    else:
        print("\n未指定 --max-pages，仅完成统计。")
        print(f"报告文件：")
        print(f" - {root / 'pdf_page_report.csv'}")
        print(f" - {root / 'pdf_page_report.json'}")


if __name__ == "__main__":
    main()
