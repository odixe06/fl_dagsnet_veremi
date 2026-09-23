"""Hình cho docs/report.md, dựng từ artifact đã kéo về và verify (không lấy số từ W&B).

    python scripts/report_figures.py --out papers/lwfednids-bouayad-2024/report_data \
        20c=<run_dir> 50c=<run_dir> 100c=<run_dir>

run_dir = papers/lwfednids-bouayad-2024/runs/pulls/<K>c/runs/lwfednids_<K>c
"""
import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Cùng bảng màu và style với perfedskd/scripts/report_data.py để hai báo cáo đọc như một hệ.
SERIES = {"20c": "#2a78d6", "50c": "#eb6834", "100c": "#1baf7a"}
BLUES = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED = "#1a1a19", "#5a5a58"


def load(d):
    d = Path(d)
    hist = list(csv.DictReader(open(d / "history.csv")))
    last = int(hist[-1]["round"])
    return dict(hist=hist, final=json.load(open(d / f"metrics/round_{last:03d}.json")),
                cm=np.load(d / f"confusion/round_{last:03d}.npy"), last=last)


def col(hist, k):
    return np.array([float(h[k]) for h in hist])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = {t: load(d) for t, d in (x.split("=", 1) for x in a.runs)}
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e5e5e0", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#c3c2b7", "figure.dpi": 150, "axes.axisbelow": True})

    # 1. Hội tụ: f1_macro, accuracy theo round + lịch lr
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, k in zip(axes[:2], ("f1_macro", "accuracy")):
        ends = []
        for t, R in runs.items():
            r, y = col(R["hist"], "round"), col(R["hist"], k)
            ax.plot(r, y, color=SERIES[t], lw=2, label=t)
            ends.append([y[-1], y[-1], r[-1]])
        ends.sort()                      # giãn nhãn cuối đường để không đè nhau
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.022)
        for y, yl, x in ends:
            ax.text(x + 0.8, yl, f"{y:.3f}", color=INK, fontsize=7, va="center")
    lr = col(next(iter(runs.values()))["hist"], "lr")
    axes[2].plot(col(next(iter(runs.values()))["hist"], "round"), lr, color=INK, lw=2)
    axes[2].set_yscale("log")
    for ax, ttl in zip(axes, (r"f1_macro (mô hình toàn cục $\theta^t$)", r"accuracy ($\theta^t$)", "lr (cosine, chung cả ba)")):
        ax.set_title(ttl); ax.set_xlabel("round")
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.out / "convergence.png"); plt.close(fig)

    # 2. Loss / gnorm / skip của client theo round
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for t, R in runs.items():
        h = R["hist"]; r = col(h, "round")
        mu, sd = col(h, "loss_client_mean"), col(h, "loss_client_std")
        axes[0].fill_between(r, mu - sd, mu + sd, color=SERIES[t], alpha=0.12, lw=0)
        axes[0].plot(r, mu, color=SERIES[t], lw=2, label=t)
        axes[1].plot(r, col(h, "gnorm_client_mean"), color=SERIES[t], lw=2, label=t)
        axes[2].plot(r, col(h, "skipped"), color=SERIES[t], lw=2, label=t)
    for ax, ttl in zip(axes, ("loss client (mean ± std)", "grad-norm client (mean)", "bước AMP bị skip / round")):
        ax.set_title(ttl); ax.set_xlabel("round")
    axes[0].set_ylim(bottom=0); axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.out / "client_loss.png"); plt.close(fig)

    # 3. F1 theo lớp ở round cuối, sắp theo tỉ lệ test
    pcs = {t: R["final"]["per_class"] for t, R in runs.items()}
    ref = next(iter(pcs.values()))
    order = np.argsort([-c["support"] for c in ref])
    tot = sum(c["support"] for c in ref)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(runs)
    for i, (t, pc) in enumerate(pcs.items()):
        ax.bar(np.arange(16) + (i - (len(runs) - 1) / 2) * w, [pc[k]["f1"] for k in order],
               width=w * 0.92, color=SERIES[t], label=t)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"{ref[k]['class']} ({ref[k]['support'] / tot * 100:.1f} %)" for k in order],
                       rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("F1 (round 50)"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(r"F1 theo lớp của $\theta^{50}$, lớp sắp theo tỉ lệ trong tập test", loc="right")
    fig.tight_layout(); fig.savefig(a.out / "per_class_f1.png"); plt.close(fig)

    # 4. Confusion chuẩn hoá theo hàng (recall) ở round cuối
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    names = [c["class"] for c in ref]
    for t, R in runs.items():
        cm = R["cm"].astype(float); rn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        im = ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(names, rotation=70, ha="right", fontsize=6.5); ax.set_yticklabels(names, fontsize=6.5)
        ax.grid(False)
        for i in range(16):
            for j in range(16):
                if rn[i, j] >= 0.05:
                    ax.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center", fontsize=5,
                            color="white" if rn[i, j] > 0.55 else INK)
        ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật")
        ax.set_title(f"{t}: confusion của $\\theta^{{{R['last']}}}$ chuẩn hoá theo hàng")
        fig.colorbar(im, ax=ax, fraction=0.04, label="tỉ lệ theo hàng (recall)")
        fig.tight_layout(); fig.savefig(a.out / f"confusion_{t}.png"); plt.close(fig)

    # 5. Thời gian train / eval mỗi round
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))   # hai thang khác nhau cả chục lần: không chung trục y
    for t, R in runs.items():
        r = col(R["hist"], "round")
        axes[0].plot(r, col(R["hist"], "train_sec") / 60, color=SERIES[t], lw=2, label=t)
        axes[1].plot(r, col(R["hist"], "eval_sec"), color=SERIES[t], lw=2, label=t)
    for ax, ttl in zip(axes, ("train (phút / round, 2×T4)", r"eval $\theta^t$ trên 10,76 M dòng (giây / round)")):
        ax.set_title(ttl); ax.set_xlabel("round"); ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(a.out / "round_time.png"); plt.close(fig)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
