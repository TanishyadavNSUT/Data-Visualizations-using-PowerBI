#!/usr/bin/env python3
# ============================================================================
#  AeroGuard TinyML — Data Visualization & Analytics
#  Capstone Project 3  :  Utilizing the Power BI Library (Plotly)
#  Unit II             :  Automation and Data Visualization using Power BI
#
#  Generates the 17 visualizations present in the R_Visualizations folder
#  and extends them with PowerBI-native visuals using Plotly.
#  Output → C:\HS\TML1\PowerBI_Visualizations\
# ============================================================================

# ── 1. Auto-install required packages ────────────────────────────────────────
import subprocess, sys, os, warnings
warnings.filterwarnings("ignore")

_REQUIRED = [
    ("plotly",  "plotly"),
    ("kaleido", "kaleido"),
    ("pandas",  "pandas"),
    ("numpy",   "numpy"),
    ("scipy",   "scipy"),
]
for _pkg, _imp in _REQUIRED:
    try:
        __import__(_imp)
    except ImportError:
        print(f"  Installing {_pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "-q"])

# ── 2. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── 3. Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = r"C:\HS\TML1"
METADATA_FILE = os.path.join(BASE_DIR, "TinyML_Dataset", "metadata", "dataset_metadata.csv")
FEATURES_DIR  = os.path.join(BASE_DIR, "TinyML_Dataset", "features")
OUTPUT_DIR    = os.path.join(BASE_DIR, "PowerBI_Visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 4. Power BI "Soft Cloud" Colour Palette ───────────────────────────────────
CLASSES   = ["Background", "Cough", "Human_Noise"]
CLS_COLOR = {
    "Background":  "#7EC8E3",   # soft sky-blue
    "Cough":       "#FF6B6B",   # coral rose
    "Human_Noise": "#95E1A3",   # mint green
}
PALETTE = [
    "#7EC8E3","#FF6B6B","#95E1A3","#BDB2FF","#FFD6A5",
    "#CAFFBF","#A0C4FF","#FFC6FF","#FDFFB6","#FFB7B2",
    "#E2F0CB","#C7CEEA",
]

FONT = "Segoe UI, Arial, sans-serif"   # Power BI native font
BG   = "#FAFBFD"
GRID = "#ECF0F1"
TXT  = "#2C3E50"
SUB  = "#5D6D7E"

# ── Helper: hex → rgba string ─────────────────────────────────────────────────
def hex_rgba(hex_c: str, a: float = 0.35) -> str:
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

# ── Helper: standard layout dict ─────────────────────────────────────────────
def lay(title: str, sub: str = "", w: int = 1200, h: int = 700) -> dict:
    sub_html = (f"<br><span style='font-size:13px;color:{SUB}'>{sub}</span>"
                if sub else "")
    return dict(
        title=dict(
            text=f"<b>{title}</b>{sub_html}",
            x=0.5, xanchor="center",
            font=dict(family=FONT, size=20, color=TXT),
        ),
        font=dict(family=FONT, color=SUB, size=12),
        paper_bgcolor=BG, plot_bgcolor=BG,
        width=w, height=h,
        margin=dict(l=75, r=75, t=115, b=75),
        legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1,
                    font=dict(size=12)),
    )

# ── Helper: styled axis dict ──────────────────────────────────────────────────
def ax(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(family=FONT, size=12, color=TXT)),
        gridcolor=GRID, showgrid=True,
        linecolor=GRID, zeroline=False,
        tickfont=dict(family=FONT, size=10, color=SUB),
    )

# ── Helper: save figure as PNG ────────────────────────────────────────────────
def save(fig: go.Figure, name: str, w: int = 1200, h: int = 700) -> None:
    fig.update_layout(width=w, height=h)
    fig.write_image(os.path.join(OUTPUT_DIR, name), scale=2)
    print(f"  [SAVED] {name}")

# ── Helper: confidence ellipse ────────────────────────────────────────────────
def conf_ellipse(x, y, n_std=1.7, n_pts=120):
    """Return x, y arrays of an n_std-sigma confidence ellipse."""
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    theta = np.arctan2(vecs[1, 0], vecs[0, 0])
    a, b = n_std * np.sqrt(vals[1]), n_std * np.sqrt(vals[0])
    t = np.linspace(0, 2 * np.pi, n_pts)
    xe = a * np.cos(t)
    ye = b * np.sin(t)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return (cos_t * xe - sin_t * ye + np.mean(x),
            sin_t * xe + cos_t * ye + np.mean(y))

# ── 5. Load Metadata ──────────────────────────────────────────────────────────
print("=" * 62)
print("  AeroGuard TinyML — PowerBI Visualization & Analytics")
print("=" * 62 + "\n")

print("[1/3] Loading metadata …")
meta = pd.read_csv(METADATA_FILE)
print(f"  Samples : {len(meta)}  |  Classes : {', '.join(meta['class'].unique())}")

# ── 6. Load MFCC Features ─────────────────────────────────────────────────────
print("[2/3] Loading MFCC features …")
np.random.seed(42)
recs, mats = [], {}

for cls in CLASSES:
    cls_rows = meta[(meta["class"] == cls) & (meta["split"] == "train")]
    sampled  = cls_rows.sample(n=min(30, len(cls_rows)))
    mc = 0
    for _, row in sampled.iterrows():
        fp = os.path.join(FEATURES_DIR, row["mfcc_file"])
        if not os.path.exists(fp):
            continue
        try:
            m = np.load(fp).T   # (13, 101) → (101, 13)  [frames × coefficients]
            for ci in range(m.shape[1]):
                v = m[:, ci]
                recs.append(dict(
                    cls=cls, file=row["filename"],
                    coefficient=f"MFCC_{ci+1}", coeff_num=ci + 1,
                    mean_val=float(v.mean()),  sd_val=float(v.std()),
                    min_val=float(v.min()),    max_val=float(v.max()),
                    median_val=float(np.median(v)),
                    range_val=float(v.max() - v.min()),
                ))
            if mc < 3:
                mats[f"{cls}_{mc+1}"] = {"mat": m, "cls": cls,
                                          "file": row["filename"]}
                mc += 1
        except Exception:
            pass

mfcc = pd.DataFrame(recs)
print(f"  Files   : {mfcc['file'].nunique()}  |  Records : {len(mfcc)}\n")

# ── 7. Derived Datasets ───────────────────────────────────────────────────────
print("[3/3] Preparing derived datasets …")

cls_tot   = meta.groupby("class").size().reset_index(name="total")
cls_tot["pct"] = cls_tot["total"] / cls_tot["total"].sum() * 100

cls_split = meta.groupby(["class","split"]).size().reset_index(name="count")

src = (meta.groupby("class")
           .agg(unique_sources=("source_file","nunique"),
                total_samples=("filename","count"))
           .reset_index())
src["avg_windows"] = src["total_samples"] / src["unique_sources"]

split_totals = meta.groupby("split").size().reset_index(name="count")

model_sizes = pd.DataFrame({
    "stage": ["Keras (.keras)", "TFLite Float32", "TFLite Int8"],
    "size_kb": [51.64, 59.67, 28.01],
})

model_perf = pd.DataFrame({
    "metric": ["Train Accuracy","Validation Accuracy",
               "Test Accuracy","Best Val Accuracy"],
    "value":  [92.14, 91.88, 96.68, 96.68],
})

arch = pd.DataFrame({
    "layer":  ["Conv1D (32)","Conv1D (64)","Conv1D (128)",
               "Dense (128)","Output (3)"],
    "params": [1280, 6208, 24704, 16512, 387],
})
arch["pct"] = arch["params"] / arch["params"].sum() * 100

training_samples = 1082
test_samples = 271
test_accuracy = model_perf.loc[model_perf["metric"] == "Test Accuracy", "value"].iat[0]
test_correct = int(round(test_samples * test_accuracy / 100.0))

print()
print("=" * 62)
print("  Generating 23 Visualizations …")
print("=" * 62 + "\n")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 01 — Class Distribution Bar Chart
# ════════════════════════════════════════════════════════════════════════════
print("Plot  1 : Class Distribution Bar Chart")

fig = go.Figure([go.Bar(
    x=cls_tot["class"],
    y=cls_tot["total"],
    marker_color=[CLS_COLOR[c] for c in cls_tot["class"]],
    marker_line_color="white", marker_line_width=2,
    text=[f"<b>{v}</b><br>({p:.1f}%)"
          for v, p in zip(cls_tot["total"], cls_tot["pct"])],
    textposition="outside",
    textfont=dict(size=14, color=TXT, family=FONT),
)])
fig.update_layout(
    **lay("AeroGuard Dataset — Class Distribution",
          "Total audio samples per sound category"),
    showlegend=False, bargap=0.45,
    xaxis=ax("Audio Class"),
    yaxis=ax("Number of Samples"),
    yaxis_range=[0, cls_tot["total"].max() * 1.25],
)
save(fig, "01_class_distribution_bar.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 02 — Train vs Test Split Grouped Bar
# ════════════════════════════════════════════════════════════════════════════
print("Plot  2 : Train/Test Split Distribution")

SPLIT_COLORS = {"train": "#74B9FF", "test": "#FD79A8"}
fig = go.Figure()
for split in ["train", "test"]:
    d = cls_split[cls_split["split"] == split]
    fig.add_trace(go.Bar(
        x=d["class"], y=d["count"],
        name=f"{split.capitalize()} Set",
        marker_color=SPLIT_COLORS[split],
        marker_line_color="white", marker_line_width=2,
        text=d["count"], textposition="outside",
        textfont=dict(size=13, color=TXT),
    ))
fig.update_layout(
    **lay("Train vs Test Split by Class",
          "80/20 stratified split across all categories"),
    barmode="group", bargap=0.25, bargroupgap=0.07,
    xaxis=ax("Audio Class"),
    yaxis=ax("Number of Samples"),
    yaxis_range=[0, cls_split["count"].max() * 1.25],
)
save(fig, "02_train_test_split.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 06 — MFCC Coefficient Box Plots
# ════════════════════════════════════════════════════════════════════════════
print("Plot  6 : MFCC Coefficient Box Plots")

fig = go.Figure()
for cls in CLASSES:
    d = mfcc[mfcc["cls"] == cls]
    fig.add_trace(go.Box(
        x=d["coefficient"], y=d["mean_val"],
        name=cls,
        marker_color=CLS_COLOR[cls],
        line_color=SUB, line_width=1.2,
        fillcolor=hex_rgba(CLS_COLOR[cls], 0.75),
        opacity=0.9,
        boxmean="sd",
        width=0.25,
    ))
fig.update_layout(
    **lay("MFCC Coefficient Distributions by Class",
          "Mean values across audio samples — key acoustic discriminators",
          w=1400, h=720),
    boxmode="group",
    xaxis=ax("MFCC Coefficient"),
    yaxis=ax("Mean Value"),
    xaxis_tickangle=-40,
)
save(fig, "06_mfcc_boxplot.png", w=1400, h=720)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 07 — MFCC Violin Plots  (first 6 coefficients, 2×3 grid)
# ════════════════════════════════════════════════════════════════════════════
print("Plot  7 : MFCC Violin Plots")

first6 = [f"MFCC_{i}" for i in range(1, 7)]
pos    = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=first6,
    vertical_spacing=0.13, horizontal_spacing=0.09,
)
for idx, coeff in enumerate(first6):
    r, c = pos[idx]
    for cls in CLASSES:
        d = mfcc[(mfcc["cls"] == cls) & (mfcc["coefficient"] == coeff)]
        fig.add_trace(go.Violin(
            y=d["mean_val"],
            name=cls,
            fillcolor=hex_rgba(CLS_COLOR[cls], 0.65),
            line_color=CLS_COLOR[cls],
            box_visible=True,
            meanline_visible=True,
            points="all",
            pointpos=0,
            marker=dict(size=3, color=CLS_COLOR[cls], opacity=0.45),
            opacity=0.85,
            showlegend=(idx == 0),
        ), row=r, col=c)

fig.update_layout(
    **lay("MFCC Feature Distribution — Violin Plots",
          "Distribution shape + raw points for MFCC coefficients 1–6",
          w=1400, h=900),
    violinmode="group",
)
fig.update_annotations(font=dict(family=FONT, size=13, color=TXT))
for axis in fig.layout:
    if axis.startswith("xaxis") or axis.startswith("yaxis"):
        fig.layout[axis].update(
            gridcolor=GRID, linecolor=GRID, zeroline=False,
            tickfont=dict(family=FONT, size=9, color=SUB),
        )
save(fig, "07_mfcc_violin.png", w=1400, h=900)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 09 — MFCC Correlation Heatmap
# ════════════════════════════════════════════════════════════════════════════
print("Plot  9 : MFCC Correlation Heatmap")

mfcc_wide = mfcc.pivot_table(
    index="file", columns="coefficient", values="mean_val"
)
mfcc_wide.columns.name = None
cor_mat = mfcc_wide.corr().round(2)
coeffs  = [f"MFCC_{i}" for i in range(1, 14)]
cor_mat = cor_mat.reindex(index=coeffs, columns=coeffs)

fig = go.Figure(go.Heatmap(
    z=cor_mat.values,
    x=cor_mat.columns.tolist(),
    y=cor_mat.index.tolist(),
    text=cor_mat.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=11, family=FONT, color=TXT),
    colorscale=[
        [0.0,  "#74B9FF"],
        [0.5,  "#FAFBFD"],
        [1.0,  "#FF6B6B"],
    ],
    zmid=0, zmin=-1, zmax=1,
    colorbar=dict(
        title=dict(text="Correlation", font=dict(family=FONT, size=12, color=TXT)),
        tickfont=dict(family=FONT, size=10, color=SUB),
        bgcolor=BG,
    ),
))
fig.update_layout(
    **lay("MFCC Coefficient Correlation Matrix",
          "Pearson correlations between all 13 MFCC features",
          w=900, h=820),
    xaxis=ax("MFCC Coefficient"),
    yaxis=ax("MFCC Coefficient"),
    xaxis_tickangle=-40,
)
save(fig, "09_mfcc_correlation.png", w=900, h=820)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 11 — MFCC Mean Profile Line Chart  (with ±1 SD ribbons)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 11 : MFCC Mean Profile Line Chart")

profile = (mfcc.groupby(["cls","coeff_num"])
               .agg(grand_mean=("mean_val","mean"),
                    grand_sd=("sd_val","mean"))
               .reset_index())

LINE_COLORS = {"Background":  "#5BA3CF",
               "Cough":       "#E05555",
               "Human_Noise": "#6DC47A"}

fig = go.Figure()
for cls in CLASSES:
    d = profile[profile["cls"] == cls].sort_values("coeff_num")
    xv = d["coeff_num"].tolist()
    mu = d["grand_mean"].tolist()
    sd = d["grand_sd"].tolist()
    col = LINE_COLORS[cls]

    # Ribbon (upper → lower)
    fig.add_trace(go.Scatter(
        x=xv + xv[::-1],
        y=[m + s for m, s in zip(mu, sd)] + [m - s for m, s in zip(mu[::-1], sd[::-1])],
        fill="toself", fillcolor=hex_rgba(CLS_COLOR[cls], 0.18),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
        name=f"{cls} band",
    ))
    # Mean line
    fig.add_trace(go.Scatter(
        x=xv, y=mu, mode="lines+markers",
        name=cls,
        line=dict(color=col, width=2.5),
        marker=dict(size=9, color="white", line=dict(color=col, width=2.5)),
        hovertemplate=f"<b>{cls}</b><br>MFCC %{{x}}<br>Mean: %{{y:.2f}}<extra></extra>",
    ))

fig.update_layout(
    **lay("MFCC Mean Profile Across Coefficients",
          "Average MFCC values ± 1 SD — distinct class fingerprints"),
    yaxis=ax("Mean Value"),
    xaxis=dict(**ax("MFCC Coefficient"),
               tickvals=list(range(1,14)),
               ticktext=[f"M{i}" for i in range(1,14)]),
)
save(fig, "11_mfcc_profile.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 12 — Feature Variability (SD) Grouped Bar
# ════════════════════════════════════════════════════════════════════════════
print("Plot 12 : Feature Variability (SD)")

sd_data = (mfcc.groupby(["cls","coefficient","coeff_num"])["sd_val"]
               .mean().reset_index(name="avg_sd"))

fig = go.Figure()
for cls in CLASSES:
    d = sd_data[sd_data["cls"] == cls].sort_values("coeff_num")
    fig.add_trace(go.Bar(
        x=d["coefficient"], y=d["avg_sd"],
        name=cls,
        marker_color=hex_rgba(CLS_COLOR[cls], 0.85),
        marker_line_color=CLS_COLOR[cls], marker_line_width=1.2,
        hovertemplate=f"<b>{cls}</b><br>%{{x}}<br>Avg SD: %{{y:.2f}}<extra></extra>",
    ))
fig.update_layout(
    **lay("MFCC Feature Variability by Class",
          "Average standard deviation per coefficient — higher = more temporal variation",
          w=1400, h=700),
    barmode="group", bargap=0.2, bargroupgap=0.05,
    xaxis=ax("MFCC Coefficient"),
    yaxis=ax("Average Standard Deviation"),
    xaxis_tickangle=-40,
)
save(fig, "12_feature_variability.png", w=1400, h=700)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 14 — Unique Source Files per Class
# ════════════════════════════════════════════════════════════════════════════
print("Plot 14 : Unique Source Files")

fig = go.Figure([go.Bar(
    x=src["class"],
    y=src["unique_sources"],
    marker_color=[CLS_COLOR[c] for c in src["class"]],
    marker_line_color="white", marker_line_width=2,
    text=src["unique_sources"],
    textposition="outside",
    textfont=dict(size=15, color=TXT, family=FONT, weight="bold"),
)])
fig.update_layout(
    **lay("Data Diversity — Unique Source Recordings",
          "Number of distinct original audio files per class"),
    showlegend=False, bargap=0.45,
    xaxis=ax("Audio Class"),
    yaxis=ax("Unique Source Files"),
    yaxis_range=[0, src["unique_sources"].max() * 1.22],
)
save(fig, "14_unique_sources.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 15 — Average Windows per Source
# ════════════════════════════════════════════════════════════════════════════
print("Plot 15 : Average Windows per Source")

fig = go.Figure([go.Bar(
    x=src["class"],
    y=src["avg_windows"],
    marker_color=[CLS_COLOR[c] for c in src["class"]],
    marker_line_color="white", marker_line_width=2,
    text=[f"{v:.1f}" for v in src["avg_windows"]],
    textposition="outside",
    textfont=dict(size=15, color=TXT, family=FONT, weight="bold"),
)])
fig.update_layout(
    **lay("Data Augmentation Intensity",
          "Average 1-second windows extracted per original recording"),
    showlegend=False, bargap=0.45,
    xaxis=ax("Audio Class"),
    yaxis=ax("Avg Windows per Source"),
    yaxis_range=[0, src["avg_windows"].max() * 1.25],
)
save(fig, "15_avg_windows_per_source.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 17 — Window Position Density (KDE)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 17 : Window Position Density")

fig = go.Figure()
for cls in CLASSES:
    vals = meta[meta["class"] == cls]["window_id"].values.astype(float)
    kde  = gaussian_kde(vals, bw_method=0.35)
    xr   = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 400)
    ykde = kde(xr)
    col  = CLS_COLOR[cls]
    fig.add_trace(go.Scatter(
        x=xr, y=ykde,
        mode="lines",
        fill="tozeroy",
        fillcolor=hex_rgba(col, 0.3),
        line=dict(color=col, width=2.5),
        name=cls,
        hovertemplate=f"<b>{cls}</b><br>Window ID: %{{x:.1f}}<br>Density: %{{y:.4f}}<extra></extra>",
    ))
fig.update_layout(
    **lay("Window Position Density by Class",
          "KDE showing temporal extraction patterns across source recordings"),
    xaxis=ax("Window ID (Position in Source)"),
    yaxis=ax("Density"),
)
save(fig, "17_window_density.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 18 — Model Accuracy Horizontal Bar
# ════════════════════════════════════════════════════════════════════════════
print("Plot 18 : Model Accuracy Comparison")

ACC_COLORS = ["#74B9FF","#A29BFE","#FF6B6B","#FFEAA7"]
fig = go.Figure([go.Bar(
    x=model_perf["value"],
    y=model_perf["metric"],
    orientation="h",
    marker_color=ACC_COLORS,
    marker_line_color="white", marker_line_width=2,
    text=[f"<b>{v:.2f}%</b>" for v in model_perf["value"]],
    textposition="outside",
    textfont=dict(size=14, color=TXT, family=FONT),
)])
fig.update_layout(
    **lay("Model Performance Metrics",
          "1D CNN TinyML — Accuracy across train / val / test splits"),
    showlegend=False,
    xaxis=dict(**ax("Accuracy (%)"), range=[0, 105]),
    yaxis=dict(**ax(), autorange="reversed"),
    yaxis_gridcolor=BG,
)
save(fig, "18_model_accuracy.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 20 — MFCC Scatter  (MFCC₁ vs MFCC₂ with 85 % confidence ellipses)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 20 : MFCC Feature Scatter Plot")

scatter_pivot = mfcc[mfcc["coeff_num"].isin([1, 2])].pivot_table(
    index=["cls","file"], columns="coefficient", values="mean_val"
).reset_index()
scatter_pivot.columns.name = None

fig = go.Figure()
for cls in CLASSES:
    d   = scatter_pivot[scatter_pivot["cls"] == cls].dropna(subset=["MFCC_1","MFCC_2"])
    col = CLS_COLOR[cls]

    # Scatter points
    fig.add_trace(go.Scatter(
        x=d["MFCC_1"], y=d["MFCC_2"],
        mode="markers",
        name=cls,
        marker=dict(size=9, color=col, opacity=0.75,
                    line=dict(color="white", width=0.8)),
        hovertemplate=f"<b>{cls}</b><br>MFCC₁: %{{x:.2f}}<br>MFCC₂: %{{y:.2f}}<extra></extra>",
    ))
    # 85 % confidence ellipse
    if len(d) >= 4:
        ex, ey = conf_ellipse(d["MFCC_1"].values, d["MFCC_2"].values, n_std=1.44)
        fig.add_trace(go.Scatter(
            x=np.append(ex, ex[0]), y=np.append(ey, ey[0]),
            mode="lines",
            line=dict(color=col, width=1.8, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ))

fig.update_layout(
    **lay("Feature Space — MFCC₁ vs MFCC₂",
          "2-D projection with 85% confidence ellipses showing class cluster separation"),
    xaxis=ax("MFCC Coefficient 1 (Energy)"),
    yaxis=ax("MFCC Coefficient 2"),
)
save(fig, "20_mfcc_scatter.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 21 — MFCC Pairwise Scatter Matrix  (coefficients 1–4)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 21 : MFCC Pairwise Scatter Matrix")

wide4 = mfcc[mfcc["coeff_num"] <= 4].pivot_table(
    index=["cls","file"], columns="coefficient", values="mean_val"
).reset_index()
wide4.columns.name = None
wide4 = wide4.dropna(subset=[f"MFCC_{i}" for i in range(1, 5)])

fig = px.scatter_matrix(
    wide4,
    dimensions=[f"MFCC_{i}" for i in range(1, 5)],
    color="cls",
    color_discrete_map=CLS_COLOR,
    labels={"cls": "Class"},
    opacity=0.65,
)
fig.update_traces(
    diagonal_visible=True,
    marker=dict(size=5, line=dict(color="white", width=0.5)),
    showupperhalf=True,
)
fig.update_layout(
    **lay("MFCC Pairwise Scatter Matrix (Coefficients 1–4)",
          "Upper triangle: scatter  |  Diagonal: class density",
          w=1100, h=1000),
    legend_title_text="Class",
)
for axis in fig.layout:
    if axis.startswith("xaxis") or axis.startswith("yaxis"):
        fig.layout[axis].update(
            gridcolor=GRID, linecolor=GRID, zeroline=False,
            tickfont=dict(family=FONT, size=9, color=SUB),
        )
save(fig, "21_mfcc_scatter_matrix.png", w=1100, h=1000)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 23 — Stacked Area Chart  (cumulative sample accumulation)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 23 : Stacked Area Chart")

# Build per-class sample index
area_frames = []
for cls in CLASSES:
    d = meta[meta["class"] == cls].reset_index(drop=True)
    d["idx"] = range(len(d))
    area_frames.append(d[["class","idx"]])
area_df = pd.concat(area_frames)

bins = 35
fig  = go.Figure()
for cls in CLASSES:
    d = area_df[area_df["class"] == cls]
    counts, edges = np.histogram(d["idx"], bins=bins)
    x_mid = 0.5 * (edges[:-1] + edges[1:])
    fig.add_trace(go.Scatter(
        x=x_mid, y=counts,
        mode="lines",
        line=dict(color=CLS_COLOR[cls], width=0.5),
        fill="tonexty" if cls != CLASSES[0] else "tozeroy",
        fillcolor=hex_rgba(CLS_COLOR[cls], 0.70),
        name=cls,
        stackgroup="one",
        hovertemplate=f"<b>{cls}</b><br>x: %{{x:.0f}}<br>Count: %{{y}}<extra></extra>",
    ))
fig.update_layout(
    **lay("Dataset Growth — Stacked Area Chart",
          "Cumulative sample accumulation across audio classes"),
    xaxis=ax("Sample Index"),
    yaxis=ax("Count"),
)
save(fig, "23_stacked_area.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 25 — Radar Chart  (MFCC coefficients 1–10 normalised)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 25 : Radar Chart — Class Fingerprints")

radar_df = (mfcc[mfcc["coeff_num"] <= 10]
            .groupby(["cls","coeff_num"])["mean_val"]
            .mean().reset_index())
vmin, vmax = radar_df["mean_val"].min(), radar_df["mean_val"].max()
radar_df["norm"] = (radar_df["mean_val"] - vmin) / (vmax - vmin + 1e-9)

angles = [f"MFCC_{i}" for i in range(1, 11)]

fig = go.Figure()
for cls in CLASSES:
    d   = radar_df[radar_df["cls"] == cls].sort_values("coeff_num")
    col = CLS_COLOR[cls]
    r   = d["norm"].tolist() + [d["norm"].iloc[0]]     # close polygon
    th  = angles + [angles[0]]
    fig.add_trace(go.Scatterpolar(
        r=r, theta=th,
        fill="toself",
        fillcolor=hex_rgba(col, 0.22),
        line=dict(color=col, width=2.2),
        marker=dict(size=7, color=col,
                    line=dict(color="white", width=1.5)),
        name=cls,
        hovertemplate=f"<b>{cls}</b><br>%{{theta}}<br>Normalised: %{{r:.3f}}<extra></extra>",
    ))
fig.update_layout(
    **lay("MFCC Feature Radar — Class Fingerprints",
          "Normalised MFCC coefficients 1–10 showing distinct class signatures",
          w=900, h=900),
    polar=dict(
        bgcolor=BG,
        angularaxis=dict(
            gridcolor=GRID, linecolor=GRID,
            tickfont=dict(family=FONT, size=12, color=TXT),
        ),
        radialaxis=dict(
            gridcolor=GRID, linecolor=GRID, showline=True,
            rangemode="tozero", range=[0, 1.05],
            tickfont=dict(family=FONT, size=9, color=SUB),
        ),
    ),
)
save(fig, "25_radar_chart.png", w=900, h=900)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 26 — Parallel Coordinates Plot  (MFCC coefficients 1–8)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 26 : Parallel Coordinates Plot")

wide8 = mfcc[mfcc["coeff_num"] <= 8].pivot_table(
    index=["cls","file"], columns="coefficient", values="mean_val"
).reset_index()
wide8.columns.name = None
wide8 = wide8.dropna(subset=[f"MFCC_{i}" for i in range(1, 9)])
wide8["cls_num"] = wide8["cls"].map(
    {"Background": 0, "Cough": 1, "Human_Noise": 2}
)

dims = [
    dict(
        range=[0, 2],
        tickvals=[0, 1, 2],
        ticktext=CLASSES,
        label="Class",
        values=wide8["cls_num"],
    )
] + [
    dict(label=f"MFCC_{i}",
         values=wide8[f"MFCC_{i}"],
         range=[wide8[f"MFCC_{i}"].min() - 5, wide8[f"MFCC_{i}"].max() + 5])
    for i in range(1, 9)
]

fig = go.Figure(go.Parcoords(
    line=dict(
        color=wide8["cls_num"],
        colorscale=[[0, "#7EC8E3"], [0.5, "#FF6B6B"], [1, "#95E1A3"]],
        cmin=0, cmax=2, showscale=False,
    ),
    dimensions=dims,
    labelangle=-20,
    labelfont=dict(family=FONT, size=12, color=TXT),
    tickfont=dict(family=FONT, size=9, color=SUB),
))
fig.update_layout(
    **lay("Parallel Coordinates — MFCC Feature Trajectories (1–8)",
          "Individual samples with colour-coded class showing acoustic feature patterns",
          w=1300, h=700),
)
save(fig, "26_parallel_coordinates.png", w=1300, h=700)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 30 — Comprehensive Multi-Panel Summary  (2 × 2 grid)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 30 : Comprehensive Multi-Panel Summary")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "<b>A) Class Distribution</b>",
        "<b>B) Train / Test Split</b>",
        "<b>C) Model Accuracy (%)</b>",
        "<b>D) MFCC Mean Profile</b>",
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.12,
    specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
    ],
)

# A) class distribution
for _, row in cls_tot.iterrows():
    fig.add_trace(go.Bar(
        x=[row["class"]], y=[row["total"]],
        marker_color=CLS_COLOR[row["class"]],
        marker_line_color="white", marker_line_width=1.5,
        text=[str(row["total"])], textposition="outside",
        textfont=dict(size=12, color=TXT),
        showlegend=False, name=row["class"],
    ), row=1, col=1)

# B) train/test split
for split, col in [("train","#74B9FF"),("test","#FD79A8")]:
    d = cls_split[cls_split["split"] == split]
    fig.add_trace(go.Bar(
        x=d["class"], y=d["count"],
        name=f"{split.capitalize()} Set",
        marker_color=col,
        marker_line_color="white", marker_line_width=1.5,
        text=d["count"], textposition="outside",
        textfont=dict(size=11, color=TXT),
        showlegend=True,
    ), row=1, col=2)

# C) model accuracy
fig.add_trace(go.Bar(
    x=model_perf["value"],
    y=model_perf["metric"],
    orientation="h",
    marker_color=["#74B9FF","#A29BFE","#FF6B6B","#FFEAA7"],
    marker_line_color="white", marker_line_width=1.5,
    text=[f"{v:.1f}%" for v in model_perf["value"]],
    textposition="outside",
    textfont=dict(size=11, color=TXT),
    showlegend=False,
), row=2, col=1)

# D) MFCC profile per class
for cls in CLASSES:
    d   = profile[profile["cls"] == cls].sort_values("coeff_num")
    col = LINE_COLORS[cls]
    fig.add_trace(go.Scatter(
        x=d["coeff_num"], y=d["grand_mean"],
        mode="lines+markers",
        name=cls,
        line=dict(color=col, width=2),
        marker=dict(size=6, color="white", line=dict(color=col, width=2)),
        showlegend=True,
    ), row=2, col=2)

fig.update_layout(
    **lay("AeroGuard TinyML — Comprehensive Summary",
          "Dataset composition  ·  Model performance  ·  Acoustic feature profiles",
          w=1500, h=1100),
    barmode="group",
)
for axis in fig.layout:
    if axis.startswith("xaxis") or axis.startswith("yaxis"):
        fig.layout[axis].update(
            gridcolor=GRID, linecolor=GRID, zeroline=False,
            tickfont=dict(family=FONT, size=9, color=SUB),
        )
fig.update_annotations(font=dict(family=FONT, size=14, color=TXT))
save(fig, "30_comprehensive_summary.png", w=1500, h=1100)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 31 — 3D MFCC Scatter Plot  (MFCC₁ × MFCC₂ × MFCC₃)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 31 : 3D MFCC Scatter Plot")

wide3 = mfcc[mfcc["coeff_num"] <= 3].pivot_table(
    index=["cls", "file"], columns="coefficient", values="mean_val"
).reset_index()
wide3.columns.name = None
wide3 = wide3.dropna(subset=["MFCC_1", "MFCC_2", "MFCC_3"])

fig = go.Figure()
for cls in CLASSES:
    d = wide3[wide3["cls"] == cls]
    fig.add_trace(go.Scatter3d(
        x=d["MFCC_1"],
        y=d["MFCC_2"],
        z=d["MFCC_3"],
        mode="markers",
        name=cls,
        marker=dict(
            size=6,
            color=CLS_COLOR[cls],
            opacity=0.78,
            line=dict(color="white", width=0.6),
        ),
        hovertemplate=(
            f"<b>{cls}</b><br>"
            "MFCC₁: %{x:.2f}<br>MFCC₂: %{y:.2f}<br>MFCC₃: %{z:.2f}<extra></extra>"
        ),
    ))
fig.update_layout(
    **lay("3D Acoustic Feature Space — MFCC₁ × MFCC₂ × MFCC₃",
          "A PowerBI-native 3D view of class separation in feature space",
          w=1200, h=900),
    scene=dict(
        bgcolor=BG,
        xaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=SUB, title="MFCC₁"),
        yaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=SUB, title="MFCC₂"),
        zaxis=dict(backgroundcolor=BG, gridcolor=GRID, color=SUB, title="MFCC₃"),
        camera=dict(eye=dict(x=1.55, y=1.45, z=1.2)),
    ),
)
save(fig, "31_mfcc_3d_scatter.png", w=1200, h=900)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 32 — Sunburst Chart  (Dataset → Split → Class)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 32 : Sunburst Chart")

sunburst_rows = [{"id": "Dataset", "label": "Dataset", "parent": "", "value": len(meta)}]
for split_name, split_count in split_totals.itertuples(index=False):
    split_id = f"Dataset/{split_name}"
    sunburst_rows.append({
        "id": split_id,
        "label": split_name.capitalize(),
        "parent": "Dataset",
        "value": int(split_count),
    })
for _, row in cls_split.iterrows():
    split_id = f"Dataset/{row['split']}"
    sunburst_rows.append({
        "id": f"{split_id}/{row['class']}",
        "label": row["class"],
        "parent": split_id,
        "value": int(row["count"]),
    })
sunburst_df = pd.DataFrame(sunburst_rows)

sunburst_colors = [
    "#DDEAF7" if label == "Dataset" else
    ("#A0C4FF" if label == "Train" else
     "#FFC6FF" if label == "Test" else CLS_COLOR.get(label, "#DDEAF7"))
    for label in sunburst_df["label"]
]

fig = go.Figure(go.Sunburst(
    ids=sunburst_df["id"],
    labels=sunburst_df["label"],
    parents=sunburst_df["parent"],
    values=sunburst_df["value"],
    branchvalues="total",
    insidetextorientation="radial",
    marker=dict(colors=sunburst_colors, line=dict(color="white", width=2)),
    hovertemplate="<b>%{label}</b><br>Samples: %{value}<extra></extra>",
))
fig.update_layout(
    **lay("Dataset Hierarchy — Sunburst Chart",
          "PowerBI-native hierarchical view: Dataset → Split → Class",
          w=1000, h=1000),
)
save(fig, "32_dataset_sunburst.png", w=1000, h=1000)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 33 — Sankey Diagram  (Dataset flow through split and class)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 33 : Sankey Diagram")

labels = ["All Samples", "Train", "Test"] + CLASSES
node_index = {label: idx for idx, label in enumerate(labels)}

sources = []
targets = []
values = []
link_colors = []

for split_name, split_count in split_totals.itertuples(index=False):
    sources.append(node_index["All Samples"])
    targets.append(node_index[split_name.capitalize()])
    values.append(int(split_count))
    link_colors.append("rgba(160,196,255,0.45)" if split_name == "train" else "rgba(255,198,255,0.45)")

for _, row in cls_split.iterrows():
    sources.append(node_index[row["split"].capitalize()])
    targets.append(node_index[row["class"]])
    values.append(int(row["count"]))
    link_colors.append(hex_rgba(CLS_COLOR[row["class"]], 0.40))

fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=22,
        thickness=20,
        line=dict(color="white", width=1.5),
        label=labels,
        color=["#DDEAF7", "#A0C4FF", "#FFC6FF"] + [CLS_COLOR[c] for c in CLASSES],
        hovertemplate="<b>%{label}</b><extra></extra>",
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors,
        hovertemplate="Flow: %{value} samples<extra></extra>",
    ),
))
fig.update_layout(
    **lay("Dataset Flow — Sankey Diagram",
          "PowerBI-native flow of samples from dataset to split and final class",
          w=1300, h=800),
)
save(fig, "33_dataset_sankey.png", w=1300, h=800)

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 34 — Waterfall Chart  (Model size transformation)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 34 : Waterfall Chart")

keras_size = model_sizes.loc[model_sizes["stage"] == "Keras (.keras)", "size_kb"].iat[0]
float_size = model_sizes.loc[model_sizes["stage"] == "TFLite Float32", "size_kb"].iat[0]
int8_size = model_sizes.loc[model_sizes["stage"] == "TFLite Int8", "size_kb"].iat[0]

fig = go.Figure(go.Waterfall(
    orientation="v",
    measure=["absolute", "relative", "relative", "total"],
    x=["Keras Base", "Float32 Conversion", "INT8 Quantization", "Final INT8 Model"],
    y=[keras_size, float_size - keras_size, int8_size - float_size, 0],
    text=[
        f"{keras_size:.2f} KB",
        f"{float_size - keras_size:+.2f} KB",
        f"{int8_size - float_size:+.2f} KB",
        f"{int8_size:.2f} KB",
    ],
    textposition="outside",
    connector=dict(line=dict(color="#B0BEC5", width=1.5)),
    increasing=dict(marker=dict(color="#A0C4FF")),
    decreasing=dict(marker=dict(color="#FF8FAB")),
    totals=dict(marker=dict(color="#95E1A3")),
    hovertemplate="<b>%{x}</b><br>%{y:.2f} KB<extra></extra>",
))
fig.update_layout(
    **lay("Model Compression Journey — Waterfall Chart",
          "PowerBI-native view of model size changes across conversion and quantization"),
    xaxis=ax("Pipeline Stage"),
    yaxis=ax("Model Size Delta (KB)"),
)
save(fig, "34_model_size_waterfall.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 35 — Funnel Chart  (Dataset to accurate test predictions)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 35 : Funnel Chart")

funnel_labels = [
    "All Dataset Samples",
    "Training Samples",
    "Test Samples",
    "Correct Test Predictions",
]
funnel_values = [len(meta), training_samples, test_samples, test_correct]

fig = go.Figure(go.Funnel(
    y=funnel_labels,
    x=funnel_values,
    text=[f"{value:,}" for value in funnel_values],
    textposition="inside",
    textfont=dict(family=FONT, size=13, color="white"),
    marker=dict(
        color=["#A0C4FF", "#7EC8E3", "#FFD6A5", "#95E1A3"],
        line=dict(color="white", width=2),
    ),
    connector=dict(line=dict(color="#B0BEC5", width=1.2)),
    hovertemplate="<b>%{y}</b><br>%{x:,} items<extra></extra>",
))
fig.update_layout(
    **lay("Model Evaluation Funnel",
          "PowerBI-native funnel from full dataset to correctly classified test samples"),
    xaxis=ax("Sample Count"),
    yaxis=ax("Pipeline Stage"),
)
save(fig, "35_model_evaluation_funnel.png")

# ════════════════════════════════════════════════════════════════════════════
#  PLOT 36 — Gauge / Indicator  (Test accuracy)
# ════════════════════════════════════════════════════════════════════════════
print("Plot 36 : Gauge Indicator")

fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=test_accuracy,
    number={"suffix": "%", "font": {"family": FONT, "size": 46, "color": TXT}},
    delta={"reference": 90, "increasing": {"color": "#2ECC71"}},
    title={
        "text": "<b>Test Accuracy</b><br><span style='font-size:14px;color:#5D6D7E'>PowerBI-native KPI gauge</span>",
        "font": {"family": FONT, "size": 20, "color": TXT},
    },
    gauge={
        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": SUB},
        "bar": {"color": "#95E1A3", "thickness": 0.34},
        "bgcolor": BG,
        "borderwidth": 2,
        "bordercolor": GRID,
        "steps": [
            {"range": [0, 60], "color": "#FADADD"},
            {"range": [60, 80], "color": "#FFF1C1"},
            {"range": [80, 100], "color": "#D9F5E1"},
        ],
        "threshold": {
            "line": {"color": "#FF6B6B", "width": 5},
            "thickness": 0.8,
            "value": test_accuracy,
        },
    },
))
fig.update_layout(
    **lay("Deployment KPI — Test Accuracy Gauge",
          "A PowerBI-style executive indicator for deployment readiness",
          w=950, h=650),
)
save(fig, "36_test_accuracy_gauge.png", w=950, h=650)

# ── Final Summary ──────────────────────────────────────────────────────────────
print()
print("=" * 62)
print("  ALL 23 VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 62)
print(f"  Output folder : {OUTPUT_DIR}")
total_pngs = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])
print(f"  Total PNGs    : {total_pngs}\n")
print("  Index:")
print("  ─────────────────────────────────────────────────────────")
print("   01  Class Distribution Bar Chart")
print("   02  Train/Test Split Grouped Bars")
print("   06  MFCC Coefficient Box Plots")
print("   07  MFCC Violin + Points (faceted)")
print("   09  MFCC Correlation Heatmap")
print("   11  MFCC Mean Profile with SD Ribbons")
print("   12  Feature Variability (SD) Bars")
print("   14  Unique Source Files Bar")
print("   15  Average Windows per Source")
print("   17  Window Position KDE Density")
print("   18  Model Accuracy Horizontal Bars")
print("   20  MFCC Scatter with Confidence Ellipses")
print("   21  MFCC Pairwise Scatter Matrix")
print("   23  Stacked Area Chart")
print("   25  Radar Chart — Class Fingerprints")
print("   26  Parallel Coordinates Plot")
print("   30  2×2 Comprehensive Summary Panel")
print("   31  3D MFCC Scatter Plot")
print("   32  Dataset Sunburst Chart")
print("   33  Dataset Sankey Diagram")
print("   34  Model Size Waterfall Chart")
print("   35  Model Evaluation Funnel")
print("   36  Test Accuracy Gauge Indicator")
print("  ─────────────────────────────────────────────────────────")
print("\n  Capstone Project 3 — Power BI Visualization Complete!")
print("=" * 62)
