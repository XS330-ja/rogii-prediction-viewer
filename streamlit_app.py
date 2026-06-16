"""Streamlit Cloud 用: well ごとに実測値と予測値 (NN / LGB / blend) を可視化する。

ローカル版 (prediction_viewer_app.py) との違い:
  - 巨大な生データ・train/ に依存せず、`app_data/*.parquet` (合計 ~49MB) のみ読む
    (build_app_data.py で事前生成)
  - PNG は同梱しない (サイズ制約のため)

ローカル確認:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "app_data"
PRED_PARQUET = DATA_DIR / "predictions.parquet"
KNOWN_PARQUET = DATA_DIR / "known_tvt.parquet"
SUMMARY_PARQUET = DATA_DIR / "well_summary.parquet"

COLOR_TRUE = "#111111"   # 実測値 (予測区間)
COLOR_KNOWN = "#9A9A9A"  # 既知区間 (予測区間以前) の実測 TVT
COLOR_NN = "#2A6FB8"     # LSTM
COLOR_LGB = "#E45756"    # physics LGB
COLOR_HEU026 = "#F58518" # heuristic exp026
COLOR_BLEND = "#54A24B"  # blend


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _sig(path: Path) -> tuple[int, int]:
    """ファイルの (更新時刻, サイズ) を返す。キャッシュキーに含めることで、
    parquet が更新されたら (= 再デプロイ時) 自動でキャッシュを無効化する。"""
    s = path.stat()
    return (s.st_mtime_ns, s.st_size)


@st.cache_data(show_spinner="データ読み込み中 …")
def load_predictions(sig: tuple[int, int]) -> pd.DataFrame:
    return pd.read_parquet(PRED_PARQUET)


@st.cache_data(show_spinner=False)
def load_summary(sig: tuple[int, int]) -> pd.DataFrame:
    return pd.read_parquet(SUMMARY_PARQUET).set_index("well")


@st.cache_data(show_spinner=False)
def load_known(well: str, sig: tuple[int, int]) -> pd.DataFrame:
    # 必要な well だけ読み出す (filters で I/O を最小化)
    k = pd.read_parquet(KNOWN_PARQUET, filters=[("well", "==", well)])
    return k[["row_idx", "rank_in_well", "tvt_known"]]


def plot_well(sub, w_nn, x_col, show, known, split_x):
    x = sub[x_col].to_numpy()
    blend = w_nn * sub["tvt_pred"].to_numpy() + (1 - w_nn) * sub["oof_tvt"].to_numpy()

    fig = go.Figure()
    if show["known"] and known is not None and len(known):
        fig.add_trace(go.Scatter(
            x=known[x_col], y=known["tvt_known"], mode="lines", name="実測 (予測区間以前)",
            line=dict(color=COLOR_KNOWN, width=1.8),
            hovertemplate=f"{x_col}=%{{x}}<br>known=%{{y:.2f}}<extra></extra>",
        ))
    if show["true"]:
        fig.add_trace(go.Scatter(
            x=x, y=sub["tvt_true"], mode="lines", name="実測 (tvt_true, 予測区間)",
            line=dict(color=COLOR_TRUE, width=2.4),
            hovertemplate=f"{x_col}=%{{x}}<br>true=%{{y:.2f}}<extra></extra>",
        ))
    if show["nn"]:
        fig.add_trace(go.Scatter(
            x=x, y=sub["tvt_pred"], mode="lines", name="NN (LSTM)",
            line=dict(color=COLOR_NN, width=1.4),
            hovertemplate=f"{x_col}=%{{x}}<br>NN=%{{y:.2f}}<extra></extra>",
        ))
    if show["lgb"]:
        fig.add_trace(go.Scatter(
            x=x, y=sub["oof_tvt"], mode="lines", name="physics LGB",
            line=dict(color=COLOR_LGB, width=1.4),
            hovertemplate=f"{x_col}=%{{x}}<br>LGB=%{{y:.2f}}<extra></extra>",
        ))
    if show["heu026"]:
        fig.add_trace(go.Scatter(
            x=x, y=sub["heu026"], mode="lines", name="heuristic exp026",
            line=dict(color=COLOR_HEU026, width=1.4),
            hovertemplate=f"{x_col}=%{{x}}<br>heu026=%{{y:.2f}}<extra></extra>",
        ))
    if show["blend"]:
        fig.add_trace(go.Scatter(
            x=x, y=blend, mode="lines", name=f"blend (NN {w_nn:.1f} : LGB {1 - w_nn:.1f})",
            line=dict(color=COLOR_BLEND, width=1.8, dash="dot"),
            hovertemplate=f"{x_col}=%{{x}}<br>blend=%{{y:.2f}}<extra></extra>",
        ))

    if split_x is not None:
        fig.add_vline(x=split_x, line=dict(color="#111", width=1, dash="dash"),
                      annotation_text="予測開始", annotation_position="top")

    fig.update_layout(
        title="実測 vs 予測 (TVT)",
        xaxis_title=x_col, yaxis_title="TVT (ft)",
        yaxis=dict(autorange="reversed"),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def plot_residual(sub, w_nn, x_col, show):
    x = sub[x_col].to_numpy()
    yt = sub["tvt_true"].to_numpy()
    blend = w_nn * sub["tvt_pred"].to_numpy() + (1 - w_nn) * sub["oof_tvt"].to_numpy()

    fig = go.Figure()
    if show["nn"]:
        fig.add_trace(go.Scatter(x=x, y=sub["tvt_pred"].to_numpy() - yt, mode="lines",
                                 name="NN", line=dict(color=COLOR_NN, width=1.2)))
    if show["lgb"]:
        fig.add_trace(go.Scatter(x=x, y=sub["oof_tvt"].to_numpy() - yt, mode="lines",
                                 name="LGB", line=dict(color=COLOR_LGB, width=1.2)))
    if show["heu026"]:
        fig.add_trace(go.Scatter(x=x, y=sub["heu026"].to_numpy() - yt, mode="lines",
                                 name="heu026", line=dict(color=COLOR_HEU026, width=1.2)))
    if show["blend"]:
        fig.add_trace(go.Scatter(x=x, y=blend - yt, mode="lines",
                                 name="blend", line=dict(color=COLOR_BLEND, width=1.4, dash="dot")))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    fig.update_layout(
        title="残差 (予測 − 実測)",
        xaxis_title=x_col, yaxis_title="残差 (ft)",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Prediction Viewer", layout="wide")
    st.title("実測 vs 予測 Viewer (per well)")
    st.caption("pred_exp010 (LSTM) / physics LGB / heuristic exp026 の OOF を well ごとに比較")

    if not PRED_PARQUET.exists():
        st.error("app_data/predictions.parquet が見つかりません。build_app_data.py を実行してください。")
        st.stop()

    df = load_predictions(_sig(PRED_PARQUET))
    summary = load_summary(_sig(SUMMARY_PARQUET))
    wells = summary.index.tolist()

    with st.sidebar:
        st.header("設定")
        sort_by = st.selectbox(
            "well 並び順",
            ["well 名", "NN が苦手な順", "LGB が苦手な順", "heu026 が苦手な順", "行数が多い順"],
        )
        if sort_by == "NN が苦手な順":
            ordered = summary.sort_values("rmse_NN", ascending=False).index.tolist()
        elif sort_by == "LGB が苦手な順":
            ordered = summary.sort_values("rmse_LGB", ascending=False).index.tolist()
        elif sort_by == "heu026 が苦手な順":
            ordered = summary.sort_values("rmse_HEU026", ascending=False).index.tolist()
        elif sort_by == "行数が多い順":
            ordered = summary.sort_values("n_rows", ascending=False).index.tolist()
        else:
            ordered = wells
        well = st.selectbox(f"well ({len(wells)} 件)", ordered, index=0)

        st.divider()
        w_nn = st.slider("blend 重み (NN の比率)", 0.0, 1.0, 0.4, 0.1)
        x_col = st.radio("横軸", ["row_idx", "rank_in_well"], horizontal=True,
                         help="rank_in_well = 予測区間先頭からの行数")
        st.divider()
        st.write("表示する系列")
        show = {
            "known": st.checkbox("実測 (予測区間以前)", value=True),
            "true": st.checkbox("実測 (tvt_true, 予測区間)", value=True),
            "nn": st.checkbox("NN (LSTM)", value=True),
            "lgb": st.checkbox("physics LGB", value=True),
            "heu026": st.checkbox("heuristic exp026", value=True),
            "blend": st.checkbox("blend", value=True),
        }
        show_resid = st.checkbox("残差プロットを表示", value=True)

    sub = df[df["well"] == well].sort_values(x_col)
    yt = sub["tvt_true"].to_numpy()
    blend = w_nn * sub["tvt_pred"].to_numpy() + (1 - w_nn) * sub["oof_tvt"].to_numpy()

    pred_start = int(sub["row_idx"].min())
    known = load_known(well, _sig(KNOWN_PARQUET))
    split_x = 0.0 if x_col == "rank_in_well" else float(pred_start)

    st.subheader(f"well: `{well}`")
    c = st.columns(6)
    c[0].metric("予測区間 行数", f"{len(sub):,}")
    c[1].metric("RMSE NN", f"{rmse(yt, sub['tvt_pred'].to_numpy()):.3f}")
    c[2].metric("RMSE LGB", f"{rmse(yt, sub['oof_tvt'].to_numpy()):.3f}")
    c[3].metric("RMSE heu026", f"{rmse(yt, sub['heu026'].to_numpy()):.3f}")
    c[4].metric(f"RMSE blend ({w_nn:.1f})", f"{rmse(yt, blend):.3f}")
    c[5].metric("既知区間 行数", f"{len(known):,}")

    st.plotly_chart(plot_well(sub, w_nn, x_col, show, known, split_x), width="stretch")
    if show_resid:
        st.plotly_chart(plot_residual(sub, w_nn, x_col, show), width="stretch")

    with st.expander("well 別 RMSE 一覧 (全 well)"):
        st.dataframe(summary.round(3), width="stretch")


if __name__ == "__main__":
    main()
