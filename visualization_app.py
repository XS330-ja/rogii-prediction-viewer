"""Streamlit app: 指定した well_id の TVT 推移を可視化する。

起動:
    pip install streamlit plotly
    streamlit run visualization_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

COLOR_KNOWN = "#2A6FB8"   # 予測開始点より前（既知）
COLOR_PRED = "#E45756"    # 予測開始点より後（予測対象）
COLOR_SPLIT = "#111"      # 予測開始点マーカー


@st.cache_data(show_spinner=False)
def list_well_ids(directory: Path) -> list[str]:
    return sorted({p.name.split("__")[0] for p in directory.glob("*__horizontal_well.csv")})


@st.cache_data(show_spinner=False)
def load_horizontal(well_id: str, split: str) -> pd.DataFrame:
    base = TRAIN_DIR if split == "train" else TEST_DIR
    return pd.read_csv(base / f"{well_id}__horizontal_well.csv")


@st.cache_data(show_spinner=False)
def load_typewell(well_id: str, split: str) -> pd.DataFrame:
    base = TRAIN_DIR if split == "train" else TEST_DIR
    return pd.read_csv(base / f"{well_id}__typewell.csv")


@st.cache_data(show_spinner=False)
def png_path(well_id: str, split: str) -> Path | None:
    base = TRAIN_DIR if split == "train" else TEST_DIR
    p = base / f"{well_id}.png"
    return p if p.exists() else None


def find_prediction_start(df_h: pd.DataFrame) -> int | None:
    """TVT_input が初めて欠損する index を返す。欠損が無ければ None。"""
    if "TVT_input" not in df_h.columns:
        return None
    mask = df_h["TVT_input"].isna().values
    if not mask.any():
        return None
    return int(mask.argmax())


def kpi_row(df_h: pd.DataFrame, split_idx: int | None) -> None:
    n = len(df_h)
    n_fill = int(df_h["TVT_input"].notna().sum()) if "TVT_input" in df_h.columns else 0
    n_miss = n - n_fill
    has_tvt = "TVT" in df_h.columns
    cols = st.columns(5)
    cols[0].metric("行数 (MD points)", f"{n:,}")
    cols[1].metric("既知区間 (filled)", f"{n_fill:,}")
    cols[2].metric("予測区間 (missing)", f"{n_miss:,}")
    if split_idx is not None:
        md_split = float(df_h["MD"].iloc[split_idx])
        cols[3].metric("予測開始 MD (ft)", f"{md_split:.0f}")
    else:
        cols[3].metric("予測開始 MD", "—")
    if has_tvt:
        cols[4].metric("TVT レンジ (ft)", f"{df_h['TVT'].min():.1f} – {df_h['TVT'].max():.1f}")
    else:
        cols[4].metric("TVT", "—(test: 未提供)")


def plot_tvt_vs_md(df_h: pd.DataFrame, split_idx: int | None) -> go.Figure:
    """TVT の推移を MD 軸でプロット。予測開始点で線色を切り替え、開始点に点を打つ。"""
    fig = go.Figure()
    has_tvt = "TVT" in df_h.columns
    md = df_h["MD"].values

    if has_tvt:
        tvt = df_h["TVT"].values
        if split_idx is None:
            # 予測開始点が無い場合（全部既知 or 全部予測）→ 単色で描画
            fig.add_trace(go.Scatter(
                x=md, y=tvt, mode="lines", name="TVT",
                line=dict(color=COLOR_KNOWN, width=1.6),
                hovertemplate="MD=%{x}<br>TVT=%{y:.2f}<extra></extra>",
            ))
        else:
            # 既知区間: [0, split_idx]（端点を含めて連続線にする）
            fig.add_trace(go.Scatter(
                x=md[: split_idx + 1], y=tvt[: split_idx + 1],
                mode="lines", name="TVT (既知)",
                line=dict(color=COLOR_KNOWN, width=1.6),
                hovertemplate="MD=%{x}<br>TVT=%{y:.2f}<extra></extra>",
            ))
            # 予測区間: [split_idx, end]
            fig.add_trace(go.Scatter(
                x=md[split_idx:], y=tvt[split_idx:],
                mode="lines", name="TVT (予測対象)",
                line=dict(color=COLOR_PRED, width=1.6),
                hovertemplate="MD=%{x}<br>TVT=%{y:.2f}<extra></extra>",
            ))
    else:
        # test: TVT が無いので TVT_input の既知部のみ描画
        ti = df_h["TVT_input"].values
        if split_idx is None:
            fig.add_trace(go.Scatter(
                x=md, y=ti, mode="lines", name="TVT_input",
                line=dict(color=COLOR_KNOWN, width=1.6),
                hovertemplate="MD=%{x}<br>TVT_input=%{y:.2f}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=md[: split_idx], y=ti[: split_idx],
                mode="lines", name="TVT_input (既知)",
                line=dict(color=COLOR_KNOWN, width=1.6),
                hovertemplate="MD=%{x}<br>TVT_input=%{y:.2f}<extra></extra>",
            ))

    # 予測開始点のマーカー
    if split_idx is not None:
        md_s = float(df_h["MD"].iloc[split_idx])
        if has_tvt:
            y_s = float(df_h["TVT"].iloc[split_idx])
        else:
            # TVT_input は split_idx で NaN なので、直前の値を使う
            prev_idx = max(split_idx - 1, 0)
            y_s = float(df_h["TVT_input"].iloc[prev_idx])
        fig.add_trace(go.Scatter(
            x=[md_s], y=[y_s], mode="markers", name="予測開始点",
            marker=dict(color=COLOR_SPLIT, size=11, symbol="circle",
                        line=dict(color="white", width=2)),
            hovertemplate=f"prediction start<br>MD={md_s:.0f}<br>TVT={y_s:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="TVT 推移 (along MD)",
        xaxis_title="MD (ft)",
        yaxis_title="TVT (ft)",
        yaxis=dict(autorange="reversed"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Wellbore TVT Viewer", layout="wide")
    st.title("Wellbore TVT Viewer")
    st.caption("選択した well_id について TVT の推移を可視化")

    with st.sidebar:
        split = st.radio("Split", ["train", "test"], horizontal=True)
        ids = list_well_ids(TRAIN_DIR if split == "train" else TEST_DIR)
        if not ids:
            st.error(f"{split} ディレクトリに well が見つかりません")
            st.stop()
        well_id = st.selectbox(f"well_id ({len(ids)} 件)", ids, index=0)
        st.divider()
        show_png = st.checkbox("配布 PNG を表示", value=False)

    df_h = load_horizontal(well_id, split)
    split_idx = find_prediction_start(df_h)

    st.subheader(f"well_id: `{well_id}`  ({split})")
    kpi_row(df_h, split_idx)

    st.plotly_chart(plot_tvt_vs_md(df_h, split_idx), use_container_width=True)

    if show_png:
        p = png_path(well_id, split)
        if p is None:
            st.info(f"{split}/{well_id}.png は存在しません")
        else:
            st.image(str(p), caption=p.name, use_container_width=True)

    with st.expander("生データ (horizontal_well.csv の先頭 200 行)"):
        st.dataframe(df_h.head(200))


if __name__ == "__main__":
    main()
