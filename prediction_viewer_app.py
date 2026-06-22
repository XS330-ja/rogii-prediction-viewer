"""Streamlit app: well ごとに実測値 (tvt_true) と予測値 (NN / LGB / blend) を可視化する。

データ:
    - pred_exp010.csv                         … LSTM 系 OOF (絶対 TVT, tvt_pred / tvt_true)
    - models/.../oof_predictions.parquet      … physics LGB OOF (オフセット oof_lgb1)
  LGB は `oof_tvt = oof_lgb1 + (tvt_true - target)` で絶対 TVT へ戻す
  (ROGII(Blending).ipynb と同じ変換)。

起動:
    pip install streamlit plotly
    streamlit run prediction_viewer_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent
PRED_EXP010 = DATA_DIR / "pred_exp010.csv"
OOF_PARQUET = DATA_DIR / "models" / "physics-informed-baseline" / "artefacts" / "oof_predictions.parquet"
HEURISTIC026 = DATA_DIR / "public_notebook_heuristic_exp026.csv"
TRAIN_DIR = DATA_DIR / "train"

# 系列ごとの色
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


@st.cache_data(show_spinner="予測ファイルを読み込み中 …")
def load_data() -> pd.DataFrame:
    """各ファイルを結合し、LGB を絶対 TVT へ戻した DataFrame を返す。"""
    pred = pd.read_csv(
        PRED_EXP010,
        usecols=["well", "row_idx", "id", "tvt_pred", "fold", "eval_mask", "tvt_true"],
        dtype={"well": "category", "row_idx": "int32", "tvt_pred": "float32", "tvt_true": "float32"},
    )
    oof = pd.read_parquet(OOF_PARQUET, columns=["id", "target", "oof_lgb1"])

    df = pred.merge(oof, on="id", how="inner")
    # LGB オフセット予測 → 絶対 TVT
    df["oof_tvt"] = (df["oof_lgb1"] + (df["tvt_true"] - df["target"])).astype("float32")

    # heuristic exp026 を追加。id を持たないため row_idx を復元して結合
    # (ROGII(Blending).ipynb と同じ: row_idx = 予測開始 + well 内 MD 順位)。
    if HEURISTIC026.exists():
        heu = pd.read_csv(HEURISTIC026, usecols=["well", "MD", "tvt_pred"]).rename(columns={"tvt_pred": "heu026"})
        pred_start = {str(k): int(v) for k, v in pred.groupby("well", observed=True)["row_idx"].min().items()}
        heu = heu.sort_values(["well", "MD"])
        heu["row_idx"] = heu.groupby("well").cumcount() + heu["well"].map(pred_start).astype("int64")
        heu["id"] = heu["well"] + "_" + heu["row_idx"].astype(str)
        df = df.merge(heu[["id", "heu026"]], on="id", how="left")
        df["heu026"] = df["heu026"].astype("float32")
    else:
        df["heu026"] = np.float32("nan")

    df = df[df["eval_mask"] == 1.0].sort_values(["well", "row_idx"]).reset_index(drop=True)
    df["rank_in_well"] = df.groupby("well", observed=True).cumcount().astype("int32")
    return df[["well", "row_idx", "rank_in_well", "fold", "tvt_pred", "oof_tvt", "heu026", "tvt_true"]]


@st.cache_data(show_spinner=False)
def well_summary(df: pd.DataFrame) -> pd.DataFrame:
    """well ごとの行数と各モデルの RMSE 一覧 (well 選択の参考用)。"""
    rows = []
    for w, sub in df.groupby("well", observed=True):
        yt = sub["tvt_true"].to_numpy()
        lgb = sub["oof_tvt"].to_numpy()
        heu = sub["heu026"].to_numpy()
        rmse_lgb = rmse(yt, lgb)
        rmse_heu = rmse(yt, heu)
        # heuristic と GBDT(LGB) の差分
        valid = np.isfinite(heu) & np.isfinite(lgb)
        diff_pred = float(np.mean(np.abs(heu[valid] - lgb[valid]))) if valid.any() else float("nan")
        rows.append({
            "well": w,
            "n_rows": len(sub),
            "rmse_NN": rmse(yt, sub["tvt_pred"].to_numpy()),
            "rmse_LGB": rmse_lgb,
            "rmse_HEU026": rmse_heu,
            "diff_pred": diff_pred,            # |heu026 - LGB| の平均 (予測の乖離)
            "diff_rmse": rmse_heu - rmse_lgb,  # RMSE差 (正: heuristic が劣る / 負: heuristic が優れる)
        })
    return pd.DataFrame(rows).set_index("well")


@st.cache_data(show_spinner=False)
def load_known_tvt(well: str, pred_start: int) -> pd.DataFrame | None:
    """予測区間以前 (既知区間) の実測 TVT を配布 horizontal_well.csv から取得。

    row_idx は horizontal_well.csv の行インデックスと一致する。予測開始 (pred_start)
    より前の行が既知区間。`rank_in_well` は予測開始を 0 とするので既知区間は負になる。
    """
    path = TRAIN_DIR / f"{well}__horizontal_well.csv"
    if not path.exists():
        return None
    h = pd.read_csv(path, usecols=["TVT"])
    known = h.iloc[:pred_start].copy()
    if known.empty:
        return None
    known["row_idx"] = np.arange(len(known), dtype="int32")
    known["rank_in_well"] = known["row_idx"] - pred_start  # 予測開始を 0 とした相対位置 (負)
    return known.rename(columns={"TVT": "tvt_known"})[["row_idx", "rank_in_well", "tvt_known"]]


def plot_well(
    sub: pd.DataFrame,
    w_nn: float,
    x_col: str,
    show: dict[str, bool],
    known: pd.DataFrame | None = None,
    split_x: float | None = None,
) -> go.Figure:
    x = sub[x_col].to_numpy()
    blend = w_nn * sub["tvt_pred"].to_numpy() + (1 - w_nn) * sub["oof_tvt"].to_numpy()

    fig = go.Figure()
    if show.get("known") and known is not None:
        fig.add_trace(go.Scatter(
            x=known[x_col], y=known["tvt_known"], mode="lines", name="実測 (予測区間以前)",
            line=dict(color=COLOR_KNOWN, width=1.8),
            hovertemplate=f"{x_col}=%{{x}}<br>known=%{{y:.2f}}<extra></extra>",
        ))
    if show["true"]:
        fig.add_trace(go.Scatter(
            x=x, y=sub["tvt_true"], mode="lines", name="実測 (tvt_true)",
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
        xaxis_title=x_col,
        yaxis_title="TVT (ft)",
        yaxis=dict(autorange="reversed"),
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def plot_residual(sub: pd.DataFrame, w_nn: float, x_col: str, show: dict[str, bool]) -> go.Figure:
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
        xaxis_title=x_col,
        yaxis_title="残差 (ft)",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Prediction Viewer", layout="wide")
    st.title("実測 vs 予測 Viewer (per well)")
    st.caption("pred_exp010 (LSTM) / physics LGB / heuristic exp026 の OOF を well ごとに比較")

    if not PRED_EXP010.exists() or not OOF_PARQUET.exists():
        st.error("pred_exp010.csv または oof_predictions.parquet が見つかりません。")
        st.stop()

    df = load_data()
    summary = well_summary(df)
    wells = summary.index.tolist()

    with st.sidebar:
        st.header("設定")
        sort_by = st.selectbox(
            "well 並び順",
            ["well 名", "NN が苦手な順", "LGB が苦手な順", "heu026 が苦手な順",
             "heu026↔LGB 予測差が大きい順", "heu026 が LGB より良い順", "LGB が heu026 より良い順",
             "行数が多い順"],
        )
        if sort_by == "NN が苦手な順":
            ordered = summary.sort_values("rmse_NN", ascending=False).index.tolist()
        elif sort_by == "LGB が苦手な順":
            ordered = summary.sort_values("rmse_LGB", ascending=False).index.tolist()
        elif sort_by == "heu026 が苦手な順":
            ordered = summary.sort_values("rmse_HEU026", ascending=False).index.tolist()
        elif sort_by == "heu026↔LGB 予測差が大きい順":
            # 2 モデルの予測が最も食い違う well 順 (|heu026 - LGB| の平均)
            ordered = summary.sort_values("diff_pred", ascending=False).index.tolist()
        elif sort_by == "heu026 が LGB より良い順":
            # diff_rmse = RMSE_heu - RMSE_LGB が小さい (負に大きい) ほど heuristic が優位
            ordered = summary.sort_values("diff_rmse", ascending=True).index.tolist()
        elif sort_by == "LGB が heu026 より良い順":
            ordered = summary.sort_values("diff_rmse", ascending=False).index.tolist()
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

    # 予測区間以前 (既知区間) の実測 TVT
    pred_start = int(sub["row_idx"].min())
    known = load_known_tvt(well, pred_start)
    split_x = 0.0 if x_col == "rank_in_well" else float(pred_start)

    rmse_lgb = rmse(yt, sub["oof_tvt"].to_numpy())
    rmse_heu = rmse(yt, sub["heu026"].to_numpy())
    heu_arr = sub["heu026"].to_numpy()
    lgb_arr = sub["oof_tvt"].to_numpy()
    vmask = np.isfinite(heu_arr) & np.isfinite(lgb_arr)
    diff_pred = float(np.mean(np.abs(heu_arr[vmask] - lgb_arr[vmask]))) if vmask.any() else float("nan")

    st.subheader(f"well: `{well}`")
    c = st.columns(7)
    c[0].metric("予測区間 行数", f"{len(sub):,}")
    c[1].metric("RMSE NN", f"{rmse(yt, sub['tvt_pred'].to_numpy()):.3f}")
    c[2].metric("RMSE LGB", f"{rmse_lgb:.3f}")
    c[3].metric("RMSE heu026", f"{rmse_heu:.3f}")
    c[4].metric(f"RMSE blend ({w_nn:.1f})", f"{rmse(yt, blend):.3f}")
    c[5].metric("heu026↔LGB 予測差", f"{diff_pred:.3f}",
                help="|heu026 − LGB| の平均 (予測の乖離)")
    c[6].metric("RMSE差 (heu026−LGB)", f"{rmse_heu - rmse_lgb:+.3f}",
                help="負: heuristic が優位 / 正: LGB が優位")

    st.plotly_chart(plot_well(sub, w_nn, x_col, show, known=known, split_x=split_x), width='stretch')
    if show_resid:
        st.plotly_chart(plot_residual(sub, w_nn, x_col, show), width='stretch')

    # 配布 PNG (該当 well の断面図) を TVT グラフの下に掲載
    png = TRAIN_DIR / f"{well}.png"
    if png.exists():
        st.image(str(png), caption=png.name, width='stretch')
    else:
        st.info(f"{png.name} は存在しません")

    with st.expander("well 別 RMSE 一覧 (全 well)"):
        st.dataframe(summary.round(3), width='stretch')

    with st.expander("この well の生データ"):
        st.dataframe(sub, width='stretch')


if __name__ == "__main__":
    main()
