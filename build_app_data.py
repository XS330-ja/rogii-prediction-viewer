"""Streamlit Cloud 用の軽量データを生成する。

入力 (ローカルのみ・リポジトリには含めない):
    pred_exp010.csv, models/.../oof_predictions.parquet, train/*__horizontal_well.csv
出力 (リポジトリ同梱・Cloud アプリが読む):
    app_data/predictions.parquet … 予測区間の well/row_idx/rank/fold/tvt_pred/oof_tvt/tvt_true
    app_data/known_tvt.parquet    … 予測区間以前 (既知区間) の実測 TVT
    app_data/well_summary.parquet … well 別 RMSE 一覧 (起動高速化用に事前計算)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PRED_EXP010 = DATA_DIR / "pred_exp010.csv"
OOF_PARQUET = DATA_DIR / "models" / "physics-informed-baseline" / "artefacts" / "oof_predictions.parquet"
HEURISTIC026 = DATA_DIR / "public_notebook_heuristic_exp026.csv"
TRAIN_DIR = DATA_DIR / "train"
OUT_DIR = DATA_DIR / "app_data"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("読み込み: pred_exp010.csv …")
    pred = pd.read_csv(
        PRED_EXP010,
        usecols=["well", "row_idx", "id", "tvt_pred", "fold", "eval_mask", "tvt_true"],
        dtype={"well": "category", "row_idx": "int32", "tvt_pred": "float32",
               "tvt_true": "float32", "fold": "int8"},
    )
    oof = pd.read_parquet(OOF_PARQUET, columns=["id", "target", "oof_lgb1"])

    df = pred.merge(oof, on="id", how="inner")
    df["oof_tvt"] = (df["oof_lgb1"] + (df["tvt_true"] - df["target"])).astype("float32")

    # heuristic exp026 を追加 (id を持たないため row_idx を復元して結合)
    print("読み込み: public_notebook_heuristic_exp026.csv …")
    heu = pd.read_csv(HEURISTIC026, usecols=["well", "MD", "tvt_pred"]).rename(columns={"tvt_pred": "heu026"})
    ps_map = {str(k): int(v) for k, v in pred.groupby("well", observed=True)["row_idx"].min().items()}
    heu = heu.sort_values(["well", "MD"])
    heu["row_idx"] = heu.groupby("well").cumcount() + heu["well"].map(ps_map).astype("int64")
    heu["id"] = heu["well"] + "_" + heu["row_idx"].astype(str)
    df = df.merge(heu[["id", "heu026"]], on="id", how="left")
    df["heu026"] = df["heu026"].astype("float32")

    df = df[df["eval_mask"] == 1.0].sort_values(["well", "row_idx"]).reset_index(drop=True)
    df["rank_in_well"] = df.groupby("well", observed=True).cumcount().astype("int32")
    pred_df = df[["well", "row_idx", "rank_in_well", "fold", "tvt_pred", "oof_tvt", "heu026", "tvt_true"]].copy()
    pred_df["well"] = pred_df["well"].astype("category")
    pred_df.to_parquet(OUT_DIR / "predictions.parquet", compression="zstd", index=False)
    print(f"  ✔ predictions.parquet  rows={len(pred_df):,}")

    # well 別 予測開始位置 (= row_idx の最小) と RMSE 一覧
    pred_start = pred_df.groupby("well", observed=True)["row_idx"].min()

    print("読み込み: train/*__horizontal_well.csv で既知区間 TVT を抽出 …")
    known_parts = []
    wells = list(pred_start.index)
    for i, well in enumerate(wells):
        ps = int(pred_start[well])
        path = TRAIN_DIR / f"{well}__horizontal_well.csv"
        if ps <= 0 or not path.exists():
            continue
        tvt = pd.read_csv(path, usecols=["TVT"])["TVT"].to_numpy()[:ps]
        n = len(tvt)
        part = pd.DataFrame({
            "well": np.repeat(well, n),
            "row_idx": np.arange(n, dtype="int32"),
            "rank_in_well": (np.arange(n, dtype="int32") - ps),
            "tvt_known": tvt.astype("float32"),
        })
        known_parts.append(part)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(wells)} wells")
    known_df = pd.concat(known_parts, ignore_index=True)
    known_df["well"] = known_df["well"].astype("category")
    known_df.to_parquet(OUT_DIR / "known_tvt.parquet", compression="zstd", index=False)
    print(f"  ✔ known_tvt.parquet  rows={len(known_df):,}")

    print("well 別サマリを計算 …")
    rows = []
    for w, sub in pred_df.groupby("well", observed=True):
        yt = sub["tvt_true"].to_numpy()
        rows.append({
            "well": w,
            "n_rows": len(sub),
            "n_known": int(pred_start[w]),
            "fold": int(sub["fold"].iloc[0]),
            "rmse_NN": rmse(yt, sub["tvt_pred"].to_numpy()),
            "rmse_LGB": rmse(yt, sub["oof_tvt"].to_numpy()),
            "rmse_HEU026": rmse(yt, sub["heu026"].to_numpy()),
        })
    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT_DIR / "well_summary.parquet", compression="zstd", index=False)
    print(f"  ✔ well_summary.parquet  wells={len(summary):,}")

    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    print(f"\napp_data 合計サイズ: {total / 1e6:.1f} MB")
    for p in sorted(OUT_DIR.glob("*.parquet")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
