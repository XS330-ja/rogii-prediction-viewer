"""Cloud アプリ用に、配布 PNG 相当の断面図を plotly で再現するための地質データを生成する。

出力 (app_data/ に同梱):
    well_geo.parquet       … 横坑 (downsample): well, md, gr, hdist, z, 地層面6本, tvt, is_known
    typewell.parquet       … タイプ坑: well, tvt, gr
    formation_tops.parquet … 地層トップ: well, formation, tvt (タイプ坑 Geology の最小 TVT)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
TRAIN_DIR = DATA_DIR / "train"
OUT_DIR = DATA_DIR / "app_data"
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
STEP = 5  # 横坑の間引き (GR ログの概形は保ちつつ容量を抑える)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    wells = sorted(p.name.split("__")[0] for p in TRAIN_DIR.glob("*__horizontal_well.csv"))
    print(f"wells: {len(wells)}")

    geo_parts, tw_parts, top_parts = [], [], []
    for i, well in enumerate(wells):
        h = pd.read_csv(TRAIN_DIR / f"{well}__horizontal_well.csv")
        x0, y0 = h["X"].iloc[0], h["Y"].iloc[0]
        h["hdist"] = np.hypot(h["X"] - x0, h["Y"] - y0)
        # 予測開始 index (TVT_input が最初に NaN)
        miss = h["TVT_input"].isna().to_numpy()
        pred_start = int(miss.argmax()) if miss.any() else len(h)
        h["is_known"] = h["TVT_input"].notna()

        # 間引き (予測開始行は必ず残す)
        keep = np.zeros(len(h), dtype=bool)
        keep[::STEP] = True
        if pred_start < len(h):
            keep[pred_start] = True
        sub = h.loc[keep, ["MD", "GR", "hdist", "Z", *FORMATIONS, "TVT", "is_known"]].copy()
        sub.insert(0, "well", well)
        sub = sub.rename(columns={"MD": "md", "GR": "gr", "Z": "z", "TVT": "tvt"})
        geo_parts.append(sub)

        # typewell
        t = pd.read_csv(TRAIN_DIR / f"{well}__typewell.csv")
        tw = t[["TVT", "GR"]].rename(columns={"TVT": "tvt", "GR": "gr"}).copy()
        tw.insert(0, "well", well)
        tw_parts.append(tw)

        # formation tops (6 main formations only)
        g = t.dropna(subset=["Geology"])
        tops = g[g["Geology"].isin(FORMATIONS)].groupby("Geology")["TVT"].min()
        for f in FORMATIONS:
            if f in tops.index:
                top_parts.append({"well": well, "formation": f, "tvt": float(tops[f])})

        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(wells)}")

    geo = pd.concat(geo_parts, ignore_index=True)
    for c in ["gr", "hdist", "z", "tvt", *FORMATIONS]:
        geo[c] = geo[c].astype("float32")
    geo["md"] = geo["md"].astype("float32")
    geo["well"] = geo["well"].astype("category")
    geo.to_parquet(OUT_DIR / "well_geo.parquet", compression="zstd", index=False)

    tw = pd.concat(tw_parts, ignore_index=True)
    tw["tvt"] = tw["tvt"].astype("float32"); tw["gr"] = tw["gr"].astype("float32")
    tw["well"] = tw["well"].astype("category")
    tw.to_parquet(OUT_DIR / "typewell.parquet", compression="zstd", index=False)

    tops_df = pd.DataFrame(top_parts)
    tops_df["well"] = tops_df["well"].astype("category")
    tops_df["tvt"] = tops_df["tvt"].astype("float32")
    tops_df.to_parquet(OUT_DIR / "formation_tops.parquet", compression="zstd", index=False)

    print(f"\nwell_geo rows={len(geo):,}  typewell rows={len(tw):,}  tops rows={len(tops_df):,}")
    for name in ["well_geo.parquet", "typewell.parquet", "formation_tops.parquet"]:
        print(f"  {name}: {(OUT_DIR / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
