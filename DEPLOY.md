# Streamlit Cloud デプロイ手順

well ごとに実測 TVT と予測 (NN / physics LGB / blend) を比較するアプリを
Streamlit Community Cloud に公開し、スマートフォンからも閲覧できるようにする。

## 構成 (リポジトリに含めるもの)

| ファイル | 役割 |
|---|---|
| `streamlit_app.py` | Cloud 用アプリ本体 (`app_data/*.parquet` を読む) |
| `app_data/predictions.parquet` | 予測区間の予測値・実測値 (~44MB) |
| `app_data/known_tvt.parquet` | 予測区間以前の実測 TVT (~5MB) |
| `app_data/well_summary.parquet` | well 別 RMSE 一覧 |
| `requirements.txt` | 依存ライブラリ |
| `.streamlit/config.toml` | テーマ等 |

> 生データ (`pred_exp010.csv`, `train/`, `models/` など合計数GB) は `.gitignore` で除外。
> PNG はサイズ制約のため Cloud 版では非表示。
> データを作り直すときは `python build_app_data.py` を実行 (ローカルに生データが必要)。

## 1. GitHub リポジトリへ push

このディレクトリで git は初期化済み・初回コミット済み。あなたの GitHub に空のリポジトリを
作成し、リモートを追加して push する:

```bash
# 例: GitHub で "rogii-prediction-viewer" という空リポジトリを作成後
git remote add origin https://github.com/<あなたのユーザー名>/rogii-prediction-viewer.git
git branch -M main
git push -u origin main
```

`app_data/predictions.parquet` は ~44MB で GitHub の 100MB 制限内なので LFS 不要。

## 2. Streamlit Community Cloud で起動

1. https://share.streamlit.io にアクセスし、GitHub アカウントでサインイン
2. 「Create app」→「Deploy a public app from GitHub」
3. 次を指定:
   - Repository: `<あなたのユーザー名>/rogii-prediction-viewer`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. 「Deploy」を押す。数分でビルドが完了し、`https://<app-name>.streamlit.app` の URL が発行される

## 3. スマートフォンから閲覧

発行された `*.streamlit.app` の URL をスマホのブラウザで開けば閲覧可能。
レイアウトはレスポンシブで、サイドバーは左上の「>」から開閉できる。

## ローカルでの動作確認

```bash
streamlit run streamlit_app.py
```
