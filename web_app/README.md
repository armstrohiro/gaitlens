# GaitLens 学会デモ用Webアプリ

動画をアップロード（スマートフォンではファイル選択から撮影可能）し、姿勢推定、顔モザイク、骨格動画、座標CSV、欠損と参照なし品質指標を出力するStreamlitアプリです。

## モデル

- 標準 YOLO11n-Pose（解析対象12点）
- 標準 MediaPipe Pose（解析対象16点）
- 提案手法：追加学習 YOLO（解析対象16点）
- 提案手法：追加学習 MediaPipe（画面には表示するが、学習完了まで実行不可）

追加学習YOLOは `results/external_video_model_comparison_20261028_v1/checkpoint/tug_all_frames_epoch5_best.pt` と、基盤重み `../pilot_method_comparison_2026-05-21/models/yolo26n-pose.pt` を使用します。学習済みMediaPipeの実運用形式が確定したら、同じアダプター方式で追加します。

## 別のPCやエッジデバイスへの配置

アプリのコードだけをコピーし、`models/` に利用する重みを別途配置します。重み・動画・仮想環境はGitHubに含めません。

| ファイル名 | 必要なモデル |
| --- | --- |
| `models/yolo11n-pose.pt` | 標準YOLO |
| `models/pose_landmarker.task` | 標準MediaPipe |
| `models/yolo26n-pose.pt` | 追加学習YOLOの基盤 |
| `models/tug_all_frames_epoch5_best.pt` | 追加学習YOLOのチェックポイント |

ファイルが存在するモデルのみ選択できます。重みは信頼できる入手元から配置し、配布権限・ライセンスを確認してください。追加学習チェックポイントのSHA-256は `fba2048ebea19e372d1afae0e0a5283e8912029804e6d865ee5b308eaefe4a93` です。`GAITLENS_YOLO_STANDARD`、`GAITLENS_MEDIAPIPE`、`GAITLENS_YOLO_BASE`、`GAITLENS_YOLO_PROPOSED` 環境変数で、それぞれ別のファイルパスも指定できます。

Python依存パッケージが端末のOS・CPUアーキテクチャに対応している必要があります。現在はWindows CPUで動作確認済みで、iPad、Raspberry Pi、Jetsonなどでの動作は未確認です。iPadのブラウザ単体でPythonサーバーを動かす構成ではありません。端末に合わせた環境構築・高速化が別途必要です。

## ローカル起動

PowerShellでこのフォルダへ移動し、推論環境に依存関係を入れて起動します。

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

同じWi-Fi内では `http://PCのIPアドレス:8501` へアクセスできます。外部公開URLは、GPU PC上の8501番ポートをCloudflare Tunnel等でHTTPS公開します。アプリ自体はログインを要求しません。患者・参加者動画を扱う場合、無制限の一般公開URLは避け、アクセス制限、保存期間、同意、通信経路を研究倫理・所属機関の規則に合わせてください。

## 発表前チェック

1. CUDA対応の推論環境で3つの使用可能モデルを各1本ずつ実行する。
2. 出力MP4が発表用ブラウザで再生できない場合は、FFmpegでH.264へ変換する。
3. iPad/スマートフォンから短いMP4/MOVをアップロードする。
4. 顔モザイク、人物選択、全身が外れた場面、複数人物の挙動を目視確認する。
5. 当日は短い匿名サンプル動画と事前解析済み結果をバックアップとして用意する。

## 研究上の注意

参照なし品質は正解率ではありません。疑似教師、人手GT、マーカー式計測との距離を計算していません。モデル間の信頼度も直接比較できるとは限らないため、学会では「動画内の品質確認指標」と説明してください。

