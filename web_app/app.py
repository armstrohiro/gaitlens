from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import streamlit as st
import torch

from pose_pipeline import available_models, analyze_video

JOINT_JA = {
    "nose":"鼻", "left_eye":"左目", "right_eye":"右目", "left_ear":"左耳", "right_ear":"右耳",
    "left_eye_inner":"左目（内側）", "left_eye_outer":"左目（外側）", "right_eye_inner":"右目（内側）", "right_eye_outer":"右目（外側）",
    "mouth_left":"口（左）", "mouth_right":"口（右）", "left_shoulder":"左肩", "right_shoulder":"右肩",
    "left_elbow":"左肘", "right_elbow":"右肘", "left_wrist":"左手首", "right_wrist":"右手首",
    "left_pinky":"左小指", "right_pinky":"右小指", "left_index":"左人差し指", "right_index":"右人差し指",
    "left_thumb":"左親指", "right_thumb":"右親指", "left_hip":"左股関節", "right_hip":"右股関節",
    "left_knee":"左膝", "right_knee":"右膝", "left_ankle":"左足首", "right_ankle":"右足首",
    "left_heel":"左かかと", "right_heel":"右かかと", "left_big_toe":"左母趾", "right_big_toe":"右母趾",
    "left_small_toe":"左小趾", "right_small_toe":"右小趾", "left_foot_index":"左足先（foot_index）", "right_foot_index":"右足先（foot_index）",
}

st.set_page_config(page_title="GaitLens | 歩行姿勢解析", page_icon="🚶", layout="wide")
st.markdown("""<style>
.stApp{background:#f4f7f5}.hero{padding:1.7rem 2rem;border-radius:22px;background:linear-gradient(125deg,#123d37,#1d6b5b);color:white;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.25rem}.hero p{margin:.5rem 0 0;color:#d9f4eb}.metric-note{font-size:.85rem;color:#52645e}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>GaitLens</h1><p>動画から姿勢を推定し、見える品質と欠損を確認する研究用Webアプリ</p></div>',unsafe_allow_html=True)

models=available_models(); ready=[k for k,v in models.items() if v["ready"]]
with st.sidebar:
    st.header("解析設定")
    model_key=st.selectbox("モデル",list(models),format_func=lambda k:("✅ " if models[k]["ready"] else "🛠️ ")+models[k]["label"])
    st.caption("解析対象：標準YOLOは共通12関節、MediaPipeと追加学習YOLOは16関節。足先の対応は研究上の便宜的な対応です。")
    threshold=st.slider("点を有効とする信頼度",0.1,0.9,0.5,.05,help="研究条件を変える設定です。比較時は同じ値を使用してください。")
    device=st.selectbox("実行デバイス",["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],help="利用可能な実行デバイスを表示します。")
    st.caption("学会デモ版。診断・転倒リスク判定には使用できません。")

left,right=st.columns([.95,1.25],gap="large")
with left:
    st.subheader("1. 動画を選ぶ／撮影する")
    uploaded=st.file_uploader("MP4・MOV・AVI",type=["mp4","mov","avi","m4v"],help="スマートフォンではファイル選択画面から動画撮影もできます。")
    st.info("頭から足先まで写し、できるだけ一人で撮影してください。顔は出力動画でモザイク処理します。")
    if uploaded: st.video(uploaded)
    if not models[model_key]["ready"]: st.warning("このモデルは準備中、または必要な重みが見つかりません。")
    run=st.button("解析を開始",type="primary",use_container_width=True,disabled=not(uploaded and models[model_key]["ready"]))

with right:
    st.subheader("2. 解析結果")
    if not run: st.caption("動画とモデルを選び、「解析を開始」を押してください。")
    else:
        work=Path(tempfile.mkdtemp(prefix="gaitlens_")); suffix=Path(uploaded.name).suffix or ".mp4"; source=work/("input"+suffix)
        source.write_bytes(uploaded.getbuffer()); bar=st.progress(0,text="モデルを準備しています…")
        try:
            result=analyze_video(source,work,model_key,threshold,device,lambda v,t:bar.progress(v,text=t));bar.progress(1.0,text="解析が完了しました")
            st.video(str(result.video_path));m=result.metrics
            a,b,c,d=st.columns(4);a.metric("出力率",f'{m["available_percent"]}%');b.metric("欠損があった関節数",f'{m["joints_with_missing"]} / {m["joint_count"]}');c.metric("平均信頼度",m["mean_confidence"]);d.metric("処理速度",f'{m["processing_fps"]} fps')
            st.caption(f'欠損があった関節数は、動画中に1フレーム以上欠損した関節の種類の数です。欠損の延べ数は {m["missing_points"]:,} 関節フレームです。')
            with st.expander("関節ごとの欠損数を見る"):
                st.caption(f'各関節について、全 {m["frames"]:,} フレーム中の欠損フレーム数です。座標なし、または信頼度が設定値未満の場合を欠損とします。')
                rows=[{"関節":JOINT_JA.get(name,name),"欠損フレーム数":count,"欠損率":f'{count/m["frames"]*100:.1f}%'} for name,count in m["missing_by_joint"].items()]
                st.dataframe(rows,hide_index=True,use_container_width=True)
            a,b,c=st.columns(3);a.metric("参照なし品質（試作）",f'{m["reference_free_quality"]}/100');b.metric("時間安定性（参考）",f'{m["temporal_stability"]}/100');c.metric("骨格長一貫性（参考）",f'{m["bone_length_consistency"]}/100')
            st.warning("参照なし品質は独自の試作スコアです。配点と正解率との関係は未検証で、モデルの精度を示すものではありません。")
            with st.expander("参照なし品質の計算方法"):
                st.markdown("""
**総合点 = 出力率 × 0.40 + 平均信頼度 × 100 × 0.30 + 時間安定性 × 0.15 + 骨格長一貫性 × 0.15**

すべて100点換算後に合計します。計算には丸める前の値を使うため、表示値から計算した結果と0.1点程度異なる場合があります。

- **出力率**：解析対象の全関節×全フレームのうち、有効な座標がある割合。
- **平均信頼度**：有効と判定された点の信頼度の平均。モデルごとに意味が異なります。
- **時間安定性**：各フレームの有効点の平均位置を人物中心とし、その中心の連続3フレームの2階差分の大きさを、人物の画面上の大きさの中央値で割った値を使います。値を `100 × exp(-8 × 正規化した2階差分の中央値)` で100点換算します。
- **骨格長一貫性**：両端が有効な骨格線ごとに、画面上の長さの標準偏差を平均で割り、その平均を求めます。`100 × (1 − min(平均変動係数, 1))` で100点換算します。各線分は3フレーム以上必要です。

時間安定性や骨格長一貫性を計算できない場合は0点として扱います。動き、遠近、遮蔽、有効点の変化によっても数値は変わります。40・30・15・15%の配点は本アプリの仮の設定であり、先行研究から検証済みの係数として採用したものではありません。
""")
            for warning in result.warnings: st.warning(warning)
            video_bytes=result.video_path.read_bytes();csv_bytes=result.csv_path.read_bytes();metrics_bytes=json.dumps(m,ensure_ascii=False,indent=2).encode("utf-8")
            x,y,z=st.columns(3);x.download_button("骨格動画を保存",video_bytes,"gaitlens_result.mp4","video/mp4",use_container_width=True,on_click="ignore");y.download_button("座標CSVを保存",csv_bytes,"gaitlens_coordinates.csv","text/csv",use_container_width=True,on_click="ignore");z.download_button("品質JSONを保存",metrics_bytes,"gaitlens_quality.json","application/json",use_container_width=True,on_click="ignore")
        except Exception as exc: st.error(f"解析できませんでした: {exc}")

with st.expander("指標の意味と限界"):
    st.markdown("""
- **出力率／欠損率**：設定した信頼度以上で座標が得られた点の割合です。正解率ではありません。
- **平均信頼度**：モデル内部の出力値です。異なるモデル間で同じ尺度とは限りません。
- **時間安定性**：人物中心のフレーム間加速度が小さいほど高くなる参照なし指標です。実際の急な動きも低値になる場合があります。
- **骨格長一貫性**：腕・脚の線分長の変動が小さいほど高くなる参照なし指標です。遠近や遮蔽の影響を受けます。
- **参照なし品質**：出力率、平均信頼度、時間安定性、骨格長一貫性の仮の重み付き合計です。正解率ではなく、配点の妥当性は未検証です。
""")


