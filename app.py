import hashlib
import io
import math
import subprocess
import tempfile
import threading
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from phonemizer import phonemize
from phonemizer.separator import Separator
from scipy.signal import resample_poly
from transformers import AutoModelForCTC, AutoProcessor


st.set_page_config(
    page_title="日本語話者のための英語発音練習",
    page_icon="🎙️",
    layout="centered",
)

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

WORDS = dict([
    ("stop", "ストップ"), ("school", "スクール"),
    ("spoon", "スプーン"), ("street", "ストリート"),
    ("spring", "スプリング"), ("snack", "スナック"),
    ("skate", "スケート"), ("sports", "スポーツ"),
    ("speed", "スピード"), ("style", "スタイル"),
    ("desk", "デスク"), ("map", "マップ"),
    ("bus", "バス"), ("cake", "ケーキ"),
    ("bag", "バッグ"), ("bed", "ベッド"),
    ("cup", "カップ"), ("cap", "キャップ"),
    ("top", "トップ"), ("shop", "ショップ"),
    ("class", "クラス"), ("glass", "グラス"),
    ("club", "クラブ"), ("plan", "プラン"),
    ("blue", "ブルー"), ("free", "フリー"),
    ("fresh", "フレッシュ"), ("small", "スモール"),
    ("snake", "スネーク"), ("milk", "ミルク"),
])

CLUSTERS = {
    "stop": "st", "school": "sk", "spoon": "sp",
    "street": "stɹ", "spring": "spɹ", "snack": "sn",
    "skate": "sk", "sports": "sp", "speed": "sp",
    "style": "st", "class": "kl", "glass": "ɡl",
    "club": "kl", "plan": "pl", "blue": "bl",
    "free": "fɹ", "fresh": "fɹ",
    "small": "sm", "snake": "sn",
}

L_WORDS = {
    "school", "style", "class", "glass",
    "club", "plan", "blue", "small", "milk",
}
R_WORDS = {"street", "spring", "free", "fresh"}


# 画面表示 / 화면 표시
language = st.selectbox(
    "안내 언어 / 表示・説明の言語",
    ["한국어", "日本語"],
)
JA = language == "日本語"


def tr(ko, ja):
    return ja if JA else ko


st.title(tr(
    "일본어 사용자를 위한 영어 발음 연습",
    "日本語話者のための英語発音練習",
))
st.write(tr(
    "익숙한 일본어 소리를 출발점으로 영어 발음을 연습해 봐. "
    "안내 언어를 바꿔도 듣고 녹음하는 언어는 영어야.",
    "なじみのある日本語の音を出発点に英語を練習しましょう。"
    "表示言語を変更しても、聞く音声と録音する言語は英語です。",
))
st.caption(tr(
    "단어 선택 → 기준 음성 듣기 → 직접 녹음 → 종료 후 자동 분석",
    "単語を選ぶ → 基準音声を聞く → 録音 → 終了後に自動分析",
))


# テキストや基準音声だけを共有キャッシュ。
# ユーザーの録音は共有キャッシュに保存しない。
@st.cache_data(show_spinner=False)
def reference(word):
    if word not in WORDS:
        raise ValueError("Unsupported word")

    phones = phonemize(
        word,
        language="en-us",
        backend="espeak",
        separator=Separator(
            phone=" ", word=" | ", syllable=""
        ),
        strip=True,
        with_stress=False,
        njobs=1,
    ).strip()

    with tempfile.TemporaryDirectory() as folder:
        wav = Path(folder) / "reference.wav"
        subprocess.run(
            [
                "espeak-ng", "-v", "en-us", "-s", "130",
                "-w", str(wav), word,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        audio_bytes = wav.read_bytes()

    return phones, audio_bytes


@st.cache_resource(show_spinner=False)
def load_model():
    torch.set_num_threads(2)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).eval()

    return processor, model, threading.Lock()


def normalize(phone):
    return unicodedata.normalize(
        "NFC", phone
    ).replace("g", "ɡ")


def tokenize(text):
    return [
        normalize(p) for p in text.split()
        if p not in {"|", "<pad>", "<s>", "</s>"}
    ]


def align(target, observed):
    n, m = len(target), len(observed)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1]
                + (target[i - 1] != observed[j - 1]),
            )

    result = []
    i, j = n, m

    while i or j:
        if (
            i and j
            and dp[i][j] == dp[i - 1][j - 1]
            + (target[i - 1] != observed[j - 1])
        ):
            kind = (
                "same" if target[i - 1] == observed[j - 1]
                else "replace"
            )
            result.append(
                (target[i - 1], observed[j - 1], kind)
            )
            i -= 1
            j -= 1

        elif i and dp[i][j] == dp[i - 1][j] + 1:
            result.append(
                (target[i - 1], "—", "missing")
            )
            i -= 1

        else:
            result.append(
                ("—", observed[j - 1], "extra")
            )
            j -= 1

    return dp[n][m], result[::-1]


def analyze(audio_bytes, target_text):
    audio, sr = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=True,
    )
    audio = audio.mean(axis=1)

    duration = len(audio) / sr
    if not 0.3 <= duration <= 10:
        raise ValueError(
            "0.3~10초로 단어를 한 번 녹음해 주세요. / "
            "0.3〜10秒で単語を1回録音してください。"
        )

    if not np.isfinite(audio).all():
        raise ValueError("Invalid audio")

    if np.sqrt(np.mean(audio ** 2)) < 0.0001:
        raise ValueError(
            "녹음 소리가 너무 작아요. / 録音の音量が小さすぎます。"
        )

    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(
            audio,
            16000 // divisor,
            sr // divisor,
        ).astype(np.float32)

    processor, model, lock = load_model()

    with lock, torch.inference_mode():
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        ids = logits.argmax(dim=-1)
        observed_text = processor.batch_decode(ids)[0].strip()

    a = tokenize(target_text)
    b = tokenize(observed_text)

    if not a or not b:
        raise ValueError(
            "음소를 인식하지 못했어요. 다시 녹음해 주세요. / "
            "音素を認識できませんでした。録音し直してください。"
        )

    distance, alignment = align(a, b)
    score = round(
        100 * max(0, 1 - distance / max(len(a), len(b))),
        1,
    )

    return {
        "score": score,
        "observed": observed_text,
        "alignment": alignment,
    }


def practice_tips(word, target):
    tips = []

    if word == "stop":
        tips.append(tr(
            "ストップ의 ス에서 ‘우’를 빼고 /s/에서 /t/로 연결해 봐. "
            "끝의 プ에서도 ‘우’를 붙이지 않고 /p/로 끝내 봐.",
            "ストップの「ス」の母音を加えず /s/ から /t/ へつなぎ、"
            "最後の「プ」も「ウ」を加えず /p/ で終えてみましょう。",
        ))
    elif word in CLUSTERS:
        cluster = CLUSTERS[word]
        tips.append(tr(
            f"처음의 /{cluster}/는 자음 사이에 모음을 넣지 않고 연결해 봐.",
            f"語頭の /{cluster}/ は、子音の間に母音を加えずにつなげましょう。",
        ))

    phones = tokenize(target)

    if word != "stop" and phones:
        final = phones[-1]
        if final in {
            "p", "b", "t", "d", "k", "ɡ",
            "s", "z", "ʃ", "f", "v", "l", "ŋ",
        }:
            tips.append(tr(
                f"끝의 /{final}/ 뒤에 ‘우’나 ‘오’ 같은 모음을 덧붙이지 않도록 해 봐.",
                f"語末の /{final}/ の後に「ウ」「オ」などの母音を加えないようにしましょう。",
            ))

    if word in L_WORDS:
        tips.append(tr(
            "/l/은 일본어 라행과 구분해서 혀끝을 윗잇몸 부근에 대는 연습을 해 봐.",
            "/l/ はラ行の音と区別し、舌先を上の歯茎付近に当てましょう。",
        ))

    if word in R_WORDS:
        tips.append(tr(
            "/ɹ/에서는 일본어 라행처럼 혀끝을 튕기지 않도록 해 봐.",
            "/ɹ/ では日本語のラ行のように舌先を弾かずに練習しましょう。",
        ))

    tips.append(tr(
        "가운데 모음도 기준 음성을 듣고 비교해 봐. "
        "가타카나는 영어 발음을 완전히 표현하지는 못해.",
        "母音も基準音声と聞き比べましょう。"
        "カタカナは英語の音を完全に表すものではありません。",
    ))

    return tips


# ユーザーごとの記録 / 사용자별 기록
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "selection" not in st.session_state:
    st.session_state.selection = None
if "record_version" not in st.session_state:
    st.session_state.record_version = 0

word = st.selectbox(
    tr("연습할 영어 단어", "練習する英単語"),
    list(WORDS),
)

selection = (word, language)
if st.session_state.selection != selection:
    st.session_state.selection = selection
    st.session_state.last_result = None
    st.session_state.record_version += 1

try:
    target, sample = reference(word)
except Exception as error:
    st.error(tr(
        "기준 음성을 만들지 못했어요. 배포 로그를 확인해 주세요.",
        "基準音声を作成できませんでした。ログを確認してください。",
    ))
    st.exception(error)
    st.stop()

st.subheader(f"{word} ／ {WORDS[word]}")
st.caption(tr(
    "일반 일본어 외래어 표기이며 영어 발음과 같다는 뜻은 아니야.",
    "一般的な外来語表記であり、英語と同じ発音という意味ではありません。",
))

st.markdown(tr("**연습 방법**", "**練習方法**"))
for tip in practice_tips(word, target):
    st.markdown(f"- {tip}")

st.markdown(tr(
    "**영어 기준 발음 · 합성 음성**",
    "**英語の基準音声 · 合成音声**",
))
st.audio(sample, format="audio/wav")

recording = st.audio_input(
    tr(
        "영어로 녹음 · 종료 후 자동 분석",
        "英語で録音 · 終了後に自動分析",
    ),
    sample_rate=16000,
    key=f"recording_{st.session_state.record_version}",
)

st.caption(tr(
    "녹음 버튼으로 시작하고, 단어를 말한 뒤 종료해 줘. "
    "녹음은 분석을 위해 서버로 전송돼.",
    "録音ボタンで開始し、単語を話した後に終了してください。"
    "録音は分析のためサーバーに送信されます。",
))

if recording is not None:
    raw = recording.getvalue()
    key = word + ":" + hashlib.sha256(raw).hexdigest()
    last = st.session_state.last_result

    if last is None or last["key"] != key:
        with st.spinner(tr(
            "발음 분석 중… 첫 분석은 모델 다운로드로 시간이 걸릴 수 있어.",
            "発音を分析中… 初回はモデルの読み込みに時間がかかる場合があります。",
        )):
            try:
                result = analyze(raw, target)
                result["key"] = key
                result["word"] = word
                st.session_state.last_result = result

                if not any(
                    row["key"] == key
                    for row in st.session_state.history
                ):
                    st.session_state.history.append({
                        "key": key,
                        "word": word,
                        "score": result["score"],
                    })

            except Exception as error:
                st.error(tr(
                    "분석하지 못했어요. 녹음을 지우고 다시 시도해 주세요.",
                    "分析できませんでした。録音を削除して再試行してください。",
                ))
                st.exception(error)

result = st.session_state.last_result

if result is not None and result["word"] == word:
    st.metric(
        tr("음소 유사도 · 참고값", "音素の類似度 · 参考値"),
        f"{result['score']:.1f}%",
    )

    st.subheader(tr(
        "다음에는 이렇게 연습해 봐",
        "次の練習のヒント",
    ))
    st.markdown(tr(
        f"**연습 출발점: {WORDS[word]} → 영어 {word}**",
        f"**練習の出発点：{WORDS[word]} → 英語の {word}**",
    ))

    for tip in practice_tips(word, target):
        st.markdown(f"- {tip}")

    vowels = set("aeiouɑæɐɒɔəɛɜɞɪʊʌɯɤœøɚɝ")
    extras = [
        b for a, b, kind in result["alignment"]
        if kind == "extra" and any(c in vowels for c in b)
    ]
    different = [
        f"/{a}/ → /{b}/"
        for a, b, kind in result["alignment"]
        if kind == "replace"
    ]

    if extras:
        st.info(tr(
            "기준에 없는 모음이 추가로 인식됐어: " + ", ".join(extras),
            "基準にない母音が追加で認識されました：" + "、".join(extras),
        ))
    if different:
        st.write(tr(
            "다르게 인식된 소리(기준 → 인식): ",
            "異なって認識された音（基準 → 認識）：",
        ) + ", ".join(different[:4]))

    st.caption(tr(
        "인식 결과만으로 실제 오류를 단정할 수는 없어. "
        "다시 녹음해 소리와 유사도의 변화를 확인해 봐.",
        "認識結果だけで実際の誤りを断定できません。"
        "録音し直し、音と類似度の変化を確認しましょう。",
    ))

    with st.expander(tr(
        "소리별 비교 자세히 보기",
        "音ごとの比較を見る",
    )):
        st.write(tr("기준: ", "基準：") + target)
        st.write(tr("AI 인식: ", "AI認識：") + result["observed"])

        names = {
            "same": tr("일치", "一致"),
            "replace": tr("다르게 인식", "相違"),
            "missing": tr("미검출", "未検出"),
            "extra": tr("추가 인식", "追加検出"),
        }
        st.dataframe([
            {
                tr("기준 소리", "基準の音"): a,
                tr("인식한 소리", "認識された音"): b,
                tr("비교", "比較"): names[kind],
            }
            for a, b, kind in result["alignment"]
        ], hide_index=True)

st.divider()
st.subheader(tr("내 연습 기록", "自分の練習記録"))

history_rows = []
previous = {}

for number, row in enumerate(st.session_state.history, start=1):
    change = (
        f"{row['score'] - previous[row['word']]:+.1f} %p"
        if row["word"] in previous else "—"
    )
    history_rows.append({
        tr("회차", "回数"): number,
        tr("단어", "単語"): row["word"],
        tr("음소 유사도(%)", "音素の類似度(%)"): row["score"],
        tr("직전 같은 단어 대비", "同じ単語の前回との差"): change,
    })
    previous[row["word"]] = row["score"]

if history_rows:
    st.dataframe(history_rows, hide_index=True)
else:
    st.caption(tr(
        "녹음하면 기록이 여기에 쌓여.",
        "録音すると記録が表示されます。",
    ))

if st.button(tr("내 연습 기록 지우기", "自分の練習記録を消去")):
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.record_version += 1
    st.rerun()

st.caption(tr(
    "기록은 현재 접속 세션에만 유지돼. 새로고침이나 연결 종료 시 초기화될 수 있어.",
    "記録は現在のセッションでのみ保持され、更新や切断で失われる場合があります。",
))

st.warning(tr(
    "음소 유사도는 검증된 발음 정확도가 아니야. "
    "일본어 기반 안내는 규칙으로 작성했으며 최적 근사음으로 검증되지 않았어.",
    "音素の類似度は検証済みの発音評価ではありません。"
    "ガイドは規則に基づくもので、最適な近似音として検証されていません。",
))

with st.expander(tr("도구 소개·출처", "ツールについて・出典")):
    st.write(tr(
        "음성 분석: 공개 사전학습 Wav2Vec2 모델. "
        "기준 음성·음소: eSpeak NG 미국 영어. "
        "Qwen 및 LoRA는 이 앱에서 사용하지 않아.",
        "音声分析：公開の事前学習済みWav2Vec2モデル。"
        "基準音声・音素：eSpeak NGの米国英語。"
        "このアプリではQwenやLoRAを使用していません。",
    ))
    st.markdown(
        "- [Wav2Vec2 음소 모델]"
        "(https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)\n"
        "- [eSpeak NG](https://github.com/espeak-ng/espeak-ng)"
    )
