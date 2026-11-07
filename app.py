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


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="일본어 소리로 시작하는 영어 발음 연습",
    page_icon="🎙️",
    layout="centered",
)

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

WORDS = dict([
    ("stop", "ストップ"),
    ("school", "スクール"),
    ("spoon", "スプーン"),
    ("street", "ストリート"),
    ("spring", "スプリング"),
    ("snack", "スナック"),
    ("skate", "スケート"),
    ("sports", "スポーツ"),
    ("speed", "スピード"),
    ("style", "スタイル"),
    ("desk", "デスク"),
    ("map", "マップ"),
    ("bus", "バス"),
    ("cake", "ケーキ"),
    ("bag", "バッグ"),
    ("bed", "ベッド"),
    ("cup", "カップ"),
    ("cap", "キャップ"),
    ("top", "トップ"),
    ("shop", "ショップ"),
    ("class", "クラス"),
    ("glass", "グラス"),
    ("club", "クラブ"),
    ("plan", "プラン"),
    ("blue", "ブルー"),
    ("free", "フリー"),
    ("fresh", "フレッシュ"),
    ("small", "スモール"),
    ("snake", "スネーク"),
    ("milk", "ミルク"),
])

# 실제 음소를 나눠 저장해 자음군 위치 판정에 사용
CLUSTERS = {
    "stop": ("s", "t"),
    "school": ("s", "k"),
    "spoon": ("s", "p"),
    "street": ("s", "t", "ɹ"),
    "spring": ("s", "p", "ɹ"),
    "snack": ("s", "n"),
    "skate": ("s", "k"),
    "sports": ("s", "p"),
    "speed": ("s", "p"),
    "style": ("s", "t"),
    "class": ("k", "l"),
    "glass": ("ɡ", "l"),
    "club": ("k", "l"),
    "plan": ("p", "l"),
    "blue": ("b", "l"),
    "free": ("f", "ɹ"),
    "fresh": ("f", "ɹ"),
    "small": ("s", "m"),
    "snake": ("s", "n"),
}

L_WORDS = {
    "school", "style", "class", "glass",
    "club", "plan", "blue", "small", "milk",
}

R_WORDS = {"street", "spring", "free", "fresh"}

VOWELS = set("aeiouɑæɐɒɔəɛɜɞɪʊʌɯɤœøɚɝ")


# =========================================================
# 2. 사용자 상태
# =========================================================

DEFAULTS = {
    "practice_history": [],
    "last_analysis": None,
    "last_error": None,
    "last_attempt_key": None,
    "current_selection": None,
    "record_version": 0,
}

for name, value in DEFAULTS.items():
    if name not in st.session_state:
        st.session_state[name] = value


# =========================================================
# 3. 안내 언어
# =========================================================

language = st.selectbox(
    "안내 언어 / 表示・説明の言語",
    ["한국어", "日本語"],
    key="display_language",
)

JA = language == "日本語"


def tr(ko, ja):
    return ja if JA else ko


# =========================================================
# 4. 기준 음성·음소 생성
# =========================================================

@st.cache_data(show_spinner=False)
def reference(word):
    if word not in WORDS:
        raise ValueError("Unsupported word")

    phones = phonemize(
        word,
        language="en-us",
        backend="espeak",
        separator=Separator(
            phone=" ",
            word=" | ",
            syllable="",
        ),
        strip=True,
        with_stress=False,
        njobs=1,
    ).strip()

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "reference.wav"

        subprocess.run(
            [
                "espeak-ng",
                "-v", "en-us",
                "-s", "130",
                "-w", str(path),
                word,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

        audio_bytes = path.read_bytes()

    return phones, audio_bytes


# =========================================================
# 5. AI 음소 인식 모델
# =========================================================

@st.cache_resource(show_spinner=False)
def load_speech_model():
    torch.set_num_threads(2)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForCTC.from_pretrained(
        MODEL_ID,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).eval()

    return processor, model, threading.Lock()


def normalize_phone(phone):
    return unicodedata.normalize(
        "NFC", phone
    ).replace("g", "ɡ")


def phone_tokens(text):
    ignored = {"|", "<pad>", "<s>", "</s>"}

    return [
        normalize_phone(phone)
        for phone in text.split()
        if phone not in ignored
    ]


# =========================================================
# 6. 음소 정렬과 유사도
# =========================================================

def align_phones(target, observed):
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
                "same"
                if target[i - 1] == observed[j - 1]
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


def analyze_recording(audio_bytes, target_text):
    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
        always_2d=True,
    )

    audio = audio.mean(axis=1)

    duration = len(audio) / sample_rate

    if not 0.3 <= duration <= 10:
        raise ValueError(
            "0.3~10초로 영어 단어를 한 번 녹음해 주세요. / "
            "0.3〜10秒で英単語を1回録音してください。"
        )

    if not np.isfinite(audio).all():
        raise ValueError(
            "오디오 데이터를 확인해 주세요. / "
            "音声データを確認してください。"
        )

    if np.sqrt(np.mean(audio ** 2)) < 0.0001:
        raise ValueError(
            "녹음 소리가 너무 작아요. / "
            "録音の音量が小さすぎます。"
        )

    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)

        audio = resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

    processor, model, lock = load_speech_model()

    with lock, torch.inference_mode():
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )

        logits = model(**inputs).logits
        ids = logits.argmax(dim=-1)

        observed_text = processor.batch_decode(
            ids
        )[0].strip()

    target = phone_tokens(target_text)
    observed = phone_tokens(observed_text)

    if not target or not observed:
        raise ValueError(
            "소리를 인식하지 못했어요. 다시 녹음해 주세요. / "
            "音を認識できませんでした。録音し直してください。"
        )

    distance, alignment = align_phones(
        target, observed
    )

    score = round(
        100 * max(
            0,
            1 - distance / max(len(target), len(observed))
        ),
        1,
    )

    return {
        "score": score,
        "target": target_text,
        "observed": observed_text,
        "alignment": alignment,
    }


# =========================================================
# 7. 녹음 전 일반 연습 안내
# =========================================================

def general_tips(word, target_text):
    tips = []

    if word == "stop":
        tips.append(tr(
            "ストップ의 ス에서 ‘우’를 덧붙이지 않고 "
            "/s/에서 /t/로 연결해 봐. "
            "마지막에도 /p/ 뒤에 ‘우’를 붙이지 않도록 해 봐.",
            "ストップの「ス」に母音を加えず /s/ から /t/ へつなぎ、"
            "最後も /p/ の後に「ウ」を加えない練習をしましょう。",
        ))

    elif word in CLUSTERS:
        cluster = "".join(CLUSTERS[word])

        tips.append(tr(
            f"처음의 /{cluster}/를 자음 사이에 모음을 "
            "넣지 않고 연결해 봐.",
            f"語頭の /{cluster}/ を、子音の間に母音を"
            "加えずにつなげましょう。",
        ))

    target = phone_tokens(target_text)

    if target and target[-1] in {
        "p", "b", "t", "d", "k", "ɡ",
        "s", "z", "ʃ", "f", "v", "l", "ŋ",
    }:
        final = target[-1]

        tips.append(tr(
            f"단어 끝의 /{final}/ 뒤에 ‘우’나 ‘오’ 같은 "
            "모음을 덧붙이지 않도록 해 봐.",
            f"語末の /{final}/ の後に「ウ」「オ」などの"
            "母音を加えないようにしましょう。",
        ))

    if word in L_WORDS:
        tips.append(tr(
            "영어 /l/은 일본어 라행과 구분해 "
            "혀끝을 윗잇몸 부근에 대며 연습해 봐.",
            "英語の /l/ はラ行と区別し、"
            "舌先を上の歯茎付近に当てて練習しましょう。",
        ))

    if word in R_WORDS:
        tips.append(tr(
            "영어 /ɹ/에서는 일본어 라행처럼 "
            "혀끝을 튕기지 않도록 해 봐.",
            "英語の /ɹ/ では、日本語のラ行のように"
            "舌先を弾かずに練習しましょう。",
        ))

    return tips


# =========================================================
# 8. 인식 결과에 따른 연습용 가나 후보 선택
# =========================================================

def kana_recommendations(word, result):
    kana = WORDS[word]
    target = phone_tokens(result["target"])
    alignment = result["alignment"]

    candidates = []

    def add(label, reason, instruction):
        candidates.append({
            "label": label,
            "reason": reason,
            "instruction": instruction,
        })

    # 기준 음소의 위치를 따라가며 추가 모음 위치 확인
    position = 0
    initial_vowels = []
    internal_vowels = []
    final_vowels = []

    cluster_length = len(CLUSTERS.get(word, ()))

    for expected, observed, kind in alignment:
        if kind == "extra":
            is_vowel = any(
                char in VOWELS for char in observed
            )

            if is_vowel:
                if position == len(target):
                    final_vowels.append(observed)

                elif (
                    cluster_length > 1
                    and 0 < position < cluster_length
                ):
                    initial_vowels.append(observed)

                elif 0 < position < len(target):
                    # 기준의 두 자음 사이에 삽입된 모음인지 확인
                    left = target[position - 1]
                    right = target[position]

                    left_is_vowel = any(
                        char in VOWELS for char in left
                    )
                    right_is_vowel = any(
                        char in VOWELS for char in right
                    )

                    if not left_is_vowel and not right_is_vowel:
                        internal_vowels.append(observed)

        else:
            position += 1

    if initial_vowels:
        cluster = "".join(CLUSTERS[word])

        add(
            tr(
                f"{kana}〔처음 자음 사이 모음 생략〕",
                f"{kana}〔語頭の子音間に母音を加えない〕",
            ),
            tr(
                "처음 자음 사이에서 추가 모음이 인식됐어: "
                + ", ".join(initial_vowels),
                "語頭の子音間に追加の母音が認識されました："
                + "、".join(initial_vowels),
            ),
            tr(
                f"가나를 한 글자씩 끊기보다 /{cluster}/를 "
                "모음 없이 연결해 봐.",
                f"カナを一文字ずつ区切らず、/{cluster}/ を"
                "母音を加えずにつなげましょう。",
            ),
        )

    if internal_vowels:
        add(
            tr(
                f"{kana}〔중간 자음 사이 모음 생략〕",
                f"{kana}〔語中の子音間に母音を加えない〕",
            ),
            tr(
                "단어 중간 자음 사이에서 추가 모음이 인식됐어: "
                + ", ".join(internal_vowels),
                "語中の子音間に追加の母音が認識されました："
                + "、".join(internal_vowels),
            ),
            tr(
                "소리별 비교에서 추가 인식된 모음의 위치를 확인하고, "
                "앞뒤 자음을 모음 없이 연결해 봐.",
                "音ごとの比較で追加の母音の位置を確認し、"
                "前後の子音を母音なしでつなげましょう。",
            ),
        )

    if final_vowels:
        final = target[-1]

        add(
            tr(
                f"{kana}〔끝 모음 생략〕",
                f"{kana}〔語末に母音を加えない〕",
            ),
            tr(
                "단어 끝에서 추가 모음이 인식됐어: "
                + ", ".join(final_vowels),
                "語末に追加の母音が認識されました："
                + "、".join(final_vowels),
            ),
            tr(
                f"마지막 가나의 모음까지 읽기보다 "
                f"영어 /{final}/로 끝내 봐.",
                f"最後のカナの母音まで読まず、"
                f"英語の /{final}/ で終えてみましょう。",
            ),
        )

    for phone in ("l", "ɹ"):
        differences = [
            observed
            for expected, observed, kind in alignment
            if expected == phone
            and kind in {"replace", "missing"}
        ]

        if not differences:
            continue

        if phone == "l":
            instruction = tr(
                "/l/에서 혀끝을 윗잇몸 부근에 대고, "
                "라행처럼 혀끝을 튕기는 소리와 구분해 봐.",
                "/l/ では舌先を上の歯茎付近に当て、"
                "ラ行のように舌先を弾く音と区別しましょう。",
            )

        else:
            instruction = tr(
                "/ɹ/에서 혀끝을 라행처럼 튕기지 않고 "
                "기준 음성과 비교해 봐.",
                "/ɹ/ ではラ行のように舌先を弾かず、"
                "基準音声と聞き比べましょう。",
            )

        add(
            tr(
                f"{kana}〔/{phone}/ 발음 조정〕",
                f"{kana}〔/{phone}/ の発音を調整〕",
            ),
            tr(
                f"기준 /{phone}/가 다르게 인식되거나 검출되지 않았어.",
                f"基準の /{phone}/ が異なって認識されたか、"
                "検出されませんでした。",
            ),
            instruction,
        )

    return candidates


# =========================================================
# 9. 상단 화면
# =========================================================

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
    "단어 선택 → 기준 발음 듣기 → 녹음 → 맞춤 연습용 가나 확인 → 다시 녹음",
    "単語を選ぶ → 基準音声を聞く → 録音 → "
    "練習用カナの提案を確認 → 再録音",
))

word = st.selectbox(
    tr("연습할 영어 단어", "練習する英単語"),
    list(WORDS),
    key="practice_word",
)

selection = (word, language)

if st.session_state.current_selection != selection:
    st.session_state.current_selection = selection
    st.session_state.last_analysis = None
    st.session_state.last_error = None
    st.session_state.last_attempt_key = None
    st.session_state.record_version += 1

try:
    target_text, reference_audio = reference(word)

except Exception as error:
    st.error(tr(
        "기준 음성을 준비하지 못했어. 관리자에게 알려 줘.",
        "基準音声を準備できませんでした。管理者にお知らせください。",
    ))

    with st.expander(tr("오류 상세", "エラー詳細")):
        st.exception(error)

    st.stop()

st.subheader(f"{word} ／ {WORDS[word]}")

st.caption(tr(
    "가타카나는 일반 일본어 외래어 표기야. "
    "영어와 같은 발음이라는 뜻은 아니야.",
    "カタカナは一般的な外来語表記です。"
    "英語と同じ発音という意味ではありません。",
))

with st.expander(
    tr("녹음 전 연습 방법", "録音前の練習ガイド"),
    expanded=True,
):
    for tip in general_tips(word, target_text):
        st.markdown(f"- {tip}")

st.markdown(tr(
    "**영어 기준 발음 · 합성 음성**",
    "**英語の基準音声 · 合成音声**",
))

st.audio(reference_audio, format="audio/wav")

recording = st.audio_input(
    tr(
        "내 영어 발음 · 녹음 종료 후 자동 분석",
        "自分の英語発音 · 録音終了後に自動分析",
    ),
    sample_rate=16000,
    key=f"audio_{st.session_state.record_version}",
)

st.caption(tr(
    "녹음 버튼을 누르고 영어 단어를 한 번 말한 뒤 종료해 줘. "
    "녹음은 분석을 위해 서버로 전송돼.",
    "録音ボタンを押し、英単語を1回話して終了してください。"
    "録音は分析のためサーバーに送信されます。",
))


# =========================================================
# 10. 녹음 종료 후 자동 분석
# =========================================================

if recording is not None:
    raw_audio = recording.getvalue()

    audio_key = (
        word + ":"
        + hashlib.sha256(raw_audio).hexdigest()
    )

    # 성공·실패 여부와 관계없이 같은 녹음의 자동 재실행 방지
    if st.session_state.last_attempt_key != audio_key:
        st.session_state.last_attempt_key = audio_key
        st.session_state.last_analysis = None
        st.session_state.last_error = None

        with st.spinner(tr(
            "발음을 분석하고 있어. "
            "첫 분석은 모델을 불러오느라 시간이 걸릴 수 있어.",
            "発音を分析しています。"
            "初回はモデルの読み込みに時間がかかる場合があります。",
        )):
            try:
                result = analyze_recording(
                    raw_audio, target_text
                )

                result["word"] = word
                result["audio_key"] = audio_key

                st.session_state.last_analysis = result

                already_saved = any(
                    item["audio_key"] == audio_key
                    for item in st.session_state.practice_history
                )

                if not already_saved:
                    st.session_state.practice_history.append({
                        "audio_key": audio_key,
                        "word": word,
                        "score": result["score"],
                    })

            except Exception as error:
                st.session_state.last_error = str(error)


if st.session_state.last_error:
    st.error(st.session_state.last_error)

    if st.button(tr("분석 다시 시도", "分析を再試行")):
        st.session_state.last_attempt_key = None
        st.rerun()


# =========================================================
# 11. 분석 결과와 개인별 가나 추천
# =========================================================

result = st.session_state.last_analysis

if (
    recording is not None
    and result is not None
    and result["word"] == word
):
    st.divider()

    st.metric(
        tr("음소 유사도", "音素の類似度"),
        f"{result['score']:.1f}%",
    )

    st.caption(tr(
        "AI가 인식한 음소와 기준 음소의 일치 정도를 비교한 참고값이야.",
        "AIが認識した音素と基準音素を比較した参考値です。",
    ))

    st.subheader(tr(
        "이번 녹음에 맞춘 연습용 가나",
        "今回の録音に合わせた練習用カナ",
    ))

    candidates = kana_recommendations(word, result)

    if candidates:
        # 감지된 차이별 추천이며 효과의 순위를 의미하지 않음
        for item in candidates[:3]:
            st.markdown(f"#### {item['label']}")
            st.write(
                tr("선택 이유: ", "選択理由：")
                + item["reason"]
            )
            st.write(item["instruction"])

        st.caption(tr(
            "〔 〕는 가나를 읽을 때 조정할 부분을 나타내는 연습 표시야.",
            "〔 〕はカナを読むときの調整点を示す練習用の表示です。",
        ))

    else:
        st.write(tr(
            "이번에는 추가 모음이나 /l/, /ɹ/ 조정 후보를 "
            "선택할 만한 차이가 검출되지 않았어. "
            "기준 음성과 내 녹음을 다시 들어 보고, "
            "아래 소리별 비교도 확인해 봐.",
            "今回は追加母音や /l/、/ɹ/ の調整候補を"
            "選ぶ対象の違いは検出されませんでした。"
            "基準音声と自分の録音を聞き比べ、"
            "下の音ごとの比較も確認してください。",
        ))

    st.write(tr(
        "추천한 방법으로 연습한 뒤 다시 녹음해서 "
        "소리와 유사도의 변화를 확인해 봐.",
        "提案された方法で練習して録音し直し、"
        "音と類似度の変化を確認しましょう。",
    ))

    with st.expander(tr(
        "소리별 비교 자세히 보기",
        "音ごとの比較を見る",
    )):
        st.write(
            tr("기준 소리: ", "基準の音：")
            + result["target"]
        )

        st.write(
            tr("AI가 인식한 소리: ", "AIが認識した音：")
            + result["observed"]
        )

        kind_names = {
            "same": tr("일치", "一致"),
            "replace": tr("다르게 인식", "相違"),
            "missing": tr("미검출", "未検出"),
            "extra": tr("추가 인식", "追加検出"),
        }

        comparison_rows = [
            {
                tr("기준 소리", "基準の音"): expected,
                tr("인식한 소리", "認識された音"): observed,
                tr("비교", "比較"): kind_names[kind],
            }
            for expected, observed, kind
            in result["alignment"]
        ]

        st.dataframe(
            comparison_rows,
            hide_index=True,
        )

        st.caption(tr(
            "음성 인식 오류나 허용되는 발음 차이도 포함될 수 있어.",
            "音声認識の誤りや、許容される発音の違いも含まれる場合があります。",
        ))


# =========================================================
# 12. 화면 아래 연습 기록
# =========================================================

st.divider()
st.subheader(tr(
    "내 연습 기록",
    "自分の練習記録",
))

history_rows = []
previous_scores = {}

for number, item in enumerate(
    st.session_state.practice_history,
    start=1,
):
    previous = previous_scores.get(item["word"])

    if previous is None:
        difference = "—"

    else:
        difference = (
            f"{item['score'] - previous:+.1f} %p"
        )

    history_rows.append({
        tr("회차", "回数"): number,
        tr("영어 단어", "英単語"): item["word"],
        tr("음소 유사도(%)", "音素の類似度(%)"): item["score"],
        tr(
            "직전 같은 단어 대비",
            "同じ単語の前回との差",
        ): difference,
    })

    previous_scores[item["word"]] = item["score"]

if history_rows:
    st.dataframe(
        history_rows,
        hide_index=True,
    )

else:
    st.caption(tr(
        "녹음하고 분석하면 연습 기록이 여기에 쌓여.",
        "録音して分析すると、練習記録がここに表示されます。",
    ))

if st.button(tr(
    "내 연습 기록 지우기",
    "自分の練習記録を消去",
)):
    st.session_state.practice_history = []
    st.session_state.last_analysis = None
    st.session_state.last_error = None
    st.session_state.last_attempt_key = None
    st.session_state.record_version += 1
    st.rerun()

st.caption(tr(
    "기록은 현재 접속 세션에서 유지돼. "
    "새로고침이나 연결 종료 시 초기화될 수 있어.",
    "記録は現在のセッションで保持され、"
    "更新や切断で失われる場合があります。",
))


# =========================================================
# 13. 도구 소개 — 필요할 때만 펼쳐 보기
# =========================================================

with st.expander(tr(
    "도구 소개와 출처",
    "ツールについて・出典",
)):
    st.write(tr(
        "AI 음소 인식 결과를 기준 발음과 비교하고, "
        "인식된 차이에 따라 미리 마련한 연습용 가나 표시와 "
        "발음 조정 안내를 선택해 제공해.",
        "AIの音素認識結果を基準の発音と比較し、"
        "認識された違いに応じて、用意した練習用カナ表示と"
        "発音調整ガイドを選択して提供します。",
    ))

    st.write(tr(
        "유사도는 사람의 발음 평가를 대체하지 않아. "
        "가나 추천은 최적 표기를 자동 탐색한 결과가 아니라 "
        "인식된 차이에 따른 연습 제안이야.",
        "類似度は人による発音評価の代わりではありません。"
        "カナの提案は最適表記の自動探索ではなく、"
        "認識された違いに基づく練習の提案です。",
    ))

    st.markdown(
        "- [Wav2Vec2 음소 인식 모델]"
        "(https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)\n"
        "- [eSpeak NG 기준 음성·음소 생성]"
        "(https://github.com/espeak-ng/espeak-ng)"
    )
