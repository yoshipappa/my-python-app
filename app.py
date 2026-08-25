import os
import tempfile
import time

import cv2
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


st.title("作業分析プロトタイプ3")
st.write("MP4をアップロードし、動画を再生しながら作業区間を記録できます。")


# =========================
# Gemini Structured Output schema
# =========================
class GeminiWorkSegment(BaseModel):
    start_time: str = Field(description="作業開始時刻。MM:SS形式。")
    end_time: str = Field(description="作業終了時刻。MM:SS形式。")
    task_name: str = Field(description="短く分かりやすい作業名。")
    description: str = Field(description="作業内容の簡単な説明。")


class GeminiWorkSegments(BaseModel):
    segments: list[GeminiWorkSegment] = Field(
        description="動画内の作業区間の一覧。"
    )


class MasterClassification(BaseModel):
    master_code: str = Field(
        description="最も適切な作業マスタのコード。該当なしの場合はUNMATCHED。"
    )
    master_name: str = Field(
        description="最も適切な作業マスタ名。該当なしの場合はマスタ該当なし。"
    )
    confidence: int = Field(
        description="分類の確信度。0から100までの整数。"
    )
    reason: str = Field(
        description="その作業マスタを選択した理由。"
    )


class MasterClassificationResults(BaseModel):
    results: list[MasterClassification] = Field(
        description="入力された作業区間と同じ順番の作業マスタ分類結果。"
    )


class ImprovementSuggestion(BaseModel):
    priority: str = Field(description="優先度。高・中・低のいずれか。")
    target_task: str = Field(description="改善対象の作業名。")
    issue: str = Field(description="作業時間や繰り返しから見た問題・着眼点。")
    evidence: str = Field(description="提示されたデータに基づく根拠。")
    suggestion: str = Field(description="具体的な改善案。")
    expected_effect: str = Field(description="期待できる効果。")


class ImprovementAnalysis(BaseModel):
    summary: str = Field(description="分析結果の総括。")
    suggestions: list[ImprovementSuggestion] = Field(
        description="改善提案の一覧。最大5件程度。"
    )


class MotionElement(BaseModel):
    start_time: str = Field(description="動作開始時刻。MM:SS形式。")
    end_time: str = Field(description="動作終了時刻。MM:SS形式。")
    motion_type: str = Field(description="動作分類。例：取る、置く、運ぶ、移動、位置決め、締める、確認、待つ、その他。")
    description: str = Field(description="その動作の具体的な説明。")


class MotionSegmentResult(BaseModel):
    segment_index: int = Field(description="作業区間の番号。1から始まる。")
    task_name: str = Field(description="対象となる作業名。")
    motions: list[MotionElement] = Field(description="作業区間内の動作要素一覧。")


class MotionAnalysisResults(BaseModel):
    segments: list[MotionSegmentResult] = Field(description="入力された作業区間ごとの動作分析結果。")


def seconds_to_hms(total_seconds):
    """秒数を 時:分:秒 の文字列に変換する"""
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def hms_to_seconds(hours, minutes, seconds):
    """時・分・秒を合計秒数に変換する"""
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_hms_parts(total_seconds):
    """秒数を 時, 分, 秒 の3つに分ける"""
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return hours, minutes, seconds


def get_default_work_master():
    """最初に用意する作業マスタ（後から追加・編集しやすい一覧形式）"""
    return [
        {"code": "001", "name": "部品を取る"},
        {"code": "002", "name": "部品を組み付ける"},
        {"code": "003", "name": "ネジを締める"},
        {"code": "004", "name": "工具を取る"},
        {"code": "005", "name": "完成品を置く"},
        {"code": "006", "name": "検査する"},
    ]


def master_label(item):
    """画面表示用の「コード　作業名」"""
    return f"{item['code']}　{item['name']}"


def apply_master_selection(segment_index, segment_id):
    """マスタ選択が変わったときに作業名へ反映する（widget生成前のcallbackで実行）"""
    select_key = f"seg_master_{segment_id}"
    name_key = f"seg_name_{segment_id}"
    selected = st.session_state.get(select_key, "-- マスタから選択 --")

    if selected == "-- マスタから選択 --":
        return

    if selected == "未設定":
        chosen_name = "未設定"
        st.session_state[name_key] = ""
    else:
        # 「001　部品を取る」→ 作業名だけ取り出す
        chosen_name = selected.split("　", 1)[-1]
        st.session_state[name_key] = chosen_name

    st.session_state.segments[segment_index]["name"] = chosen_name

    # 同じselectboxの値を本文で書き換えるとエラーになるため、
    # callback内でkeyを削除して次回表示時に初期選択肢へ戻す
    if select_key in st.session_state:
        del st.session_state[select_key]


def parse_mmss(value):
    """MM:SS または HH:MM:SS を秒に変換する。失敗時はNone。"""
    try:
        parts = str(value).strip().split(":")
        if len(parts) == 2:
            minutes, seconds = [int(x) for x in parts]
            if minutes < 0 or not (0 <= seconds <= 59):
                return None
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = [int(x) for x in parts]
            if hours < 0 or not (0 <= minutes <= 59) or not (0 <= seconds <= 59):
                return None
            return hours * 3600 + minutes * 60 + seconds
        return None
    except (TypeError, ValueError):
        return None


def run_gemini_video_analysis(video_bytes, filename, video_duration_seconds=None):
    """Gemini Files APIで動画を解析し、構造化された作業区間を返す。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY が設定されていません。Windowsの環境変数を確認してください。"
        )

    client = genai.Client()

    tmp_path = None
    uploaded = None

    try:
        # Gemini Files APIへ渡すための一時ファイル
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        uploaded = client.files.upload(file=tmp_path)

        # 動画処理がACTIVEになるまで待機
        while uploaded.state and uploaded.state.name not in {"ACTIVE", "FAILED"}:
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if not uploaded.state or uploaded.state.name != "ACTIVE":
            raise RuntimeError("Gemini側で動画の処理に失敗しました。")

        prompt = """
動画を時系列に分析し、作業内容が変化するたびに1つの作業区間として分けてください。

各区間について、
- start_time: 作業開始時刻（MM:SS形式）
- end_time: 作業終了時刻（MM:SS形式）
- task_name: 短く分かりやすい作業名
- description: 作業内容の簡単な説明

を返してください。

ルール:
- 動画開始から終了まで、可能な限り連続した区間にしてください。
- 作業内容が変わった時点で区間を分けてください。
- 細かすぎる動作の違いではなく、工程・作業として意味のある変化で区切ってください。
- 時刻は動画上の実際の時刻に基づいてください。
- 判断できない場合は推測せず、descriptionに「判断困難」と記載してください。
"""

        # 現在の google-genai SDK で動作する generate_content + Structured Output
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiWorkSegments,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            result = parsed
        else:
            result = GeminiWorkSegments.model_validate_json(response.text)

        segments = []
        for item in result.segments:
            start = parse_mmss(item.start_time)
            end = parse_mmss(item.end_time)

            if start is None or end is None or end <= start:
                continue

            if (
                video_duration_seconds is not None
                and end > video_duration_seconds
            ):
                end = int(video_duration_seconds)

            if end <= start:
                continue

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "name": item.task_name.strip() or "未設定",
                    "description": item.description.strip(),
                }
            )

        if not segments:
            raise RuntimeError(
                "AIは作業区間を抽出しましたが、有効な時刻データがありませんでした。"
            )

        return segments

    finally:
        # ローカル一時ファイルを削除
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def classify_segments_with_master(ai_segments, work_master):
    """AIが抽出した作業区間を、現在の作業マスタへ分類する。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY が設定されていません。Windowsの環境変数を確認してください。"
        )

    if not work_master:
        raise RuntimeError("作業マスタが登録されていません。")

    client = genai.Client()

    master_text = "\n".join(
        f"{item['code']}：{item['name']}" for item in work_master
    )

    segment_text_lines = []
    for idx, segment in enumerate(ai_segments, start=1):
        segment_text_lines.append(
            f"{idx}. 作業名: {segment['name']} / 説明: {segment.get('description', '')}"
        )
    segment_text = "\n".join(segment_text_lines)

    prompt = f"""
以下の「作業マスタ」の中から、各AI認識作業に最も適切なものを1件ずつ選択してください。

【作業マスタ】
{master_text}

【AI認識作業】
{segment_text}

【ルール】
- 入力された作業区間と同じ順番で分類結果を返してください。
- 作業内容・説明をもとに最も適切なマスタを選んでください。
- 完全一致でなくても意味的に最も近いものを選んで構いません。
- 適切なマスタがない場合は master_code="UNMATCHED"、master_name="マスタ該当なし" としてください。
- confidenceは0～100の整数です。
- reasonには簡潔な理由を書いてください。
- 作業マスタ名そのものは変更しないでください。
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MasterClassificationResults,
        ),
    )

    parsed = getattr(response, "parsed", None)
    result = (
        parsed
        if parsed is not None
        else MasterClassificationResults.model_validate_json(response.text)
    )

    if len(result.results) != len(ai_segments):
        raise RuntimeError(
            "AIのマスタ分類結果の件数が、作業区間の件数と一致しませんでした。"
        )

    valid_codes = {item["code"]: item["name"] for item in work_master}
    classified = []

    for segment, classification in zip(ai_segments, result.results):
        code = classification.master_code.strip()
        name = classification.master_name.strip()
        confidence = max(0, min(100, int(classification.confidence)))

        if code != "UNMATCHED" and code not in valid_codes:
            code = "UNMATCHED"
            name = "マスタ該当なし"
            confidence = 0

        if code in valid_codes:
            name = valid_codes[code]
        else:
            name = "マスタ該当なし"

        classified.append(
            {
                **segment,
                "master_code": code,
                "master_name": name,
                "confidence": confidence,
                "reason": classification.reason.strip(),
            }
        )

    return classified


def analyze_data_improvements_with_gemini(segments):
    """登録された作業区間をGeminiで分析し、改善提案を作成する。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY が設定されていません。Windowsの環境変数を確認してください。"
        )

    if not segments:
        raise RuntimeError("分析対象の作業区間がありません。")

    client = genai.Client()

    task_rows = []
    for idx, segment in enumerate(segments, start=1):
        duration = max(0, int(segment["end"]) - int(segment["start"]))
        task_rows.append(
            f"{idx}. 作業名: {segment.get('name', '未設定')}; "
            f"開始: {seconds_to_hms(segment['start'])}; "
            f"終了: {seconds_to_hms(segment['end'])}; "
            f"時間: {seconds_to_hms(duration)}; "
            f"説明: {segment.get('description', '')}"
        )

    task_text = "\n".join(task_rows)

    # 集計値も明示して、AIが単純な印象論ではなく数字を根拠にするようにする。
    summary = {}
    for segment in segments:
        name = str(segment.get("name", "未設定")).strip() or "未設定"
        duration = max(0, int(segment["end"]) - int(segment["start"]))
        summary.setdefault(name, {"count": 0, "total_seconds": 0})
        summary[name]["count"] += 1
        summary[name]["total_seconds"] += duration

    summary_rows = []
    for name, data in summary.items():
        avg = data["total_seconds"] / data["count"] if data["count"] else 0
        summary_rows.append(
            f"- {name}: 回数={data['count']}, "
            f"合計={seconds_to_hms(data['total_seconds'])}, "
            f"平均={seconds_to_hms(round(avg))}"
        )
    summary_text = "\n".join(sorted(summary_rows))

    prompt = f"""
あなたは製造現場の作業分析担当者です。
以下の作業区間と作業別集計を分析し、改善提案を作ってください。

【作業区間】
{task_text}

【作業別集計】
{summary_text}

【分析ルール】
- 実際に提示されたデータから読み取れることを根拠にしてください。
- データだけでは断定できないことは、推測・仮説として扱ってください。
- 作業時間が長い作業だけでなく、繰り返し回数が多い作業、同じ作業が分散している状況にも着目してください。
- 具体的な改善案を出してください。
- 人・設備・レイアウト・工具・部品供給・標準作業の観点から検討してください。
- 改善案による効果を断定せず、「期待できる効果」として記載してください。
- 優先度は「高」「中」「低」のいずれかにしてください。
- 改善提案は最大5件程度にしてください。
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ImprovementAnalysis,
        ),
    )

    parsed = getattr(response, "parsed", None)
    result = parsed if parsed is not None else ImprovementAnalysis.model_validate_json(response.text)

    return result


def analyze_video_improvements_with_gemini(
    video_bytes,
    filename,
    segments,
):
    """作業区間データと動画映像を合わせてGeminiで改善分析する。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY が設定されていません。Windowsの環境変数を確認してください。"
        )

    if not segments:
        raise RuntimeError("分析対象の作業区間がありません。")

    client = genai.Client()

    # 作業区間はプロンプトで明示し、動画上の時刻と結び付ける。
    segment_rows = []
    for idx, segment in enumerate(segments, start=1):
        duration = max(0, int(segment["end"]) - int(segment["start"]))
        segment_rows.append(
            f"{idx}. 作業名: {segment.get('name', '未設定')}; "
            f"開始: {seconds_to_hms(segment['start'])}; "
            f"終了: {seconds_to_hms(segment['end'])}; "
            f"時間: {seconds_to_hms(duration)}; "
            f"説明: {segment.get('description', '')}"
        )

    segment_text = "\n".join(segment_rows)

    tmp_path = None
    uploaded = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        uploaded = client.files.upload(file=tmp_path)

        while uploaded.state and uploaded.state.name not in {"ACTIVE", "FAILED"}:
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if not uploaded.state or uploaded.state.name != "ACTIVE":
            raise RuntimeError("Gemini側で動画の処理に失敗しました。")

        prompt = f"""
あなたは製造現場の作業分析担当者です。
動画映像そのものを観察し、以下の既存作業区間データと照合して改善ポイントを分析してください。

【既存作業区間】
{segment_text}

【分析観点】
- 不要な移動・手待ち・持ち替え・取り直し
- 工具や部品を取るための余分な動作
- 作業姿勢や手の動かし方から見える改善余地
- 同じ動作の繰り返し
- 部品・工具の配置や供給方法
- 作業手順の順番
- 作業時間が長い区間に映像上の理由があるか
- 作業区間の時間情報と映像の内容が一致しているか

【重要なルール】
- 映像から確認できる事実と、推測・仮説を区別してください。
- 映像から確認できないことを断定しないでください。
- 改善提案は最大5件程度。
- 優先度は「高」「中」「低」のいずれか。
- evidenceには、可能なら具体的な時刻と映像上の観察事実を含めてください。
- expected_effectは可能性として表現し、効果を断定しないでください。
- 改善案は製造現場で実行可能なレベルで具体化してください。
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImprovementAnalysis,
            ),
        )

        parsed = getattr(response, "parsed", None)
        result = (
            parsed
            if parsed is not None
            else ImprovementAnalysis.model_validate_json(response.text)
        )
        return result

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def analyze_motion_with_gemini(video_bytes, filename, segments):
    """作業区間ごとに動画を確認し、動作要素へ分解する。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません。Windowsの環境変数を確認してください。")
    if not segments:
        raise RuntimeError("分析対象の作業区間がありません。")

    client = genai.Client()
    segment_rows = []
    for idx, segment in enumerate(segments, start=1):
        duration = max(0, int(segment["end"]) - int(segment["start"]))
        segment_rows.append(
            f"{idx}. 作業名: {segment.get('name', '未設定')}; "
            f"開始: {seconds_to_hms(segment['start'])}; "
            f"終了: {seconds_to_hms(segment['end'])}; "
            f"時間: {seconds_to_hms(duration)}"
        )
    segment_text = "\n".join(segment_rows)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        uploaded = client.files.upload(file=tmp_path)
        while uploaded.state and uploaded.state.name not in {"ACTIVE", "FAILED"}:
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if not uploaded.state or uploaded.state.name != "ACTIVE":
            raise RuntimeError("Gemini側で動画の処理に失敗しました。")

        prompt = f"""
あなたは製造現場の作業分析担当者です。
以下の作業区間について、動画映像を観察して「動作要素」に分解してください。

【作業区間】
{segment_text}

【動作分類】
基本的には以下の分類を使用してください。
- 取る
- 置く
- 運ぶ
- 移動
- 位置決め
- 締める
- 確認
- 待つ
- その他

【重要】
- 各作業区間の開始時刻・終了時刻の範囲内だけを分析してください。
- 1つの動作要素は、意味のある連続した動作単位にしてください。
- 細かく分けすぎず、作業改善に使える粒度にしてください。
- 動作の境界は、映像から判断してください。
- 判断できない場合は「その他」または説明に「判断困難」と記載してください。
- 入力された作業区間と同じ順番・同じ件数で結果を返してください。
- 各動作要素のstart_time/end_timeは動画内の実際の時刻を使用してください。
"""
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MotionAnalysisResults,
            ),
        )
        parsed = getattr(response, "parsed", None)
        result = parsed if parsed is not None else MotionAnalysisResults.model_validate_json(response.text)
        if len(result.segments) != len(segments):
            raise RuntimeError("AIの動作分析結果の件数が、作業区間の件数と一致しませんでした。")
        valid = set(range(1, len(segments) + 1))
        for motion_result in result.segments:
            if motion_result.segment_index not in valid:
                raise RuntimeError(f"不正な作業区間番号が返されました: {motion_result.segment_index}")
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass



def build_motion_summary_rows(motion_analysis):
    """動作分析結果から動作種別ごとの集計を作る。"""
    summary = {}

    if motion_analysis is None:
        return []

    for segment_result in motion_analysis.segments:
        for motion in segment_result.motions:
            start = parse_mmss(motion.start_time)
            end = parse_mmss(motion.end_time)

            if start is None or end is None or end < start:
                continue

            duration = end - start
            motion_type = motion.motion_type.strip() or "その他"

            if motion_type not in summary:
                summary[motion_type] = {
                    "count": 0,
                    "total_seconds": 0,
                }

            summary[motion_type]["count"] += 1
            summary[motion_type]["total_seconds"] += duration

    rows = []
    total_seconds = sum(item["total_seconds"] for item in summary.values())

    for motion_type, item in sorted(summary.items()):
        average_seconds = (
            item["total_seconds"] / item["count"]
            if item["count"] > 0
            else 0
        )
        ratio = (
            item["total_seconds"] / total_seconds * 100
            if total_seconds > 0
            else 0
        )

        rows.append({
            "動作分類": motion_type,
            "回数": item["count"],
            "合計時間": seconds_to_hms(item["total_seconds"]),
            "平均時間": seconds_to_hms(round(average_seconds)),
            "時間割合（%）": round(ratio, 1),
            "_total_seconds": item["total_seconds"],
        })

    rows.sort(key=lambda x: x["_total_seconds"], reverse=True)
    return rows


# ----- session_state の初期化 -----

# 作業マスタ
if "work_master" not in st.session_state:
    st.session_state.work_master = get_default_work_master()

# 作業区間の一覧
if "segments" not in st.session_state:
    st.session_state.segments = []

# 各作業区間を区別するための番号
if "next_segment_id" not in st.session_state:
    st.session_state.next_segment_id = 1

# AI分析結果の一時保管
if "ai_results" not in st.session_state:
    st.session_state.ai_results = []

# 作業マスタ分類が実行済みか
if "ai_master_classified" not in st.session_state:
    st.session_state.ai_master_classified = False

# AI分析対象の動画名
if "ai_result_filename" not in st.session_state:
    st.session_state.ai_result_filename = ""

# 改善分析結果
if "improvement_analysis" not in st.session_state:
    st.session_state.improvement_analysis = None

if "improvement_analysis_mode" not in st.session_state:
    st.session_state.improvement_analysis_mode = "video"

# 動作分析結果
if "motion_analysis" not in st.session_state:
    st.session_state.motion_analysis = None

# 動作別集計
if "motion_summary" not in st.session_state:
    st.session_state.motion_summary = []

# 古いデータに id / name が無い場合に付ける
for segment in st.session_state.segments:
    if "id" not in segment:
        segment["id"] = st.session_state.next_segment_id
        st.session_state.next_segment_id += 1
    if "name" not in segment or not str(segment["name"]).strip():
        segment["name"] = "未設定"

# 現在位置から設定した開始・終了時刻
if "mark_start" not in st.session_state:
    st.session_state.mark_start = None
if "mark_end" not in st.session_state:
    st.session_state.mark_end = None

# 手動入力用の初期値
for key in ["start_h", "start_m", "start_s", "end_h", "end_m", "end_s"]:
    if key not in st.session_state:
        st.session_state[key] = 0


# ----- 作業マスタ -----
st.subheader("作業マスタ")
st.write("あらかじめ登録した作業名です。作業区間ではここから選ぶことも、自由に入力することもできます。")

if len(st.session_state.work_master) == 0:
    st.write("作業マスタがありません。")
else:
    master_header = st.columns([1, 4])
    master_header[0].write("コード")
    master_header[1].write("作業名")
    for item in st.session_state.work_master:
        master_row = st.columns([1, 4])
        master_row[0].write(item["code"])
        master_row[1].write(item["name"])

# 後からマスタを追加できる簡単な入力欄
with st.expander("作業マスタを追加する"):
    add_col1, add_col2, add_col3 = st.columns([1, 3, 1])
    with add_col1:
        new_code = st.text_input("コード", key="new_master_code", placeholder="007")
    with add_col2:
        new_name = st.text_input("作業名", key="new_master_name", placeholder="新しい作業名")
    with add_col3:
        st.write("")
        st.write("")
        if st.button("マスタに追加"):
            code = new_code.strip()
            name = new_name.strip()
            if code == "" or name == "":
                st.error("コードと作業名の両方を入力してください。")
            elif any(item["code"] == code for item in st.session_state.work_master):
                st.error("同じコードがすでにあります。")
            else:
                st.session_state.work_master.append({"code": code, "name": name})
                st.session_state.new_master_code = ""
                st.session_state.new_master_name = ""
                st.success(f"作業マスタに追加しました：{code}　{name}")
                st.rerun()

st.divider()

uploaded_file = st.file_uploader("動画ファイルを選択してください", type=["mp4"])

if uploaded_file is not None:
    # アップロードされたファイルの中身を読み取る
    video_bytes = uploaded_file.read()

    # OpenCV用に一時ファイルとして保存
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # OpenCVで動画を開いて再生時間を取得
        cap = cv2.VideoCapture(tmp_path)
        video_duration_seconds = None

        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if fps > 0:
                video_duration_seconds = int(frame_count / fps)
                st.write(f"ファイル名：{uploaded_file.name}")
                st.success(f"再生時間：{seconds_to_hms(video_duration_seconds)}")
            else:
                st.error("FPS情報を取得できませんでした。")

            cap.release()
        else:
            st.error("動画ファイルを開けませんでした。")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # アプリ画面内で動画を再生
    st.video(video_bytes)

    # ----- Gemini AI分析 -----
    st.subheader("AIによる作業分析")
    st.write(
        "Geminiに動画を送信し、作業内容が変化する区間を自動抽出します。"
        "動画はGoogle Gemini APIへ送信されます。"
    )

    if st.button("AIで作業分析", type="primary"):
        try:
            with st.spinner("AIが動画を分析しています。しばらくお待ちください。"):
                st.session_state.ai_results = run_gemini_video_analysis(
                    video_bytes=video_bytes,
                    filename=uploaded_file.name,
                    video_duration_seconds=video_duration_seconds,
                )
                st.session_state.ai_result_filename = uploaded_file.name
                st.session_state.ai_master_classified = False

            st.success(
                f"AI分析が完了しました。{len(st.session_state.ai_results)}件の作業区間を抽出しました。"
            )
        except Exception as exc:
            st.error(f"AI分析に失敗しました：{exc}")

    # ----- AI分析結果 -----
    if st.session_state.ai_results:
        st.subheader("AI分析結果（確認してから追加）")
        st.write(
            f"対象動画：{st.session_state.ai_result_filename}"
        )
        st.caption(
            "AIの結果は自動で作業区間一覧には入りません。内容を確認してから追加してください。"
        )

        # 作業マスタ分類
        if not st.session_state.ai_master_classified:
            st.info(
                "次に「AIで作業マスタ分類」を実行すると、各AI作業名を現在の作業マスタへ分類できます。"
            )
            if st.button("AIで作業マスタ分類", type="secondary"):
                try:
                    with st.spinner("AIが作業マスタとの対応を判定しています。"):
                        st.session_state.ai_results = classify_segments_with_master(
                            st.session_state.ai_results,
                            st.session_state.work_master,
                        )
                        st.session_state.ai_master_classified = True
                    st.success("作業マスタ分類が完了しました。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"作業マスタ分類に失敗しました：{exc}")

        result_header = st.columns([0.6, 1.3, 1.3, 2.0, 2.5, 1.0])
        result_header[0].write("No.")
        result_header[1].write("開始")
        result_header[2].write("終了")
        result_header[3].write("AI認識作業")
        result_header[4].write("作業マスタ候補")
        result_header[5].write("信頼度")

        for idx, result in enumerate(st.session_state.ai_results):
            row = st.columns([0.6, 1.3, 1.3, 2.0, 2.5, 1.0])
            row[0].write(str(idx + 1))
            row[1].write(seconds_to_hms(result["start"]))
            row[2].write(seconds_to_hms(result["end"]))
            row[3].write(result["name"])

            if st.session_state.ai_master_classified and "master_name" in result:
                master_code = result.get("master_code", "UNMATCHED")
                master_name = result.get("master_name", "マスタ該当なし")
                confidence = int(result.get("confidence", 0))
                reason = result.get("reason", "")

                if master_code == "UNMATCHED":
                    row[4].warning("マスタ該当なし")
                else:
                    row[4].write(f"{master_code}　{master_name}")
                row[5].write(f"{confidence}%")

                st.caption(f"理由：{reason}")

                if master_code != "UNMATCHED":
                    if row[4].button(
                        "候補を採用",
                        key=f"accept_master_{idx}",
                    ):
                        st.session_state.ai_results[idx]["name"] = master_name
                        st.success(
                            f"{idx + 1}番の作業名を「{master_name}」に変更しました。"
                        )
                        st.rerun()
            else:
                row[4].write("未分類")
                row[5].write("-")

            st.write("---")

        # AI分析結果を作業区間へ追加
        if st.button("AI分析結果を作業区間に追加"):
            added_count = 0
            duplicate_count = 0

            for result in st.session_state.ai_results:
                is_duplicate = any(
                    seg["start"] == result["start"]
                    and seg["end"] == result["end"]
                    and seg.get("name", "未設定") == result["name"]
                    for seg in st.session_state.segments
                )

                if is_duplicate:
                    duplicate_count += 1
                    continue

                segment_id = st.session_state.next_segment_id
                st.session_state.next_segment_id += 1

                st.session_state.segments.append(
                    {
                        "id": segment_id,
                        "start": result["start"],
                        "end": result["end"],
                        "name": result["name"],
                        "description": result["description"],
                    }
                )
                added_count += 1

            st.session_state.ai_results = []
            st.session_state.ai_master_classified = False

            if added_count:
                st.success(f"{added_count}件のAI分析結果を作業区間に追加しました。")
            if duplicate_count:
                st.info(f"{duplicate_count}件は重複のため追加しませんでした。")
            st.rerun()

    # ----- AI改善分析 -----
    if len(st.session_state.segments) > 0:
        st.divider()
        st.subheader("AIによる改善分析")
        st.write(
            "登録された作業区間・作業時間に加えて、動画映像そのものをGeminiに確認させ、"
            "実際の動作・配置・手待ち・移動などを含めて改善候補を分析します。"
        )

        st.session_state.improvement_analysis_mode = st.radio(
            "分析方法",
            options=["video", "data"],
            format_func=lambda x: (
                "動画＋作業データで分析" if x == "video"
                else "作業データだけで分析"
            ),
            horizontal=True,
            key="improvement_mode_radio",
        )

        if st.session_state.improvement_analysis_mode == "video":
            st.caption(
                "動画をGemini APIへ送信します。映像から確認できる事実を根拠に改善候補を作成します。"
            )
        else:
            st.caption(
                "動画は送信せず、現在登録されている作業区間・時間データだけを分析します。"
            )

        if st.button("AIで改善分析", type="primary"):
            try:
                with st.spinner(
                    "AIが作業区間・作業時間・映像を分析しています。しばらくお待ちください。"
                ):
                    if st.session_state.improvement_analysis_mode == "video":
                        st.session_state.improvement_analysis = (
                            analyze_video_improvements_with_gemini(
                                video_bytes=video_bytes,
                                filename=uploaded_file.name,
                                segments=st.session_state.segments,
                            )
                        )
                    else:
                        st.session_state.improvement_analysis = (
                            analyze_data_improvements_with_gemini(
                                st.session_state.segments
                            )
                        )

                st.success("改善分析が完了しました。")
            except Exception as exc:
                st.error(f"改善分析に失敗しました：{exc}")

        if st.session_state.improvement_analysis is not None:
            analysis = st.session_state.improvement_analysis

            st.markdown("### 分析総括")
            st.info(analysis.summary)

            st.markdown("### 改善提案")

            priority_icons = {
                "高": "🔴",
                "中": "🟡",
                "低": "🟢",
            }

            if len(analysis.suggestions) == 0:
                st.write("現時点では明確な改善提案はありません。")
            else:
                for idx, suggestion in enumerate(analysis.suggestions, start=1):
                    icon = priority_icons.get(suggestion.priority, "")
                    with st.expander(
                        f"{icon} {idx}. {suggestion.target_task}（優先度：{suggestion.priority}）",
                        expanded=(idx == 1),
                    ):
                        st.write("**問題・着眼点**")
                        st.write(suggestion.issue)

                        st.write("**根拠**")
                        st.write(suggestion.evidence)

                        st.write("**改善案**")
                        st.write(suggestion.suggestion)

                        st.write("**期待できる効果**")
                        st.write(suggestion.expected_effect)

    # ----- AI動作分類 -----
    if len(st.session_state.segments) > 0:
        st.divider()
        st.subheader("AIによる動作分類")
        st.write("登録された作業区間を、動画映像から「取る・置く・運ぶ・移動・位置決め・締める・確認・待つ」などの動作要素に分解します。")
        st.caption("動画をGemini APIへ送信し、作業区間ごとに動作を分析します。")

        if st.button("AIで動作分類", type="primary"):
            try:
                with st.spinner("AIが動画内の動作を分析しています。しばらくお待ちください。"):
                    st.session_state.motion_analysis = analyze_motion_with_gemini(
                        video_bytes=video_bytes,
                        filename=uploaded_file.name,
                        segments=st.session_state.segments,
                    )
                    st.session_state.motion_summary = build_motion_summary_rows(
                        st.session_state.motion_analysis
                    )
                st.success("動作分類が完了しました。")
            except Exception as exc:
                st.error(f"動作分類に失敗しました：{exc}")

        if st.session_state.motion_analysis is not None:
            st.markdown("### 動作分析結果")
            for result in st.session_state.motion_analysis.segments:
                st.markdown(f"**作業 {result.segment_index}：{result.task_name}**")
                motion_rows = []
                for motion in result.motions:
                    start = parse_mmss(motion.start_time)
                    end = parse_mmss(motion.end_time)
                    duration = end - start if start is not None and end is not None and end >= start else None
                    motion_rows.append({
                        "開始": motion.start_time,
                        "終了": motion.end_time,
                        "動作分類": motion.motion_type,
                        "動作時間": seconds_to_hms(duration) if duration is not None else "-",
                        "説明": motion.description,
                    })
                if motion_rows:
                    st.dataframe(motion_rows, use_container_width=True, hide_index=True)
                else:
                    st.write("動作要素を抽出できませんでした。")


    # ----- 動作別集計・改善優先順位 -----
    if st.session_state.motion_analysis is not None:
        st.divider()
        st.subheader("動作別集計")

        motion_rows = st.session_state.motion_summary

        if motion_rows:
            display_motion_rows = [
                {
                    "動作分類": row["動作分類"],
                    "回数": row["回数"],
                    "合計時間": row["合計時間"],
                    "平均時間": row["平均時間"],
                    "時間割合（%）": row["時間割合（%）"],
                }
                for row in motion_rows
            ]

            st.dataframe(
                display_motion_rows,
                use_container_width=True,
                hide_index=True,
            )

            total_motion_seconds = sum(
                row["_total_seconds"] for row in motion_rows
            )
            st.write(
                f"**分析対象動作の合計時間："
                f"{seconds_to_hms(total_motion_seconds)}**"
            )

            st.markdown("### 改善の着眼点")
            candidates = []

            for row in motion_rows:
                score = 0

                if total_motion_seconds > 0:
                    share = row["_total_seconds"] / total_motion_seconds
                    score += share * 60

                score += min(row["回数"], 20) * 2

                if row["動作分類"] in {"待つ", "移動", "運ぶ"}:
                    score += 15

                candidates.append({
                    "動作分類": row["動作分類"],
                    "回数": row["回数"],
                    "合計時間": row["合計時間"],
                    "時間割合（%）": row["時間割合（%）"],
                    "_score": score,
                })

            candidates.sort(key=lambda x: x["_score"], reverse=True)

            for rank, candidate in enumerate(candidates[:5], start=1):
                st.write(
                    f"{rank}. **{candidate['動作分類']}**："
                    f"{candidate['回数']}回、"
                    f"合計{candidate['合計時間']}、"
                    f"全体の{candidate['時間割合（%）']}%"
                )

            st.caption(
                "※この順位は改善効果を断定するものではなく、"
                "動作時間・回数などの数値から抽出した改善検討の優先候補です。"
            )
        else:
            st.write("動作別集計を作成できませんでした。")

    # ----- 現在位置から開始・終了を設定 -----
    st.subheader("現在位置の指定")
    st.write("動画を見ながら、下のスライダーで現在位置を合わせ、開始・終了に設定できます。")

    if video_duration_seconds is not None and video_duration_seconds > 0:
        current_position = st.slider(
            "現在位置",
            min_value=0,
            max_value=video_duration_seconds,
            value=0,
            step=1,
            key="current_position",
            format="%d 秒",
        )
        st.write(f"現在位置：{seconds_to_hms(current_position)}")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("現在位置を開始時刻に設定"):
                h, m, s = seconds_to_hms_parts(current_position)
                st.session_state.start_h = h
                st.session_state.start_m = m
                st.session_state.start_s = s
                st.session_state.mark_start = current_position
        with btn_col2:
            if st.button("現在位置を終了時刻に設定"):
                h, m, s = seconds_to_hms_parts(current_position)
                st.session_state.end_h = h
                st.session_state.end_m = m
                st.session_state.end_s = s
                st.session_state.mark_end = current_position

        if st.session_state.mark_start is not None:
            st.info(f"設定中の開始時刻：{seconds_to_hms(st.session_state.mark_start)}")
        else:
            st.write("設定中の開始時刻：まだ設定されていません")

        if st.session_state.mark_end is not None:
            st.info(f"設定中の終了時刻：{seconds_to_hms(st.session_state.mark_end)}")
        else:
            st.write("設定中の終了時刻：まだ設定されていません")
    else:
        st.warning("動画の長さが取得できないため、現在位置の指定は使えません。下の手動入力をご利用ください。")

    # ----- 作業区間の登録 -----
    st.subheader("作業区間の登録")
    st.write("開始時刻と終了時刻を指定して、作業区間を登録します。（上のボタンで設定した値もここに反映されます）")

    col_start, col_end = st.columns(2)

    with col_start:
        st.markdown("**開始時刻**")
        start_h = st.number_input("開始・時", min_value=0, step=1, key="start_h")
        start_m = st.number_input("開始・分", min_value=0, max_value=59, step=1, key="start_m")
        start_s = st.number_input("開始・秒", min_value=0, max_value=59, step=1, key="start_s")

    with col_end:
        st.markdown("**終了時刻**")
        end_h = st.number_input("終了・時", min_value=0, step=1, key="end_h")
        end_m = st.number_input("終了・分", min_value=0, max_value=59, step=1, key="end_m")
        end_s = st.number_input("終了・秒", min_value=0, max_value=59, step=1, key="end_s")

    if st.button("作業区間を登録"):
        start_seconds = hms_to_seconds(start_h, start_m, start_s)
        end_seconds = hms_to_seconds(end_h, end_m, end_s)

        if end_seconds <= start_seconds:
            st.error("終了時刻は開始時刻より後にしてください。")
        elif video_duration_seconds is not None and end_seconds > video_duration_seconds:
            st.error("終了時刻が動画の長さを超えています。")
        else:
            segment_id = st.session_state.next_segment_id
            st.session_state.next_segment_id += 1
            st.session_state.segments.append(
                {
                    "id": segment_id,
                    "start": start_seconds,
                    "end": end_seconds,
                    "name": "未設定",
                }
            )
            st.success("作業区間を登録しました。")

    # ----- 作業区間の一覧 -----
    st.subheader("作業区間一覧")
    st.write(
        "作業名は「マスタから選択」か、右の自由入力で設定できます。"
        "マスタにない名前も入力できます。空欄にすると「未設定」になります。"
    )

    if len(st.session_state.segments) == 0:
        st.write("まだ作業区間は登録されていません。")
    else:
        master_options = ["-- マスタから選択 --", "未設定"] + [
            master_label(item) for item in st.session_state.work_master
        ]

        header = st.columns([0.7, 1.5, 1.5, 1.5, 2.5, 2.5, 0.8])
        header[0].write("No.")
        header[1].write("開始時刻")
        header[2].write("終了時刻")
        header[3].write("作業時間")
        header[4].write("マスタから選択")
        header[5].write("作業名（自由入力）")
        header[6].write("削除")

        for i, segment in enumerate(st.session_state.segments):
            work_seconds = segment["end"] - segment["start"]
            row = st.columns([0.7, 1.5, 1.5, 1.5, 2.5, 2.5, 0.8])
            row[0].write(str(i + 1))
            row[1].write(seconds_to_hms(segment["start"]))
            row[2].write(seconds_to_hms(segment["end"]))
            row[3].write(seconds_to_hms(work_seconds))

            name_key = f"seg_name_{segment['id']}"
            select_key = f"seg_master_{segment['id']}"

            if name_key not in st.session_state:
                st.session_state[name_key] = (
                    "" if segment["name"] == "未設定" else segment["name"]
                )

            row[4].selectbox(
                "マスタ選択",
                options=master_options,
                key=select_key,
                label_visibility="collapsed",
                on_change=apply_master_selection,
                args=(i, segment["id"]),
            )

            edited_name = row[5].text_input(
                "作業名",
                key=name_key,
                placeholder="未設定",
                label_visibility="collapsed",
            )
            cleaned_name = edited_name.strip()
            st.session_state.segments[i]["name"] = (
                cleaned_name if cleaned_name else "未設定"
            )

            if row[6].button("削除", key=f"delete_{segment['id']}"):
                removed = st.session_state.segments.pop(i)
                for key in [
                    f"seg_name_{removed['id']}",
                    f"seg_master_{removed['id']}",
                ]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


    # ----- 作業別集計 -----
    if len(st.session_state.segments) > 0:
        st.divider()
        st.subheader("作業別集計")
        st.write("登録された作業区間を作業名ごとに集計します。")

        summary = {}
        for segment in st.session_state.segments:
            name = str(segment.get("name", "未設定")).strip() or "未設定"
            work_seconds = max(0, int(segment["end"]) - int(segment["start"]))

            if name not in summary:
                summary[name] = {"count": 0, "total_seconds": 0}

            summary[name]["count"] += 1
            summary[name]["total_seconds"] += work_seconds

        summary_rows = []
        for name, data in sorted(summary.items()):
            total_seconds = data["total_seconds"]
            average_seconds = (
                total_seconds / data["count"] if data["count"] > 0 else 0
            )
            summary_rows.append({
                "作業名": name,
                "回数": data["count"],
                "合計時間": seconds_to_hms(total_seconds),
                "平均時間": seconds_to_hms(round(average_seconds)),
                "_total_seconds": total_seconds,
            })

        total_work_seconds = sum(
            row["_total_seconds"] for row in summary_rows
        )

        display_rows = [
            {
                "作業名": row["作業名"],
                "回数": row["回数"],
                "合計時間": row["合計時間"],
                "平均時間": row["平均時間"],
            }
            for row in summary_rows
        ]

        st.dataframe(
            display_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            f"**登録作業の合計時間：{seconds_to_hms(total_work_seconds)}**"
        )

        if total_work_seconds > 0:
            st.caption("合計時間に占める各作業の割合")

            ratio_rows = []
            for row in summary_rows:
                ratio = row["_total_seconds"] / total_work_seconds * 100
                ratio_rows.append({
                    "作業名": row["作業名"],
                    "時間割合": round(ratio, 1),
                })

            ratio_rows.sort(
                key=lambda x: x["時間割合"],
                reverse=True,
            )

            st.dataframe(
                ratio_rows,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "時間割合": st.column_config.NumberColumn(
                        "時間割合（%）",
                        format="%.1f%%",
                    )
                },
            )