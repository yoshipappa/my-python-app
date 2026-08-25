"""
Gemini API 作業区間抽出の単独テスト

test_video.mp4 を Files API でアップロードし、
Structured Output（JSON Schema）で作業区間を取得します。
APIキーは環境変数 GEMINI_API_KEY から読み込みます。
"""

import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


VIDEO_PATH = Path(__file__).resolve().parent / "test_video.mp4"
MODEL_NAME = "gemini-3.6-flash"

PROMPT = """
動画を時系列に分析してください。

作業内容が変化するたびに、1つの作業区間として分けてください。

各作業区間について、以下の項目を含めてください。
- start_time: 作業開始時刻（MM:SS形式）
- end_time: 作業終了時刻（MM:SS形式）
- task_name: 短く分かりやすい作業名
- description: 作業内容の簡単な説明

ルール:
- 作業内容が変わった時点で区間を分ける
- 動画開始から終了まで、可能な限り連続した区間になるようにする
- 判断できない場合は推測せず、descriptionに「判断困難」と記載する
""".strip()


class WorkSegment(BaseModel):
    """1つの作業区間"""

    start_time: str = Field(description="作業開始時刻（MM:SS形式）")
    end_time: str = Field(description="作業終了時刻（MM:SS形式）")
    task_name: str = Field(description="短く分かりやすい作業名")
    description: str = Field(description="作業内容の簡単な説明")


class WorkSegmentResult(BaseModel):
    """動画から抽出した作業区間の一覧"""

    segments: list[WorkSegment] = Field(description="作業区間の配列")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("エラー: 環境変数 GEMINI_API_KEY が設定されていません。")
        sys.exit(1)

    if not VIDEO_PATH.is_file():
        print(f"エラー: 動画ファイルが見つかりません: {VIDEO_PATH}")
        sys.exit(1)

    print(f"動画ファイル: {VIDEO_PATH}")
    print(f"モデル: {MODEL_NAME}")
    print("Files API で動画をアップロードしています...")

    client = genai.Client(api_key=api_key)

    # 1. Files API で動画をアップロード
    video_file = client.files.upload(file=str(VIDEO_PATH))
    print(f"アップロード開始: name={video_file.name}, state={video_file.state}")

    # 2. 処理完了（ACTIVE）まで待機
    while video_file.state == types.FileState.PROCESSING:
        print("動画を処理中です。しばらくお待ちください...")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state == types.FileState.FAILED:
        print(f"エラー: 動画の処理に失敗しました。state={video_file.state}")
        sys.exit(1)

    if video_file.state != types.FileState.ACTIVE:
        print(f"エラー: 予期しない状態です。state={video_file.state}")
        sys.exit(1)

    print("動画のアップロード・処理が完了しました。作業区間を抽出します...")

    # 3. Structured Output で作業区間を JSON として取得
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[video_file, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WorkSegmentResult,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    # 構造化結果を優先し、なければテキストの JSON を使う
    if response.parsed is not None:
        segments_data = [
            segment.model_dump() for segment in response.parsed.segments
        ]
    else:
        parsed_json = json.loads(response.text)
        segments_data = parsed_json.get("segments", parsed_json)

    print("\n========== 抽出した作業区間（JSON） ==========\n")
    print(json.dumps(segments_data, ensure_ascii=False, indent=2))
    print("\n==============================================")
    print(f"区間数: {len(segments_data)}")


if __name__ == "__main__":
    main()
