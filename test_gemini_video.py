"""
Gemini API 動画分析の単独テスト

プロジェクト内の test_video.mp4 を Files API でアップロードし、
処理完了後に Gemini Flash で内容を分析します。
APIキーは環境変数 GEMINI_API_KEY から読み込みます。
"""

import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types


# このスクリプトと同じフォルダにある動画を使う
VIDEO_PATH = Path(__file__).resolve().parent / "test_video.mp4"

# 現在利用可能な Gemini Flash 系モデル
MODEL_NAME = "gemini-3.6-flash"

PROMPT = (
    "この動画で作業者が行っている作業を、時系列に沿って初心者にも分かるように説明してください。\n"
    "作業内容が変化する場合は、その変化も説明してください。"
)


def main():
    # Windows のターミナルでも日本語・記号を表示できるようにする
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

    # 2. 処理が完了（ACTIVE）するまで待つ
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

    print("動画のアップロード・処理が完了しました。分析を開始します...")

    # 3. 動画を添付して質問する
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[video_file, PROMPT],
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    print("\n========== Gemini の回答 ==========\n")
    print(response.text)
    print("\n===================================")


if __name__ == "__main__":
    main()
