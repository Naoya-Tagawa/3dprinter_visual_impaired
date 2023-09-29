#!/bin/bash
 
# 音声ファイルの名前を指定
VOICE_FILE="voice.wav"
 
# 古い音声ファイルが残っている？
if [ -f $VOICE_FILE ]; then
    # 古い音声ファイルを削除
    rm $VOICE_FILE
fi
 
# 音声ファイルを作成
./p6.py $VOICE_FILE
CODE=$?
if [ $CODE -ne 0 ]; then
    # 音声ファイル作成に失敗 -> エラーメッセージ
    echo 音声ファイルの作成に失敗 code=$CODE
    exit
fi
 
# 音声ファイル作成に成功 -> 再生
./p4.py $VOICE_FILE