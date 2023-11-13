import datetime
import pyttsx3
import os
from operator import itemgetter
import pyaudio
import wave
import threading
import time


CHUNK = 1024
# 話すスピード
speed = 300
# ボリューム
vol = 10.0
path = "VoiceFile/"


class AudioPlayer(object):  # 音声ファイルを再生、停止する
    """A Class For Playing Audio"""

    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.playing = threading.Event()  # 再生中フラグ

    def run(self):
        """Play audio in a sub-thread"""
        audio = pyaudio.PyAudio()
        input = wave.open(self.audio_file, "rb")
        output = audio.open(
            format=audio.get_format_from_width(input.getsampwidth()),
            channels=input.getnchannels(),
            rate=input.getframerate(),
            output=True,
        )

        while self.playing.is_set():
            data = input.readframes(CHUNK)
            if len(data) > 0:
                # play audio
                output.write(data)
            else:
                # end playing audio
                self.playing.clear()

        # stop and close the output stream
        output.stop_stream()
        output.close()
        # close the input file
        input.close()
        # close the PyAudio
        audio.terminate()

    def play(self):
        """Play audio."""
        if not self.playing.is_set():
            self.playing.set()
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    def wait(self):
        if self.playing.is_set():
            self.thread.join()

    def stop(self):
        """Stop playing audio and wait until the sub-thread terminates."""
        if self.playing.is_set():
            self.playing.clear()
            self.thread.join()


def make_voice_file(text):  # 音声ファイル作成
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    now = str(datetime.datetime.now())
    now_day, now_time = now.split()
    dh, m, s = now.split(":")
    sec, msec = s.split(".")
    now_time = sec + msec
    file_name = path + "voice_" + now_time + ".wav"
    # print(file_name)
    engine.save_to_file(text, file_name)
    engine.runAndWait()


def delete_voice_file():  # 音声ファイルを5つになるまで削除
    file_list = []
    for file in os.listdir(path):
        base, ext = os.path.splitext(file)
        if ext == ".wav":
            wav_file = path + file
            file_list.append([file, os.path.getctime(wav_file)])
    file_list.sort(key=itemgetter(1), reverse=True)
    # print(file_list)
    max_file = 3
    for i, file in enumerate(file_list):
        if i > max_file - 1:
            wav_file = path + file[0]
            os.remove(wav_file)


def latest_play_voice_file():  # 最新の音声ファイルを返す
    file_list = []
    for file in os.listdir(path):
        base, ext = os.path.splitext(file)
        if ext == ".wav":
            wav_file = path + file
            file_list.append([file, os.path.getctime(wav_file)])
    file_list.sort(key=itemgetter(1), reverse=True)
    return path + file_list[0][0]


def text_read(output_text):
    start = 0
    Interrupted_count = 0
    while True:
        text = output_text.get()
        # cv2.imwrite("real.jpg",img)
        # print("queu size :{0}".format(output_text.qsize()))
        if output_text.qsize() >= 1:
            while output_text.qsize() > 1:
                text = output_text.get()
                # cv2.imwrite("real.jpg",img)

        # out = match_text3(img_temp,label_temp,img)
        print(text)
        make_voice_file(text)
        file_name = latest_play_voice_file()
        if start != 0:
            player.stop()
            Interrupted_count += 1
        player = AudioPlayer(file_name)
        player.play()
        if start == 5:
            delete_voice_file()
            start = 1
        start += 1
        # time.sleep(0.1)


# def text_read_input(output_text):
#     last_text = ""
#     insert_count = 0  # 1秒間に挿入された回数
#     last_insert_time = time.time()
#     while True:
#         try:
#             # Empty the queue and keep only the last value
#             while True:
#                 last_text = output_text.get_nowait()
#                 last_insert_time = time.time()
#                 insert_count += 1
#         except output_text.Empty:
#             pass

#         # Check if there were more than 5 inserts in the last second
#         current_time = time.time()
#         print(current_time - last_insert_time)
#         if current_time - last_insert_time >= 1 and insert_count >= 5 and last_text:
#             print("Text to read:", last_text)
#             make_voice_file(last_text)
#             file_name = latest_play_voice_file()
#             player = AudioPlayer(file_name)
#             player.play()
#             delete_voice_file()

#             # Reset the insert count and last insert time
#             insert_count = 0
#             last_insert_time = current_time
