<<<<<<< HEAD
<<<<<<< HEAD
from cmath import phase
import multiprocessing
import pyttsx3
import time
from threading import Thread
import pygame
engine = pyttsx3.init()
phrase = "あおいくさいあおいくさいあおいくさいあおいくさいあおいくさいあおいくさいあおいくさいあおいくさい"
engine.save_to_file(phrase, 'test.mp3')
#engine.runAndWait()
pygame.mixer.init(frequency = 44100)    # 初期設定
pygame.mixer.music.load("sample.mp3")     # 音楽ファイルの読み込み
pygame.mixer.music.play()
if engine.isBusy == True:
    print("busy")
else:
    print("暇")
engine.runAndWait()
if engine.isBusy == True:
    print("jj")
engine.stop()
=======
import subprocess
subprocess.Popen('aplay C:\\Users\\naoya\\3dprinter_visual_impaired\\voice.wav',shell=True)
#subprocess.run(['aplay', "C:\\Users\\naoya\\3dprinter_visual_impaired\\sample.mp3"],shell=True)
>>>>>>> bf09c8666cb39ead9d2842cc7c484ef1083bf4ca
=======
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyaudio
import wave
import threading
from time import sleep

CHUNK = 1024


class AudioPlayer(object):
    """ A Class For Playing Audio """

    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.playing = threading.Event()    # 再生中フラグ

    def run(self):
        """ Play audio in a sub-thread """
        audio = pyaudio.PyAudio()
        input = wave.open(self.audio_file, "rb")
        output = audio.open(format=audio.get_format_from_width(input.getsampwidth()),
                            channels=input.getnchannels(),
                            rate=input.getframerate(),
                            output=True)

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
        """ Play audio. """
        if not self.playing.is_set():
            self.playing.set()
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    def wait(self):
        if self.playing.is_set():
            self.thread.join()

    def stop(self):
        """ Stop playing audio and wait until the sub-thread terminates. """
        if self.playing.is_set():
            self.playing.clear()
            self.thread.join()


if __name__ == "__main__":
    player1 = AudioPlayer("voice.wav")
    player2 = AudioPlayer("ki.wav")

    player1.play()
    # 例えば0,5秒後に別の音源に変える
    sleep(0.5)
    # 1を止めて
    player1.stop()
    # 2を再生
    player2.play()
    player2.wait()
    # もう一度1を再生
    player1.play()
>>>>>>> f1eb15826c2ab34b2373b455cba875a9a45f4215
