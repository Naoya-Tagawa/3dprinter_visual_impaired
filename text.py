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