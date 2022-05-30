import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import difflib
import time
import cv2
import numpy as np
import glob
from natsort import natsorted
import threading
from PIL import Image , ImageTk , ImageOps
import pyttsx3 
from dictionary_word import speling
import difflib
import numpy as np
import cv2
import matplotlib.pyplot as plt
#話すスピード
speed = 300
#ボリューム
vol = 1.0
#カーソルの表示を探す
def kersol_search(text):
    i = 0
    kersol1 = ""
    for word in text:
        if word[0] == ">":
            i = 1
            kersol1 += word + ' '
        elif (i == 1) & (word == '\n'):
            i = 0
        elif i == 1:
            kersol1 += word
    return kersol1
def cusor_search(text):
    for word in text:
        
        try:
            if ">" in word[0]:
                return word
        except: IndexError
        continue
    return []
#カーソルの位置をいう
def kersol_read(text):
    engine = pyttsx3.init()
    voice = engine.getProperty('voices')
    engine.setProperty("voice",voice[1].id)
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    count = 0
    engine.say("The current cursor position is")
    for word in text:
        if word == ' ':
            continue
        if '/' in word:
            target = '/'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("スラッシュ")
            r = word[idx:]
            engine.say(r)
            continue
        if ',' in word:
            target = ','
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("カンマ")
            r = word[idx:]
            engine.say(r)
            continue
        if "." in word:
            target = '.'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("ドット")
            r = word[idx:]
            engine.say(word)
            continue
        engine.say(word)
    
    engine.runAndWait()

#テキスト全部読み上げ
def whole_text_read(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice",voices[1].id)
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    count = 0
    for word in text:
        if word == ' ':
            continue
        if '/' in word:
            target = '/'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("スラッシュ")
            r = word[idx:]
            engine.say(r)
            continue
        if ',' in word:
            target = ','
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("カンマ")
            r = word[idx:]
            engine.say(r)
            continue
        if "." in word:
            target = '.'
            idx = word.find(target)
            r = word[:idx]
            engine.say(r)
            engine.say("ドット")
            r = word[idx:]
            engine.say(r)
            continue
        engine.say(word)
    
    engine.runAndWait()

#カーソルがテキストにあるか
def kersol_exist_search(kersol,text):
    text = text.splitlines()
    for word in text:
        s = difflib.SequenceMatcher(None,kersol[1:],word)
        if s.ratio() >= 0.90:
            return True
    return False

def partial_text_read(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice",voices[1].id)
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',speed)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',vol)
    engine.say(text)
    engine.runAndWait()