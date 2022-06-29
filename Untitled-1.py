
import multiprocessing
import pyttsx3
from threading import Thread
from multiprocessing import process
import time
speed = 300
#ボリューム
vol = 1.0
engine = pyttsx3.init()
from audio_output import whole_text_read
def onWord(name, location, length):
    print('word', name, location, length)
    if location == 1:
        print("stop")
        engine.stop()
def sy(phrase):
    global flg
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("voice",voices[1].id)
    engine.say(phrase)
    engine.connect('started-word', onWord("kk",flg,10))
    engine.runAndWait()
    
def say(phrase,flg):
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
    #st = flg.get()
    for word in phrase:
        st = flg.value
        print(st)
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
        engine.connect('started-word', onWord("kk",st,10))
    engine.runAndWait()
    engine.stop()
        #engine.connect('started-word', onWord("kk",flg,10))
if __name__ == '__main__':
    engine = pyttsx3.init()
    flg1 = multiprocessing.Value('i',0)
    q = ['The','quick','is','very','cute']
    #whole_text_read(q)
    er = multiprocessing.Process(target=say,args=(q,flg1))
    #flg1 = False
    er.start()
#er = Thread(target=say,args=(q,))
#er = multiprocessing.Process(target=sy,args=('The quick brown fox jumped over the lazy dog.',))
#er.start()
#engine.say('The quick brown fox jumped over the lazy dog.')
#engine.connect('started-word', onWord("kk",20,10))
    print("sleep")
    time.sleep(1)
    flg1.value = 1
    print(flg1.value)
    print("wake up")
    flg1.value = 1
    #flg2.send(True)
    #flg.put(False)
#engine.runAndWait()
#engine.say('The quick brown fox jumped over the lazy dog.')
#engine.runAndWait()