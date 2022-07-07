
import multiprocessing
import pyttsx3
from threading import Thread
from multiprocessing import process
import time
speed = 200
#ボリューム
vol = 1.0
engine = pyttsx3.init()
global output_text
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
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper
@threaded
def manage_process(p,flg1):
    while p.is_alive():
        if flg1.value == 1:
            print("stop")
            engine.stop()
            flg1.value = 0
        else:
            continue

def say(phrase,flg,output_text):
    while True:
        #engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty("voice",voices[1].id)
        #rateはデフォルトが200
        rate = engine.getProperty('rate')
        engine.setProperty('rate',speed)
        #volume デフォルトは1.0 設定は0.0~1.0
        volume = engine.getProperty('volume')
        engine.setProperty('volume',vol)
        count = 0
        text = output_text.get()
        for word in text:
            #st = flg.value
            if word == ' ':
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
 
            engine.say(word)
                #engine.say(word)
        #engine.connect('started-word', onWord("kk",st,10))
        engine.runAndWait()
        engine.stop()
        #engine.connect('started-word', onWord("kk",flg,10))
if __name__ == '__main__':
    engine = pyttsx3.init()
    q = ['The quick', 'is /very cute']
    flg1 = multiprocessing.Value('i',0)
    output_text = multiprocessing.Queue()
    
    #output_text = multiprocessing.Array('s',q)
    #whole_text_read(q)
    er = multiprocessing.Process(target=say,args=(q,flg1,output_text))
    #flg1 = False
    er.start()
    t = manage_process(er,flg1)
    p1,p2 =multiprocessing.Pipe()
    p1.send(1)
    print(p2.recv())
    p1.send(2)
    print(p2.recv())
#er = Thread(target=say,args=(q,))
#er = multiprocessing.Process(target=sy,args=('The quick brown fox jumped over the lazy dog.',))
#er.start()
#engine.say('The quick brown fox jumped over the lazy dog.')
#engine.connect('started-word', onWord("kk",20,10))
    print("sleep")
    flg1.value = 0
    output_text.put(q)
    output_text.put(q)
    flg1.value = int(input())
    time.sleep(5)
    #output_text.put(q)
    flg1.value = 1
    print(flg1.value)
    print("wake up")
    flg1.value = 0
    #output_text.put(q)
    #flg2.send(True)
    #flg.put(False)
#engine.runAndWait()
#engine.say('The quick brown fox jumped over the lazy dog.')
#engine.runAndWait()