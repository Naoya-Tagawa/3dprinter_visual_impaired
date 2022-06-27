
import multiprocessing
import pyttsx3
import time
from threading import Thread

#from audio_output import whole_text_read
vol =1.0
speed =300

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def speak(phrase):
    engine = pyttsx3.init()
    engine.say(phrase)
    engine.runAndWait()
    engine.stop()

def stop_speaker():
    global term
    term = True
    t.join()

@threaded
def manage_process(p):
	global term
	while p.is_alive():
		if term:
			p.terminate()
			term = False
		else:
			continue

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
    engine.stop()
def say(phrase):
	global t
	global term
	term = False
	p = multiprocessing.Process(target=whole_text_read, args=(phrase,))
	p.start()
	t = manage_process(p)
		
if __name__ == "__main__":
    start = time.perf_counter()
    txt = ["this process is running right now"]
    say(txt)
    end = time.perf_counter()
    print(end-start)
    time.sleep(1)
    stop_speaker()
    end = time.perf_counter()
    print(end-start)
    say(txt)
    time.sleep(1.5)
    stop_speaker()