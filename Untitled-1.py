
import pyttsx3
from threading import Thread

def onWord(name, location, length):
    print('word', name, location, length)
    if location > 10:
        engine.stop()

def say(phrase):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate*2.0)
    engine.say(phrase)
    engine.runAndWait()

engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 160)
er = Thread(target=say,args=('The quick is very cute',))
er.start()
#engine.say('The quick brown fox jumped over the lazy dog.')
#engine.connect('started-word', onWord("kk",20,10))
onWord("kk",20,10)
#engine.say('The quick brown fox jumped over the lazy dog.')
#engine.runAndWait()