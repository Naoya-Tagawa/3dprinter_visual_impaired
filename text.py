from cmath import phase
import multiprocessing
import pyttsx3
import time
from threading import Thread

engine = pyttsx3.init()
phrase = "あおいくさいあおいくさいあおいくさいあおいくさいあおいくさいあおいくさいあおいくさいあおいくさい"
engine.say(phrase)
if engine.isBusy == True:
    print("busy")
else:
    print("暇")
engine.runAndWait()
if engine.isBusy == True:
    print("jj")
engine.stop()