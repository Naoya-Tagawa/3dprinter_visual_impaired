from cmath import e
import pyttsx3
from numpy import char
import multiprocessing
import time

event = multiprocessing.Event()
count = 0

def lighter(q):
    '''
    flag=True: 青信号
    flag=False: 赤信号
    '''
    count = 0
    event.set()
    while True:
        q.put(count)
        count += 1
        time.sleep(1)

def car(q):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    global count
    while True:
        if q.get < 5:
            event.set()
        elif 5 <= q.get <10:
            event.clear()
        else:
            event.set()
            q.put(0)

def sayFunc(phrase):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(phrase)
    engine.runAndWait()

if __name__ == '__main__':
    q = multiprocessing.Queue()
    light = multiprocessing.Process(target=lighter,args=(q,))
    light.start()

    car = multiprocessing.Process(target=car, args=(q,))
    car.start()
    while True:
        if event.is_set():
            v = multiprocessing.Process(target=sayFunc,args=("blue",))
            v.start()
            while v.is_alive():
                if event.is_set == False:
                    v.terminate()
        else:
            event.wait()