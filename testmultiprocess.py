import threading
import time
import pyttsx3
from numpy import char
import multiprocessing
import os
event = multiprocessing.Event()
count = 0
def lighter(q):

#flag=True: 青信号
#flag=False: 赤信号

    global count
    #event.set()  # 初期値は青信号
    q.put(False)
    while True:

        
        count += 1
        print(count)
        check(q)
        time.sleep(1)


def check(q):
    global count
    if  5 <count <= 10:
        #event.clear() #赤信号にする
        q.put(True)
        #while car1.is_alive():
            #print("alive")
            #if event.is_set():
                #car1.terminate()
            #time.sleep(1)
        
            
        print("\33[41;1m赤信号...\033[0m")
    elif count > 10:
        #event.set() #青信号
        q.put(False)
        #car2.start()
        count = 0
    else:
        print("\33[42;1m青信号...\033[0m")
        


    #time.sleep(1)


def car(q):
    engine = pyttsx3.init()
    #rate デフォルト値は200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',300)

    #volume デフォルト値は1.0、設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)
    while True:
        st = q.get()
        if st == True:
            p2 = multiprocessing.Process(target=sayfunc,args=("とまれとまれとまれとまれとまれとまれとまれとまれとまれとまれとまれ",))
            p2.start()
        else:
            p2 = multiprocessing.Process(target=sayfunc,args=("すすめ",))
            p2.start()
            print("赤から青")
        print(st)
        time.sleep(1)
    engine.say(name)
    engine.runAndWait()
    
    #engine.endLoop()
def sayfunc(phrase):
    engine = pyttsx3.init()
    #rate デフォルト値は200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',300)
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)
    if engine.isBusy == False:
        print("jj")
        engine.stop()
        engine.say(phrase)
        engine.runAndWait()
    else:
        engine.say(phrase)
        engine.runAndWait()

if __name__ == '__main__':
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=lighter,args=(q,))
    p.start()
    p1 = multiprocessing.Process(target=car,args=(q,))
    p1.start()
    print(q.get())
#car = threading.Thread(target=car, args=("MINI",))
#car.start()