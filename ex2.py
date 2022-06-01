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


def check():
    global count
    if  5 <count <= 10:
        #event.clear() #赤信号にする
        print("\33[41;1m赤信号...\033[0m")
        return False
        #while car1.is_alive():
            #print("alive")
            #if event.is_set():
                #car1.terminate()
            #time.sleep(1)
        
    elif count > 10:
        #event.set() #青信号
        #car2.start()
        count = 0
        return True
    else:
        print("\33[42;1m青信号...\033[0m")
        return True
        


    #time.sleep(1)


def car():
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

def sayfunc(q,ph):
    engine = q.get()
    #engine = pyttsx3.init()
    #rate デフォルト値は200
    engine.say(ph)
    engine.runAndWait()
    q.put(False)


def say(ph):
    engine = pyttsx3.init()
    #rate デフォルト値は200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',200)
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)
    engine.say(ph)
    engine.runAndWait()


if __name__ == '__main__':
    q = multiprocessing.Queue()
    engine = pyttsx3.init()
    #engine = pyttsx3.init()
    #rate デフォルト値は200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',300)
    volume = engine.getProperty('volume')
    engine.setProperty('volume',3.0)
    #rate デフォルト値は200
    q.put(engine)
    while True:
        count +=1
        print(count)
        judge = check()
        
        
        if judge == True:
            st = q.get()
            print(st)
            #print(len(q))
            if st == engine:
                engine.stop()
                voice1.terminate()
            voice1 = multiprocessing.Process(target=sayfunc,args=(q,"あの客はよく柿食う客だ"))
            voice1.start()
            q.put(engine)
        else:
            st = q.get()
            if st == engine:
                engine.stop()
                voice1.terminate()
                
            voice1 = multiprocessing.Process(target=sayfunc,args=(q,"go"))
            voice1.start()
            q.put(engine)
        time.sleep(2)
#car = threading.Thread(target=car, args=("MINI",))
#car.start()