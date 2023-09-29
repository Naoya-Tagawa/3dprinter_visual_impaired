
import time
import pyttsx3
from numpy import char
import multiprocessing
event = bool
count = 0
def lighter(q):

#flag=True: 青信号
#flag=False: 赤信号

    global count
    #event.set()  # 初期値は青信号
    while True:

        check(q)
        time.sleep(1)
        count += 1
        print(count)


def check(q):
    global count
    global event
    if  5 <count <= 10:
        #event = False #赤信号にする
        q.put("red")
        car1 = multiprocessing.Process(target=car, args=("とまれとまれとまれとまれとまれとまれとまれとまれとまれとまれとまれとまれとまれ",))
        car1.daemon = True
        car1.start()
        #time.sleep(1)
        if q.get() == "blue":
            print("chudan")
            car1.terminate()
        print("\33[41;1m赤信号...\033[0m")
    elif count > 10:
        event == True #青信号
        q.put("blue")
        print(q)
        #car2.start()
        count = 0
    else:
        print("\33[42;1m青信号...\033[0m")
        

    print("check")
    #time.sleep(1)
def car(name):
    engine = pyttsx3.init()

    #rate デフォルト値は200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',300)

    #volume デフォルト値は1.0、設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)
    engine.say(name)
    if engine.isBusy:
        print("True")
        engine.stop()
    engine.runAndWait()
    
    #engine.endLoop()

if __name__ == '__main__':
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=lighter,args=(q,))
    p.start()


#car = threading.Thread(target=car, args=("MINI",))
#car.start()