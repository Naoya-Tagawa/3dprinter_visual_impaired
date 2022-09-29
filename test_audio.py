from cmath import phase
import multiprocessing
import pyttsx3
import time
from threading import Thread

count = 0


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

def speak(q):
    global count 
    #print(count)
    print("プロセス")
    engine = pyttsx3.init()
    phrase = q.get()
    print(phrase)
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
            print("終了")
        else:
            continue

	
def say(q):
    global t
    global term
    term = False
    p = multiprocessing.Process(target=speak, args=(q,))
    p.start()
    t = manage_process(p)


if __name__ == "__main__":
    q = multiprocessing.Queue()
    #for i in range(10):
     #   say(q)
    #time.sleep(1.5)
    while True:
        say(q)
        count += 1
        print(count)
        if  5 <count <= 10:
            #event.clear() #赤信号にする
            #stop_speaker()
            q.put("とまれ")
            #stop_speaker()
            #while car1.is_alive():
            #print("alive")
            #if event.is_set():
                #car1.terminate()
            #time.sleep(1)
        
            
            print("\33[41;1m赤信号...\033[0m")
        elif count > 10:
            #event.set() #青信号
            #stop_speaker()
            
            q.put("すすめすすめすすめすすめすすめ")

            #stop_speaker()
            #car2.start()
            count = 0
            break
        else:
            print("\33[42;1m青信号...\033[0m")
    stop_speaker()
    time.sleep(1.5)
    q.put("まって")
    stop_speaker()
	#say("this process is running right now")
	#time.sleep(1)
	#stop_speaker()
	#say("this process is running right now")
	#time.sleep(1.5)
	#stop_speaker()