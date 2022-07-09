import multiprocessing
import pyttsx3
import time
from threading import Thread


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

	
def say(phrase):
	global t
	global term
	term = False
	p = multiprocessing.Process(target=speak, args=(phrase,))
	p.start()
	t = manage_process(p)
		
if __name__ == "__main__":
    print("ss")
    say("this process is kindergarden")
    print("hh")
    time.sleep(1)
    stop_speaker()
    time.sleep(5)