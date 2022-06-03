
import multiprocessing
import pyttsx3
import keyboard
import time
count =0
event = multiprocessing.Event()
def light():
#flag=True: 青信号
#flag=False: 赤信号
	global count
	event.set()
	if event.is_set():
		print("True")
	else:
		print("false")
	#print(event)  # 初期値は青信号
	while True:
		count += 1
		print(count)
		if count == 5:
			print("sdayo")
			event.clear()
		if count == 6:
			event.set()
		time.sleep(1)


def sayFunc(phrase):
	engine = pyttsx3.init()
	engine.setProperty('rate',160)
	while True:
		if event.is_set():
			engine.say(phrase)
			engine.runAndWait()
			time.sleep(1)
		else:
			print("false")
			if not engine.isBusy():
				print("話し中")
				engine.stop()
			event.wait()

def say(phrase):
	if __name__ == "__main__":
		p = multiprocessing.Process(target=light,)
		p.start()

		p2 = multiprocessing.Process(target=sayFunc,args=("blueblueblueblueblueblueblueblueblueblueblueblueblueblueblue",))
		p2.start()


		#while p.is_alive():
			#if keyboard.is_pressed('q'):
				#p.terminate()
			#else:
				#continue
		p.join()

say("this process is running right now")