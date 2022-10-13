
import multiprocessing
import pyttsx3
import os
from operator import itemgetter
import datetime
import pyaudio
import wave
import threading
from time import sleep
import time
CHUNK = 1024
def make_voice_file(text):
    engine = pyttsx3.init()
    path = "./voice/"
    now = str(datetime.datetime.now())
    now_day , now_time = now.split()
    dh,m,s = now.split(':')
    sec , msec = s.split('.')
    now_time = sec + msec
    file_name = path + "voice_" + now_time + ".wav"
    print(file_name)
    engine.save_to_file(text,file_name)
    engine.runAndWait()

def delete_voice_file():
    file_list = []
    path = "./voice/"
    for file in os.listdir("./voice"):
        base , ext = os.path.splitext(file)
        if ext == '.wav':
            wav_file = path + file
            file_list.append([file,os.path.getctime(wav_file)])
    file_list.sort(key = itemgetter(1),reverse=True)
    print(file_list)
    max_file = 3
    for i , file in enumerate(file_list):
        if i > max_file -1:
            print("{}は削除します".format(file[0]))
            wav_file = path + file[0]
            os.remove(wav_file)


class AudioPlayer(object):
    """ A Class For Playing Audio """

    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.playing = threading.Event()    # 再生中フラグ

    def run(self):
        """ Play audio in a sub-thread """
        audio = pyaudio.PyAudio()
        input = wave.open(self.audio_file, "rb")
        output = audio.open(format=audio.get_format_from_width(input.getsampwidth()),
                            channels=input.getnchannels(),
                            rate=input.getframerate(),
                            output=True)

        while self.playing.is_set():
            data = input.readframes(CHUNK)
            if len(data) > 0:
                # play audio
                output.write(data)
            else:
                # end playing audio
                self.playing.clear()

        # stop and close the output stream
        output.stop_stream()
        output.close()
        # close the input file
        input.close()
        # close the PyAudio
        audio.terminate()

    def play(self):
        """ Play audio. """
        if not self.playing.is_set():
            self.playing.set()
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    def wait(self):
        if self.playing.is_set():
            self.thread.join()

    def stop(self):
        """ Stop playing audio and wait until the sub-thread terminates. """
        if self.playing.is_set():
            self.playing.clear()
            self.thread.join()

def latest_play_voice_file():
    file_list = []
    path = "./voice/"
    for file in os.listdir("./voice"):
        base , ext = os.path.splitext(file)
        if ext == '.wav':
            wav_file = path + file
            file_list.append([file,os.path.getctime(wav_file)])
    file_list.sort(key = itemgetter(1),reverse=True)
    return path + file_list[0][0]

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