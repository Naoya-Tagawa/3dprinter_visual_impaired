import threading
import time
import pyttsx3
from numpy import char
import multiprocessing
import os
event = multiprocessing.Event()
count = 0
import os
from operator import itemgetter
import datetime
import pyaudio
import wave
CHUNK = 1024
def make_voice_file(text):
    start = time.perf_counter()
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
    end = time.perf_counter()
    time = end -start
    print(time)
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
    i = 0
    while True:
        st = q.get()
        if st == True:
            if i != 0:
                player.stop()
            #player.stop()
            make_voice_file("とまれとまれとまれとまれとまれとまれとまれとまれとまれとまれとまれ")
            file_name = latest_play_voice_file()
            player = AudioPlayer(file_name)
            player.play()
            i = 1
        else:
            player.stop()
            make_voice_file("すすめ")
            file_name = latest_play_voice_file() 
            player = AudioPlayer(file_name)
            player.play()
            print("赤から青")
            delete_voice_file()
            time.sleep(1)
        #time.sleep(1)
if __name__ == '__main__':
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=lighter,args=(q,))
    p.start()
    p1 = multiprocessing.Process(target=car,args=(q,))
    p1.start()
    print(q.get())
#car = threading.Thread(target=car, args=("MINI",))
#car.start()