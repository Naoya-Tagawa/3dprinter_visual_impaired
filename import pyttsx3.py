import pyttsx3
from multiprocessing import Process
import keyboard

def speakfunc(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 140)
    engine.say(audio)
    engine.runAndWait()


def speak(audio):
    if __name__ == '__main__':
        p = Process(target=speakfunc, args=(audio,))
        p.start()

        while p.is_alive():
            if keyboard.is_pressed('q'):
                p.terminate()
            else:
                continue
        p.join()

def stop():
    p.terminate()

speak("隣の客はよく柿食う客だ")
stop()
speak("隣の柿はよく客食う柿だ")
