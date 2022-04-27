import pyttsx3
import time
def text_read(text):
    engine = pyttsx3.init()
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',150)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)
    for word in text:
        if word == '\n':
            time.sleep(5)
        engine.say(word)
        engine.runAndWait()

s = ["hello",' ',"python",'\n','window']
engine = pyttsx3.init()
text_read(s)
#for word in s:
 #   if word == '\n':
  #      time.sleep(15)
   # engine.say(word)
    #engine.runAndWait()
#engine.runAndWait()