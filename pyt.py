from dictionary_word import speling
import pyttsx3
import time
import difflib
def whole_text_read(text):
    engine = pyttsx3.init()
    #rateはデフォルトが200
    rate = engine.getProperty('rate')
    engine.setProperty('rate',150)
    #volume デフォルトは1.0 設定は0.0~1.0
    volume = engine.getProperty('volume')
    engine.setProperty('volume',1.0)
    count = 0
    engine.say(text)
    engine.runAndWait()
def kersol_exist_search(kersol,text): 
    text = text.splitlines()
    for word in text:
        s = difflib.SequenceMatcher(None,kersol,word)
        if s.ratio() >= 0.90:
            return True
    
    return False

sd = ""
s1 = 'Bed Level correct → \nPID calibration     →\nReset XYZ calibr. \n>Temp. calibration →\n'
s2 = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'Timeout: ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '10', '\n', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 
'', ' ', '\n']
kersol = ">Temp. calibration →"
print(kersol_exist_search(kersol,s1))
whole_text_read(s1)
res = difflib.context_diff(s1,s2)
print(speling.correct("+1Iament"))
for word in res:
    if word[0] == '!':
        sd += word[2:]
print(sd)
s3 = -1.4
print(float(s3))