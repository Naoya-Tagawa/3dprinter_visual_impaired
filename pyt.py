from ntpath import join
import pyttsx3
import time
import difflib
def kersol_exist_search(kersol,text): 
    for word in text:
        s = difflib.SequenceMatcher(None,kersol,word)
        print(s.ratio())
        if s.ratio() >= 0.90:
            return True
    
    return False

sd = ""
s1 = ['bacon\n', 'eggs\n', 'ham\n', 'guido\n']
s2 = ['python\n', 'eggy\n', 'hamster\n', 'guido\n']
kersol = ""
print(kersol_exist_search(kersol,s1))

res = difflib.context_diff(s1,s2)
for word in res:
    if word[0] == '!':
        sd += word[2:]
print(sd)
s3 = -1.4
print(float(s3))