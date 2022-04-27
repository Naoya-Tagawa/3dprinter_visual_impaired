from ntpath import join
import pyttsx3
import time
import difflib
sd = ""
s1 = ['bacon\n', 'eggs\n', 'ham\n', 'guido\n']
s2 = ['python\n', 'eggy\n', 'hamster\n', 'guido\n']
res = difflib.context_diff(s1,s2)
for word in res:
    if word[0] == '!':
        sd += word[2:]
print(sd)
s3 = -1.4
print(float(s3))