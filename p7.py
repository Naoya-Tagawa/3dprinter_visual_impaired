import sys
import os
import datetime
 
import pyttsx3
import pygame.mixer
pygame.mixer.init()
#screen = pygame.display.set_mode((640, 480))
 
voice_file = "voice.wav"
message = "Hello, good to meet you."

pygame.mixer.music.load('C:\\Users\\naoya\\sample\\3dprinter_visual_impaired\\voice.wav')
pygame.mixer.music.play(1)
