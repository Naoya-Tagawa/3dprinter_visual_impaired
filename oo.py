#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import subprocess

class AudioPlayer:
    """ A Class For Playing Audio """

    def __init__(self):
        self.audio_file = ""
        self.is_playing = False

    def setAudioFile(self, audio_file):
        self.audio_file = audio_file

    def playAudio(self):
        if(self.audio_file == ""):
            return
        cmd = 'mpc stop'
        cmd = "ls -l"
        import subprocess
        res = subprocess.run("echo %date%", stdout=subprocess.PIPE, 
                     shell=True, encoding="shift-jis")
        print ('play ' + self.audio_file)
        subprocess.call(cmd.split(),shell=True)
        #subprocess.call(cmd ,shell=True)   #音声停止
        #subprocess.call(['mpc stop'],shell=True)
        #subprocess.call(["mpc", "clear"],shell=True)  #プレイリストのクリア 
        #subprocess.call(["mpc", "update"],shell=True) #音声ファイルの読み込み
        #subprocess.call(["mpc", "add", self.audio_file],shell=True) #プレイリストに追加
        #subprocess.call('mpc play',shell=True) #再生
v = AudioPlayer()
v.setAudioFile("C:\\Users\\naoya\\3dprinter_visual_impaired\\voice.wav")
v.playAudio()
v.setAudioFile("C:\\Users\\naoya\\3dprinter_visual_impaired\\sample.mp3")
v.playAudio()