#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyaudio
import wave
import threading
from time import sleep

CHUNK = 1024


class AudioPlayer(threading.Thread):
    """ A Class For Playing Audio """

    def __init__(self, audio_file):
        super(AudioPlayer, self).__init__()
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
        self.playing.set()
        self.start()

    def stop(self):
        """ Stop playing audio and wait until the sub-thread terminates. """
        if self.playing.is_set():
            self.playing.clear()
            self.join()


if __name__ == "__main__":
    player1 = AudioPlayer("voice.wav")
    player2 = AudioPlayer("sample.mp3")

    player1.play()
    # 例えば0,5秒後に別の音源に変える
    sleep(0.5)
    # 1を止めて
    player1.stop()
    # 2を再生
    print("jj")
    player2.play()