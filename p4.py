import sys
import os
import subprocess
 
 
def main():
     
    if len(sys.argv) < 2 or sys.argv[1] == "":
        return 1
    voice_file = sys.argv[1]
     
    code = speech(voice_file)
    return code
 
 
def speech(voice_file):
     
    if not os.path.exists(voice_file):
        return 2
     
    subprocess.call(["ffplay", "-nodisp", "-autoexit", voice_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
     
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())