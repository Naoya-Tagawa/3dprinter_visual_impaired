import tkinter as tk
import threading
import time
global a
a = 0
def func1():
    global a
    while True:
        a += 1
        time.sleep(1)
        print(a)
        if a == 50:
            break

def func2():
    global a
    
    
    
    #print(a)
    if a == 10:
        func3()

def func3():
    print("成功!")
    exit()


if __name__ == "__main__":
    func1_thread = threading.Thread(target=func1)
    func2_thread = threading.Thread(target=func2)
    func2_thread.start()
    func1()