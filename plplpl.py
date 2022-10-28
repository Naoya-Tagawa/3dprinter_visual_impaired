import ctypes
path = "./mymodule.cpython-310-x86_64-linux-gnu.so"
lib = ctypes.cdll.LoadLibrary(path)
print(lib.add(1,3))


