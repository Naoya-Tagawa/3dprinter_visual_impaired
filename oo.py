def func(m,n):
    if n != 0:
        r = m%n
        m = n
        n= r
        func(m,n)
    return m

func(24,15)