import time

def tictoc(func):
    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time() - t1
        print(f'Tock {t2} seconds')
    return wrapper

@tictoc
def do_this():
    time.sleep(1.3)

@tictoc
def do_thas():
    time.sleep(0.4)


@tictoc
def x():
    v = 253
    z = 400
    a=0.00001
    t= v*z/a
    print(t)


do_this()
do_thas()
x()
print('Done')

