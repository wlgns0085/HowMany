from threading import Thread
from mobilenet import *
from web import *

if __name__ == "__main__":
    init_detection()
    
    t = Thread(target = counting)
    t.start()
    
    app.run(threaded=True)
