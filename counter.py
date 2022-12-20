from threading import Thread
from mobilenet import *
from web import *



if __name__ == "__main__":

    init_detection()
    
    t = Thread(target = counting)
    t.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True)
