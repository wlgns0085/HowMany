from flask import Flask, render_template, Response

# init app
app = Flask(__name__)

#To replace with real datas
global data
data = [
	'''
	("17-12-2022", 60),
	("16-12-2022", 100),
	("15-12-2022", 85),
	("14-12-2022", 90),
	("13-12-2022", 110),
	("12-12-2022", 60),
	("11-12-2022", 45)
	'''
]

# Route
@app.route('/')
def index():

	dates = [row[0] for row in data]
	counting = [row[1] for row in data]

	return render_template("graph.html", labels = dates, values = counting)

# Route
@app.route('/live')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    camera = cv2.VideoCapture('http://192.168.0.11:8090/?action=stream')
    
    while True:
        ret, img = camera.read()

        if ret:
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break
