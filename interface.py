# example run:
# $ python3 interface.py 0

from flask import Flask, render_template, Response
from waitress import serve

from queue import Queue
from threading import Thread
from modules.config.config import config

from inference import serveProcessingAI, serveStreaming

stream_host = config["stream_host"]
stream_port = config["stream_port"]
stream_endpoint = config["stream_endpoint"]

app = Flask(__name__)
@app.route(stream_endpoint, methods=['GET'])
def get_image():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    """Webservice for request via GET method."""
    return Response(serveStreaming(qe), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    qe = Queue()
    # thread 1: background task
    frame_queue = 3 # 3 frame in buffer
    ai_process = Thread(target=serveProcessingAI, args=(qe, frame_queue), daemon=True)
    ai_process.start()
    # thread 2
    print(f'Open http://{stream_host}:{stream_port}{stream_endpoint} in browser...')
    serve(app, host = stream_host, port = stream_port, threads = 6)
    qe.join()