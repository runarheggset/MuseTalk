from http.server import BaseHTTPRequestHandler, HTTPServer
from omegaconf import OmegaConf
from urllib.parse import urlparse, parse_qsl
import numpy as np

from scripts.realtime_inference import Avatar;

hostName = "0.0.0.0"
serverPort = 8080

batch_size = 8
bbox_shift = 5
fps = 25

skip_save_images = False
preparation = False

avatars: dict[str, Avatar] = {}

class MuseTalkServer(BaseHTTPRequestHandler):

    def do_POST(self):
        parsed = urlparse(self.path)
        query_params = dict(parse_qsl(parsed.query))

        avatar_id = query_params.get("avatar_id", "yongen")

        if avatars.get(avatar_id) is None:
            avatars[avatar_id] = Avatar(
                avatar_id = avatar_id, 
                video_path = "data/video/" + avatar_id + ".mp4", 
                bbox_shift = bbox_shift, 
                batch_size = batch_size,
                preparation = True,
            )
        else:
            avatars[avatar_id].preparation = False

        avatar = avatars[avatar_id]

        length = int(self.headers.get("content-length"))
        audio_data = self.rfile.read(length)

        buf = np.frombuffer(audio_data, np.int16).flatten().astype(np.float32) / 32768.0

        self.send_response(200)
        self.end_headers()

        avatar.inference(buf, None, fps, skip_save_images, self.wfile)

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MuseTalkServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
