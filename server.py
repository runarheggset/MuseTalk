from http.server import BaseHTTPRequestHandler, HTTPServer
from omegaconf import OmegaConf
from urllib.parse import urlparse, parse_qsl
import argparse
import numpy as np
import uuid
import os

from scripts.realtime_inference import Avatar;

avatars: dict[str, Avatar] = {}

class MuseTalkServer(BaseHTTPRequestHandler):

    def handle_inference(self, query_params: dict[str, str]):
        avatar_id = query_params.get("avatar_id", "yongen")

        if avatars.get(avatar_id) is None:
            avatars[avatar_id] = Avatar(
                avatar_id = avatar_id, 
                video_path = "data/video/" + avatar_id + ".mp4", 
                bbox_shift = self.bbox_shift,
                batch_size = self.batch_size,
                preparation = True,
            )
        else:
            avatars[avatar_id].preparation = False

        avatar = avatars[avatar_id]

        length = int(self.headers.get("content-length"))
        audio_data = self.rfile.read(length)

        random = str(uuid.uuid4())

        with open("data/audio/"+random+".wav", "wb") as f:
            f.write(audio_data)

        self.send_response(200)
        self.end_headers()

        avatar.inference("data/audio/"+random+".wav", None, self.fps, False, self.wfile)

        os.remove("data/audio/"+random+".wav")

    def handle_create_avatar(self, query_params: dict[str, str]):
        avatar_id = query_params.get("avatar_id", "yongen")

        if avatars.get(avatar_id) is not None:
            self.send_response(400)
            self.end_headers()
            return

        length = int(self.headers.get("content-length"))
        video_data = self.rfile.read(length)

        with open("data/video/"+avatar_id+".mp4", "wb") as f:
            f.write(video_data)

        avatars[avatar_id] = Avatar(
            avatar_id = avatar_id,
            video_path = "data/video/" + avatar_id + ".mp4",
            bbox_shift = self.bbox_shift,
            batch_size = self.batch_size,
            preparation = True,
        )

        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        query_params = dict(parse_qsl(parsed.query))

        match parsed.path:
            case "/inference":
                self.handle_inference(query_params)
            case "/create_avatar":
                self.handle_create_avatar(query_params)
            case _:
                self.send_response(404)
                self.end_headers()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",
                        type=int,
                        default=10,
    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
    )
    parser.add_argument("--bbox_shift",
                        type=int,
                        default=5,
    )
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
    )
    parser.add_argument("--port",
                        type=int,
                        default=8080,
    )

    args = parser.parse_args()

    MuseTalkServer.fps = args.fps
    MuseTalkServer.batch_size = args.batch_size
    MuseTalkServer.bbox_shift = args.bbox_shift

    webServer = HTTPServer((args.host, args.port), MuseTalkServer)
    print("Server started http://%s:%s" % (args.host, args.port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
