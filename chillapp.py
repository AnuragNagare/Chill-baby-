"""
chillapp.py – Application entrypoint.

All logic lives in routes/ and services/.
This file only wires routes, configures SSL, and starts the server.
"""

import logging
import ssl
import argparse

import aiohttp_jinja2
import jinja2
from aiohttp import web

import config  # noqa: F401 – loads .env at import time
from routes import webrtc, video, websocket, processing

# ── Shared mutable state ─────────────────────────────────────────────────────
connections: dict = {}          # { user_id: WebSocketResponse }
globalvars: dict  = {
    "processing":   False,
    "intolerances": [],
    "mainFood":     "",
    "filepath":     "",
    "filename":     "",
    "insertedId":   "",
    "video_url":    "",
    "alert_msg":    "",
    "processed":    False,
}

# ── App setup ────────────────────────────────────────────────────────────────
app = web.Application()
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))

# ── Template routes ──────────────────────────────────────────────────────────
@aiohttp_jinja2.template('process.html')
async def index(request):
    return {'alert_msg': ''}

async def view(request):
    import os
    content = open(os.path.join(os.path.dirname(__file__), "templates", "view.html")).read()
    return web.Response(content_type="text/html", text=content)

app.router.add_get('/', index)
app.router.add_get('/view', view)
app.router.add_static('/static/', path='./static', name='static')

# ── Module routes ────────────────────────────────────────────────────────────
webrtc.setup_routes(app, connections, globalvars)
video.setup_routes(app, connections, globalvars)
websocket.setup_routes(app, connections, globalvars)
processing.setup_routes(app, connections, globalvars)

# ── Shutdown hook ────────────────────────────────────────────────────────────
app.on_shutdown.append(webrtc.on_shutdown)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cammy – AI child monitoring server")
    parser.add_argument("--cert-file", default="cert.pem")
    parser.add_argument("--key-file",  default="key.pem")
    parser.add_argument("--host",      default="0.0.0.0")
    parser.add_argument("--port",      type=int, default=8080)
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ssl_context = None
    if args.cert_file and args.key_file:
        import os
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        ssl_context.load_cert_chain(
            os.path.join(os.getcwd(), args.cert_file),
            os.path.join(os.getcwd(), args.key_file),
        )

    web.run_app(app, access_log=None, host=args.host,
                port=args.port, ssl_context=ssl_context)
