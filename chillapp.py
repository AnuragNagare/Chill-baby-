"""
chillapp.py – Application entrypoint.

All logic lives in routes/ and services/.
This file only wires routes, configures SSL, and starts the server.
"""

import logging
import ssl
import argparse
from datetime import datetime

import aiohttp_jinja2
import jinja2
from aiohttp import web

import config  # noqa: F401 – loads .env at import time
import db
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

# GET / → show login screen (index.html) – no camera involved
@aiohttp_jinja2.template('index.html')
async def login_get(request):
    return {}


# POST /login → save user info, redirect to detection screen
async def login_post(request: web.Request) -> web.Response:
    data          = await request.post()
    parent_name   = data.get('parent_name', '').strip()
    email         = data.get('email', '').strip()
    company       = data.get('company', '').strip()
    intolerances  = [i.strip() for i in data.get('intolerances', '').split(',') if i.strip()]

    # Store in shared state so detection screen & processing routes can use them
    globalvars["intolerances"] = intolerances

    # Create the MongoDB session document upfront (camera starts later)
    new_session = {
        "name":         parent_name,
        "email":        email,
        "company":      company,
        "intolerances": intolerances,
        "started_at":   datetime.utcnow(),
        "video_link":   None,
    }
    try:
        result = await db.sessions().insert_one(new_session)
        globalvars["insertedId"] = result.inserted_id
        logging.getLogger(__name__).info(
            "Session created at login: id=%s user=%s", result.inserted_id, parent_name
        )
    except Exception:
        logging.getLogger(__name__).exception("Failed to insert session at login")

    # Redirect to the detection/process screen
    raise web.HTTPFound('/process')


# GET /process → main detection screen (camera opens here)
@aiohttp_jinja2.template('process.html')
async def process_get(request):
    return {'alert_msg': ''}


async def view(request):
    import os
    content = open(os.path.join(os.path.dirname(__file__), "templates", "view.html")).read()
    return web.Response(content_type="text/html", text=content)


async def favicon(request):
    raise web.HTTPFound('/static/favicon.svg')

app.router.add_get('/',            login_get)
app.router.add_post('/login',      login_post)
app.router.add_get('/process',     process_get)
app.router.add_get('/view',        view)
app.router.add_get('/favicon.ico', favicon)
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
    parser = argparse.ArgumentParser(description="Chill Baby AI – AI child monitoring server")
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
