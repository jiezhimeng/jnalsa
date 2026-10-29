"""
Copy from pywebio.platform.fastapi
"""
import os
import secrets
import re
from pathlib import Path

import uvicorn
from pywebio.platform.fastapi import (STATIC_PATH, Session, cdn_validation,
                                      get_free_port,
                                      open_webbrowser_on_server_started,
                                      start_remote_access_service,
                                      webio_routes)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles


CANAL_MAP_DIR = (
    Path(__file__).resolve().parents[2]
    / "assets" / "jiangnan" / "grand_canal" / "map"
)

INN_MAP_DIR = Path(__file__).resolve().parents[2] / 'assets' / 'jiangnan' / 'inn' / 'map'
INN_MAP_STATE_DIR = Path(__file__).resolve().parents[2] / 'log' / 'inn_map'


async def inn_map_asset(request):
    """Expose only bundled map files and isolated, read-only observations."""
    name = request.path_params['asset']
    if name in {'overview.html', 'viewer.js', 'viewer.css', 'panorama.png'}:
        path = INN_MAP_DIR / name
    elif re.fullmatch(r'state-[0-9a-f]{16}\.json', name):
        path = INN_MAP_STATE_DIR / name
    else:
        return PlainTextResponse('Map asset not found', status_code=404)
    if not path.is_file():
        return PlainTextResponse('Map asset not found', status_code=404)
    return FileResponse(path, headers={'Cache-Control': 'no-store'} if name.endswith('.json') else None)


async def canal_map_asset(request):
    """Serve the standalone Yangzhou canal map bundled with the project."""
    name = request.path_params["asset"]
    if name not in {"overview.html", "overview_preview.jpg", "map_tiles.js", "custom_route.js"} and not re.fullmatch(
        r"tiles/l(?:1|4)/\d+_\d+\.jpg", name
    ):
        return PlainTextResponse("Map asset not found", status_code=404)
    path = CANAL_MAP_DIR / name
    if not path.is_file():
        return PlainTextResponse("Map asset not found", status_code=404)
    return FileResponse(path)


class HeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/canal-map/") and request.url.path.endswith(".jpg") and response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=86400"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response


async def monitor_snapshot(request):
    """返回指定实例最近一次设备截图，由浏览器每秒刷新一次。"""
    # 延迟导入以避免 process_manager -> setting -> webui 的循环依赖。
    from module.webui.process_manager import ProcessManager

    config_name = request.path_params.get("config_name", "")
    manager = ProcessManager._processes.get(config_name)
    supplied_token = request.query_params.get("token", "")
    if (
        manager is None
        or not supplied_token
        or not secrets.compare_digest(supplied_token, manager.monitor_token)
    ):
        return PlainTextResponse("Monitor snapshot not found", status_code=404)

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "X-Content-Type-Options": "nosniff",
    }
    image = manager.latest_screenshot
    if not manager.monitor_enabled or not image:
        return Response(status_code=204, headers=headers)
    return Response(content=image, media_type="image/jpeg", headers=headers)


def asgi_app(
    applications,
    cdn=True,
    static_dir=None,
    debug=False,
    allowed_origins=None,
    check_origin=None,
    **starlette_settings
):
    debug = Session.debug = os.environ.get("PYWEBIO_DEBUG", debug)
    cdn = cdn_validation(cdn, "warn")
    if cdn is False:
        cdn = "pywebio_static"
    routes = [
        Route("/inn-map/{asset:path}", inn_map_asset),
        Route("/monitor/{config_name:str}.jpg", monitor_snapshot),
        Route("/canal-map/{asset:path}", canal_map_asset),
    ]
    routes.extend(webio_routes(
        applications,
        cdn=cdn,
        allowed_origins=allowed_origins,
        check_origin=check_origin,
    ))
    if static_dir:
        routes.append(
            Mount("/static", app=StaticFiles(directory=static_dir), name="static")
        )
    routes.append(
        Mount(
            "/pywebio_static",
            app=StaticFiles(directory=STATIC_PATH),
            name="pywebio_static",
        )
    )
    middleware = [Middleware(HeaderMiddleware)]
    return Starlette(
        routes=routes, middleware=middleware, debug=debug, **starlette_settings
    )


def start_server(
    applications,
    port=0,
    host="",
    cdn=True,
    static_dir=None,
    remote_access=False,
    debug=False,
    allowed_origins=None,
    check_origin=None,
    auto_open_webbrowser=False,
    **uvicorn_settings
):

    app = asgi_app(
        applications,
        cdn=cdn,
        static_dir=static_dir,
        debug=debug,
        allowed_origins=allowed_origins,
        check_origin=check_origin,
    )

    if auto_open_webbrowser:
        asyncio.get_event_loop().create_task(
            open_webbrowser_on_server_started("localhost", port)
        )

    if not host:
        host = "0.0.0.0"

    if port == 0:
        port = get_free_port()

    if remote_access:
        start_remote_access_service(local_port=port)

    uvicorn.run(app, host=host, port=port, **uvicorn_settings)
