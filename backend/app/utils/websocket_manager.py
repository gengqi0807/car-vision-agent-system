from collections import defaultdict

from fastapi import WebSocket
from starlette.websockets import WebSocketState


class WebSocketManager:
    def __init__(self) -> None:
        self.connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections[channel].append(websocket)

    def disconnect(self, channel: str, websocket: WebSocket) -> None:
        if websocket in self.connections[channel]:
            self.connections[channel].remove(websocket)

    async def broadcast(self, channel: str, payload: dict) -> int:
        delivered = 0
        stale_connections: list[WebSocket] = []

        for websocket in list(self.connections[channel]):
            try:
                if websocket.application_state != WebSocketState.CONNECTED:
                    stale_connections.append(websocket)
                    continue

                await websocket.send_json(payload)
                delivered += 1
            except Exception:
                stale_connections.append(websocket)

        for websocket in stale_connections:
            self.disconnect(channel, websocket)

        return delivered


websocket_manager = WebSocketManager()
