from collections import defaultdict

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self.connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections[channel].append(websocket)

    def disconnect(self, channel: str, websocket: WebSocket) -> None:
        if websocket in self.connections[channel]:
            self.connections[channel].remove(websocket)

    async def broadcast(self, channel: str, payload: dict) -> None:
        for websocket in self.connections[channel]:
            await websocket.send_json(payload)
