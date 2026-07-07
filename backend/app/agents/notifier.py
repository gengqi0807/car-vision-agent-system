from app.utils.websocket_manager import websocket_manager


class Notifier:
    async def broadcast(self, message: dict) -> int:
        return await websocket_manager.broadcast("alerts", message)
