class HistoryService:
    def paginate(self, items: list[dict], page: int = 1, page_size: int = 10) -> dict:
        start = (page - 1) * page_size
        end = start + page_size
        return {
          "page": page,
          "page_size": page_size,
          "total": len(items),
          "items": items[start:end],
        }
