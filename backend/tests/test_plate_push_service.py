from app.services.plate_push_service import PlatePushService


def test_resolve_stream_name_uses_rtsp_tail_by_default():
    service = PlatePushService()

    result = service._resolve_stream_name("rtsp://10.126.59.120:8554/live/live7", None)

    assert result == "plate-live7"


def test_resolve_stream_name_sanitizes_custom_name():
    service = PlatePushService()

    result = service._resolve_stream_name("rtsp://10.126.59.120:8554/live/live1", " Road 1 / demo ")

    assert result == "Road-1-demo"


def test_status_skips_probe_before_publisher_starts(monkeypatch):
    service = PlatePushService()
    called = {"value": False}

    service._state.running = True
    service._state.publisher_started = False
    service._state.published = False
    service._state.publish_rtsp_url = "rtsp://127.0.0.1:8554/plate-live1"

    monkeypatch.setattr(service, "_probe_publish_stream", lambda _url: called.update({"value": True}) or True)

    response = service.status()

    assert response.published is False
    assert called["value"] is False


def test_status_probes_after_publisher_starts(monkeypatch):
    service = PlatePushService()

    service._state.running = True
    service._state.publisher_started = True
    service._state.published = False
    service._state.publish_rtsp_url = "rtsp://127.0.0.1:8554/plate-live1"

    monkeypatch.setattr(service, "_probe_publish_stream", lambda _url: True)

    response = service.status()

    assert response.published is True


def test_status_response_reports_connecting_source_phase():
    service = PlatePushService()
    service._state.running = True
    service._state.publisher_started = False
    service._state.published = False

    response = service._to_response()

    assert response.phase == "connecting_source"
    assert response.status_message == "正在连接源 RTSP，等待首帧"


def test_status_response_reports_source_unavailable_phase():
    service = PlatePushService()
    service._state.last_error = "Failed to open the RTSP stream. Diagnostics: FFMPEG backend open failed"

    response = service._to_response()

    assert response.phase == "source_unavailable"


def test_response_reports_passthrough_mode_when_processing_disabled():
    service = PlatePushService()
    service._state.running = True
    service._state.published = True
    service._state.process_frames = False

    response = service._to_response()

    assert response.process_frames is False
    assert response.status_message == "实时直推预览中"
