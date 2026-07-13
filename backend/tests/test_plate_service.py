from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.models_infer.paddleocr_recognizer import PaddleOCRRecognizer
from app.models_infer.vehicle_type_classifier import VehicleTypePrediction
from app.models_infer.yolo_detector import YoloDetector
from app.services import plate_service as plate_service_module
from app.services.plate_service import PlateService

FAKE_PLATE = "TEST123"
FAKE_COLOR = "蓝牌"


@dataclass
class FakeDetection:
    plate_number: str
    plate_color: str
    confidence: float
    bbox: list[int]


class FakeRecognizer:
    def is_available(self):
        return True

    def recognize_all(self, _image_source, **_kwargs):
        return [
            FakeDetection(
                plate_number=FAKE_PLATE,
                plate_color="BLUE",
                confidence=0.93,
                bbox=[10, 20, 120, 40],
            )
        ]

    def recognize(self, _image_source, **_kwargs):
        return FakeDetection(
            plate_number="粤B12345",
            plate_color="blue",
            confidence=0.96,
            bbox=[0, 0, 0, 0],
        )


class FallbackOnlyRecognizer(FakeRecognizer):
    def recognize(self, _image_source, **_kwargs):
        return None


class ValueErrorCropRecognizer(FakeRecognizer):
    def recognize(self, _image_source, **_kwargs):
        raise ValueError("bad crop")


class RuntimeErrorCropRecognizer(FakeRecognizer):
    def __init__(self) -> None:
        self.reset_called = False

    def recognize(self, _image_source, **_kwargs):
        raise RuntimeError("paddle crashed")

    def reset_runtime(self):
        self.reset_called = True


class FakeDetector:
    def detect(self, _source):
        return [
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [30, 40, 140, 44],
                "confidence": 0.88,
            }
        ]


class FakeVehicleClassifier:
    def __init__(self, label: str = "Jeep", available: bool = True) -> None:
        self.label = label
        self.available = available
        self.calls: list[object] = []

    def is_available(self):
        return self.available

    def classify(self, source):
        self.calls.append(source)
        return VehicleTypePrediction(label=self.label, confidence=0.91)


def test_recognize_image_bytes_persists_history(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'plate.db'}", future=True)
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(plate_service_module, "SessionLocal", testing_session)

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", FakeRecognizer())
    monkeypatch.setattr(service, "_should_use_detector", lambda: False)

    result = service.recognize_image_bytes(b"fake-image-bytes", "sample.jpg")

    assert result.frame_id == "sample.jpg"
    assert len(result.detections) == 1
    assert result.detections[0].plate_number == FAKE_PLATE
    assert result.detections[0].plate_color == FAKE_COLOR
    assert result.detections[0].confidence == 0.93

    history = service.list_history()
    assert history
    assert history[0].plate_number == FAKE_PLATE


def test_recognize_image_bytes_handles_empty_input():
    service = PlateService()

    result = service.recognize_image_bytes(b"", "sample.jpg")

    assert result.frame_id == "sample.jpg"
    assert result.detections == []


def test_recognize_image_bytes_uses_fast_detector_path_when_detector_available(monkeypatch):
    service = PlateService()
    captured: dict[str, object] = {}
    decoded_image = object()

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda image_bytes: decoded_image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda image: [{"bbox": [1, 2, 3, 4], "confidence": 0.9}])
    captured_call: dict[str, object] = {}
    monkeypatch.setattr(
        service,
        "_recognize_detector_hits",
        lambda image, hits, *, fast_mode, preserve_unread, allow_ocr_fallback: captured_call.update(
            {
                "image": image,
                "hits": hits,
                "fast_mode": fast_mode,
                "preserve_unread": preserve_unread,
                "allow_ocr_fallback": allow_ocr_fallback,
            }
        )
        or [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.9,
                bbox=[1, 2, 3, 4],
            )
        ],
    )
    monkeypatch.setattr(service, "_save_history", lambda *_args, **_kwargs: None)

    result = service.recognize_image_bytes(b"fake-image-bytes", "fast-path.jpg")

    assert len(result.detections) == 1
    assert result.detections[0].plate_number == FAKE_PLATE
    assert captured_call["image"] is decoded_image
    assert captured_call["fast_mode"] is True
    assert captured_call["preserve_unread"] is False
    assert captured_call["allow_ocr_fallback"] is False


def test_recognize_detections_via_detector_uses_video_detector_path(monkeypatch):
    service = PlateService()
    decoded_image = object()
    captured: dict[str, object] = {}

    class VideoDetector:
        def detect(self, _image):
            raise AssertionError("video mode should not call detect()")

        def detect_video(self, image):
            captured["image"] = image
            return [{"label": "plate", "kind": "plate", "bbox": [1, 2, 30, 10], "confidence": 0.9}]

    monkeypatch.setattr(service, "detector", VideoDetector())
    monkeypatch.setattr(service, "_decode_image_source", lambda _source: decoded_image)
    monkeypatch.setattr(
        service,
        "_recognize_detector_hit",
        lambda image, hit, aggressive=False, fast_mode=False, preserve_unread=False, allow_ocr_fallback=True, video_mode=False: [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.9,
                bbox=hit["bbox"],
            )
        ],
    )

    result = service._recognize_detections_via_detector(object(), video_mode=True)

    assert captured["image"] is decoded_image
    assert len(result) == 1
    assert result[0].plate_number == FAKE_PLATE


def test_recognize_detections_via_detector_keeps_image_detector_path(monkeypatch):
    service = PlateService()
    decoded_image = object()

    class ImageDetector:
        def detect(self, image):
            assert image is decoded_image
            return [{"label": "plate", "kind": "plate", "bbox": [1, 2, 30, 10], "confidence": 0.9}]

        def detect_video(self, _image):
            raise AssertionError("image mode should not call detect_video()")

    monkeypatch.setattr(service, "detector", ImageDetector())
    monkeypatch.setattr(service, "_decode_image_source", lambda _source: decoded_image)
    monkeypatch.setattr(
        service,
        "_recognize_detector_hit",
        lambda image, hit, aggressive=False, fast_mode=False, preserve_unread=False, allow_ocr_fallback=True, video_mode=False: [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.9,
                bbox=hit["bbox"],
            )
        ],
    )

    result = service._recognize_detections_via_detector(b"fake-image", video_mode=False)

    assert len(result) == 1
    assert result[0].plate_number == FAKE_PLATE


def test_attach_vehicle_type_context_to_overlapping_plate_hit():
    service = PlateService()

    hits = [
        {
            "label": "car",
            "kind": "vehicle",
            "bbox": [10, 10, 220, 120],
            "confidence": 0.82,
        },
        {
            "label": "plate",
            "kind": "plate",
            "bbox": [70, 60, 90, 28],
            "confidence": 0.91,
        },
    ]

    enriched = service._attach_vehicle_type_context_to_hits(hits)

    assert enriched[0]["vehicle_type"] == "\u8f7f\u8f66"
    assert enriched[1]["vehicle_type"] == "\u8f7f\u8f66"


def test_recognize_detector_hit_preserves_vehicle_type_from_hit(monkeypatch):
    import numpy as np

    service = PlateService()
    crop = np.zeros((20, 60, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "_resolve_detector_crop_bboxes", lambda image, hit, video_mode=False: [[5, 6, 60, 20]])
    monkeypatch.setattr(service, "_crop_detection", lambda image, bbox: crop)
    monkeypatch.setattr(service, "_build_ocr_crop_variants", lambda _crop: [crop])
    monkeypatch.setattr(
        service,
        "_recognize_crop_with_paddleocr",
        lambda crop_image, hit, crop_bbox, allow_ocr_fallback=True: plate_service_module.PlateDetection(
            plate_number=FAKE_PLATE,
            plate_color=FAKE_COLOR,
            confidence=0.94,
            bbox=list(crop_bbox),
        ),
    )

    detections = service._recognize_detector_hit(
        crop,
        {
            "label": "plate",
            "kind": "plate",
            "bbox": [5, 6, 60, 20],
            "confidence": 0.87,
            "vehicle_type": "\u5361\u8f66",
        },
    )

    assert len(detections) == 1
    assert detections[0].vehicle_type == "\u5361\u8f66"


def test_resolve_vehicle_type_for_hit_accepts_lower_center_vehicle_context():
    service = PlateService()

    vehicle_type = service._resolve_vehicle_type_for_hit(
        {
            "label": "plate",
            "kind": "plate",
            "bbox": [118, 112, 52, 16],
            "confidence": 0.71,
        },
        [
            {
                "label": "car",
                "kind": "vehicle",
                "bbox": [40, 40, 220, 120],
                "confidence": 0.83,
            }
        ],
    )

    assert vehicle_type == "\u8f7f\u8f66"


def test_resolve_vehicle_type_for_plate_detection_prefers_fine_classifier(monkeypatch):
    import numpy as np

    service = PlateService()
    classifier = FakeVehicleClassifier(label="Jeep")
    crop = np.zeros((40, 80, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "vehicle_classifier", classifier)
    monkeypatch.setattr(service, "_crop_vehicle_candidate_for_classifier", lambda *args, **kwargs: crop)
    monkeypatch.setattr(
        service,
        "_classify_vehicle_type_near_plate_bbox",
        lambda *args, **kwargs: "\u8f7f\u8f66",
    )

    result = service._resolve_vehicle_type_for_plate_detection(
        np.zeros((120, 240, 3), dtype=np.uint8),
        plate_service_module.PlateDetection(
            plate_number="ABC1234",
            plate_color="blue",
            confidence=0.8,
            bbox=[20, 20, 40, 12],
        ),
        fast_mode=True,
        video_mode=False,
    )

    assert result == "\u5409\u666e\u8f66"
    assert classifier.calls == [crop]


def test_resolve_vehicle_type_for_plate_detection_rejects_bus_override_from_classifier(monkeypatch):
    import numpy as np

    service = PlateService()
    classifier = FakeVehicleClassifier(label="Bus")
    crop = np.zeros((40, 80, 3), dtype=np.uint8)

    monkeypatch.setattr(service, "vehicle_classifier", classifier)
    monkeypatch.setattr(service, "_crop_vehicle_candidate_for_classifier", lambda *args, **kwargs: crop)
    monkeypatch.setattr(
        service,
        "_classify_vehicle_type_near_plate_bbox",
        lambda *args, **kwargs: "\u8f7f\u8f66",
    )

    result = service._resolve_vehicle_type_for_plate_detection(
        np.zeros((120, 240, 3), dtype=np.uint8),
        plate_service_module.PlateDetection(
            plate_number="ABC1234",
            plate_color="blue",
            confidence=0.8,
            bbox=[20, 20, 40, 12],
        ),
        fast_mode=True,
        video_mode=False,
    )

    assert result == "\u8f7f\u8f66"


def test_recognize_image_detections_keeps_vehicle_context_without_ocr_vehicle_pass(monkeypatch):
    service = PlateService()
    image = object()
    captured_hits: list[dict] = []

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits",
        lambda _image: [
            {
                "label": "car",
                "kind": "vehicle",
                "bbox": [20, 20, 220, 120],
                "confidence": 0.82,
            },
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [92, 92, 64, 18],
                "confidence": 0.91,
            },
        ],
    )
    monkeypatch.setattr(
        service,
        "_recognize_detector_hits",
        lambda image_obj, hits, **kwargs: captured_hits.extend(hits)
        or [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                vehicle_type=hits[0].get("vehicle_type", "\u672a\u8bc6\u522b"),
                confidence=0.93,
                bbox=hits[0]["bbox"],
            )
        ],
    )

    result = service._recognize_image_detections(b"fake-image")

    assert len(captured_hits) == 1
    assert captured_hits[0]["kind"] == "plate"
    assert captured_hits[0]["vehicle_type"] == "\u8f7f\u8f66"
    assert result[0].vehicle_type == "\u8f7f\u8f66"


def test_recognize_image_detections_falls_back_to_detailed_small_plate_scan(monkeypatch):
    service = PlateService()
    image = object()
    captured_call: dict[str, object] = {}

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda _image: [])
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits_detailed",
        lambda _image: [
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [92, 92, 28, 10],
                "confidence": 0.48,
            }
        ],
    )
    monkeypatch.setattr(
        service,
        "_recognize_detector_hits",
        lambda image_obj, hits, *, fast_mode, preserve_unread, allow_ocr_fallback: captured_call.update(
            {
                "image": image_obj,
                "hits": hits,
                "fast_mode": fast_mode,
                "preserve_unread": preserve_unread,
                "allow_ocr_fallback": allow_ocr_fallback,
            }
        )
        or [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.88,
                bbox=[92, 92, 28, 10],
            )
        ],
    )

    result = service._recognize_image_detections(b"fake-image")

    assert len(result) == 1
    assert captured_call["image"] is image
    assert captured_call["fast_mode"] is False
    assert captured_call["preserve_unread"] is False
    assert captured_call["allow_ocr_fallback"] is True


def test_recognize_image_detections_uses_detailed_vehicle_hit_when_no_plate_hit_exists(monkeypatch):
    service = PlateService()
    image = object()
    fallback_called = {"value": False}

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda _image: [])
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits_detailed",
        lambda _image: [
            {
                "label": "car",
                "kind": "vehicle",
                "bbox": [30, 250, 210, 118],
                "confidence": 0.79,
            }
        ],
    )
    monkeypatch.setattr(
        service,
        "_recognize_detector_hits",
        lambda image_obj, hits, *, fast_mode, preserve_unread, allow_ocr_fallback, video_mode=False: [
            plate_service_module.PlateDetection(
                plate_number="京K9134",
                plate_color=FAKE_COLOR,
                confidence=0.83,
                bbox=[88, 300, 52, 16],
            )
        ]
        if any(str(hit.get("kind")) == "vehicle" for hit in hits) and video_mode
        else [],
    )
    monkeypatch.setattr(
        service,
        "_recognize_detections",
        lambda *_args, **_kwargs: fallback_called.update({"value": True}) or [],
    )

    result = service._recognize_image_detections(b"fake-image")

    assert len(result) == 1
    assert result[0].plate_number == "京K9134"
    assert fallback_called["value"] is False


def test_augment_image_hits_with_vehicle_detector_adds_vehicle_hits(monkeypatch):
    service = PlateService()
    image = object()

    class VehicleDetector:
        def detect_vehicle_classes(self, source, *, fast_mode=False):
            assert source is image
            assert fast_mode is False
            return [
                {
                    "label": "car",
                    "kind": "vehicle",
                    "bbox": [30, 250, 210, 118],
                    "confidence": 0.79,
                }
            ]

    monkeypatch.setattr(service, "detector", VehicleDetector())

    result = service._augment_image_hits_with_vehicle_detector(
        image,
        [
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [92, 92, 64, 18],
                "confidence": 0.91,
            }
        ],
    )

    assert len(result) == 2
    assert any(str(item.get("kind")) == "vehicle" for item in result)


def test_recognize_image_detections_skips_detailed_scan_when_fast_plate_hit_exists(monkeypatch):
    service = PlateService()
    image = object()
    detailed_called = {"value": False}

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits",
        lambda _image: [
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [30, 40, 140, 44],
                "confidence": 0.9,
            }
        ],
    )
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits_detailed",
        lambda _image: detailed_called.update({"value": True}) or [],
    )
    monkeypatch.setattr(
        service,
        "_recognize_detector_hits",
        lambda image_obj, hits, *, fast_mode, preserve_unread, allow_ocr_fallback: [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.9,
                bbox=[30, 40, 140, 44],
            )
        ],
    )

    result = service._recognize_image_detections(b"fake-image")

    assert len(result) == 1
    assert detailed_called["value"] is False


def test_recognize_image_detections_supplements_unmatched_vehicle_hit(monkeypatch):
    service = PlateService()
    image = object()
    detailed_called = {"value": False}

    fast_hits = [
        {
            "label": "car",
            "kind": "vehicle",
            "bbox": [20, 20, 220, 120],
            "confidence": 0.84,
        },
        {
            "label": "plate",
            "kind": "plate",
            "bbox": [92, 92, 64, 18],
            "confidence": 0.91,
        },
        {
            "label": "car",
            "kind": "vehicle",
            "bbox": [30, 250, 210, 118],
            "confidence": 0.79,
        },
    ]

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda _image: fast_hits)
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits_detailed",
        lambda _image: detailed_called.update({"value": True}) or [],
    )

    def fake_recognize_detector_hits(
        image_obj,
        hits,
        *,
        fast_mode,
        preserve_unread,
        allow_ocr_fallback,
        video_mode=False,
    ):
        if any(str(hit.get("kind")) == "vehicle" for hit in hits):
            return [
                plate_service_module.PlateDetection(
                    plate_number="京K9134",
                    plate_color=FAKE_COLOR,
                    confidence=0.83,
                    bbox=[88, 300, 52, 16],
                )
            ]
        return [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.91,
                bbox=[92, 92, 64, 18],
            )
        ]

    monkeypatch.setattr(service, "_recognize_detector_hits", fake_recognize_detector_hits)

    result = service._recognize_image_detections(b"fake-image")

    assert len(result) == 2
    assert {item.plate_number for item in result} == {FAKE_PLATE, "京K9134"}
    assert detailed_called["value"] is False


def test_recognize_image_detections_boosts_unmatched_vehicle_hit_with_local_ocr(monkeypatch):
    service = PlateService()
    image = object()
    detailed_called = {"value": False}

    fast_hits = [
        {
            "label": "car",
            "kind": "vehicle",
            "bbox": [20, 20, 220, 120],
            "confidence": 0.84,
        },
        {
            "label": "plate",
            "kind": "plate",
            "bbox": [92, 92, 64, 18],
            "confidence": 0.91,
        },
        {
            "label": "car",
            "kind": "vehicle",
            "bbox": [30, 250, 210, 118],
            "confidence": 0.79,
        },
    ]

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda _image: fast_hits)
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits_detailed",
        lambda _image: detailed_called.update({"value": True}) or [],
    )

    def fake_recognize_detector_hits(
        image_obj,
        hits,
        *,
        fast_mode,
        preserve_unread,
        allow_ocr_fallback,
        video_mode=False,
    ):
        if any(str(hit.get("kind")) == "vehicle" for hit in hits):
            return []
        return [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.91,
                bbox=[92, 92, 64, 18],
            )
        ]

    monkeypatch.setattr(service, "_recognize_detector_hits", fake_recognize_detector_hits)
    monkeypatch.setattr(
        service,
        "_resolve_detector_crop_bboxes",
        lambda image_obj, hit, *, video_mode=False: [[40, 280, 90, 24]] if str(hit.get("kind")) == "vehicle" else [],
    )
    monkeypatch.setattr(service, "_crop_detection", lambda _image_obj, _bbox: object())
    monkeypatch.setattr(service, "_build_best_effort_ocr_crop_variants", lambda _crop: ["base", "boosted"])

    def fake_recognize_crop(crop, hit, crop_bbox, confidence_threshold=None, allow_ocr_fallback=True):
        if crop != "boosted":
            return None
        return plate_service_module.PlateDetection(
            plate_number="浜琄9134",
            plate_color=FAKE_COLOR,
            confidence=0.82,
            bbox=list(crop_bbox),
        )

    monkeypatch.setattr(service, "_recognize_crop_with_paddleocr", fake_recognize_crop)

    result = service._recognize_image_detections(b"fake-image")

    assert len(result) == 2
    assert {item.plate_number for item in result} == {FAKE_PLATE, "浜琄9134"}
    assert detailed_called["value"] is False


def test_recognize_image_detections_boosts_failed_plate_hit_with_local_ocr(monkeypatch):
    service = PlateService()
    image = object()
    detailed_called = {"value": False}

    fast_hits = [
        {
            "label": "plate",
            "kind": "plate",
            "bbox": [215, 1025, 163, 55],
            "confidence": 0.93,
        },
        {
            "label": "plate",
            "kind": "plate",
            "bbox": [357, 285, 108, 38],
            "confidence": 0.81,
        },
    ]

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda _image: fast_hits)
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits_detailed",
        lambda _image: detailed_called.update({"value": True}) or [],
    )
    monkeypatch.setattr(service, "_augment_image_hits_with_vehicle_detector", lambda image_obj, hits: hits)

    def fake_recognize_detector_hits(
        image_obj,
        hits,
        *,
        fast_mode,
        preserve_unread,
        allow_ocr_fallback,
        video_mode=False,
    ):
        bbox = list(hits[0].get("bbox", []))
        if bbox == [215, 1025, 163, 55]:
            return [
                plate_service_module.PlateDetection(
                    plate_number=FAKE_PLATE,
                    plate_color=FAKE_COLOR,
                    confidence=0.93,
                    bbox=bbox,
                )
            ]
        return []

    monkeypatch.setattr(service, "_recognize_detector_hits", fake_recognize_detector_hits)
    monkeypatch.setattr(
        service,
        "_build_failed_image_plate_crop_bboxes",
        lambda image_obj, hit: [[360, 280, 118, 46]] if list(hit.get("bbox", [])) == [357, 285, 108, 38] else [list(hit.get("bbox", []))],
    )
    monkeypatch.setattr(
        service,
        "_resolve_vehicle_type_for_plate_detection",
        lambda image_obj, detection, *, fast_mode, video_mode: plate_service_module.VEHICLE_TYPE_UNKNOWN,
    )
    monkeypatch.setattr(service, "_crop_detection", lambda _image_obj, _bbox: object())
    monkeypatch.setattr(service, "_build_best_effort_ocr_crop_variants", lambda _crop: ["base", "boosted"])

    def fake_recognize_crop(crop, hit, crop_bbox, confidence_threshold=None, allow_ocr_fallback=True):
        if list(crop_bbox) != [360, 280, 118, 46] or crop != "boosted":
            return None
        return plate_service_module.PlateDetection(
            plate_number="浜琄9134",
            plate_color=FAKE_COLOR,
            confidence=0.79,
            bbox=list(crop_bbox),
        )

    monkeypatch.setattr(service, "_recognize_crop_with_paddleocr", fake_recognize_crop)

    result = service._recognize_image_detections(b"fake-image")

    assert len(result) == 2
    assert {item.plate_number for item in result} == {FAKE_PLATE, "浜琄9134"}
    assert detailed_called["value"] is False


def test_recognize_image_detections_falls_back_to_aggressive_full_frame_scan_when_detector_misses(monkeypatch):
    service = PlateService()
    image = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda _image: [])
    monkeypatch.setattr(service, "_detect_image_detector_hits_detailed", lambda _image: [])
    monkeypatch.setattr(
        service,
        "_recognize_detections",
        lambda image_source, **kwargs: captured.update(
            {
                "image_source": image_source,
                "kwargs": kwargs,
            }
        )
        or [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.76,
                bbox=[92, 92, 24, 8],
            )
        ],
    )

    result = service._recognize_image_detections(b"fake-image")

    assert len(result) == 1
    assert captured["image_source"] == b"fake-image"
    assert captured["kwargs"] == {
        "aggressive": True,
        "heavy_scan": True,
        "allow_full_frame_fallback": True,
        "fast_mode": False,
    }


def test_recognize_image_detections_skips_aggressive_full_frame_scan_when_detailed_hits_exist(monkeypatch):
    service = PlateService()
    image = object()
    captured = {"called": False}

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda _image_bytes: image)
    monkeypatch.setattr(service, "_detect_image_detector_hits", lambda _image: [])
    monkeypatch.setattr(
        service,
        "_detect_image_detector_hits_detailed",
        lambda _image: [
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [357, 285, 108, 38],
                "confidence": 0.81,
            }
        ],
    )
    monkeypatch.setattr(service, "_augment_image_hits_with_vehicle_detector", lambda image_obj, hits: hits)
    monkeypatch.setattr(service, "_recognize_image_unmatched_vehicle_hits", lambda image_obj, hits, plate_hits: [])
    monkeypatch.setattr(service, "_recognize_detector_hits", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        service,
        "_recognize_detections",
        lambda *args, **kwargs: captured.update({"called": True}) or [],
    )

    result = service._recognize_image_detections(b"fake-image")

    assert result == []
    assert captured["called"] is False


def test_augment_hits_with_secondary_vehicle_context_uses_vehicle_detector(monkeypatch):
    service = PlateService()
    image = object()

    class VehicleDetector:
        def detect_vehicle_classes(self, source, *, fast_mode=False):
            assert source is image
            assert fast_mode is True
            return [
                {
                    "label": "car",
                    "kind": "vehicle",
                    "bbox": [20, 20, 200, 100],
                    "confidence": 0.81,
                }
            ]

    monkeypatch.setattr(service, "detector", VehicleDetector())

    result = service._augment_hits_with_secondary_vehicle_context(
        image,
        [
            {
                "label": "plate",
                "kind": "plate",
                "bbox": [88, 86, 62, 18],
                "confidence": 0.91,
            }
        ],
        fast_mode=True,
    )

    assert len(result) == 2
    assert any(item["kind"] == "vehicle" for item in result)


def test_recognize_image_bytes_uses_ocr_fallback_when_detector_unavailable(monkeypatch):
    service = PlateService()
    captured: dict[str, object] = {}

    def fake_recognize(image_source, **kwargs):
        captured["image_source"] = image_source
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(service, "_should_use_detector", lambda: False)
    monkeypatch.setattr(service, "_recognize_detections", fake_recognize)

    result = service.recognize_image_bytes(b"fake-image-bytes", "ocr-path.jpg")

    assert result.detections == []
    assert captured["image_source"] == b"fake-image-bytes"
    assert captured["kwargs"] == {
        "heavy_scan": False,
        "allow_full_frame_fallback": True,
        "fast_mode": True,
    }


def test_recognize_image_bytes_uses_detector_fast_hits_when_available(monkeypatch):
    service = PlateService()
    detector_hits = [{"label": "plate", "kind": "plate", "bbox": [30, 40, 140, 44], "confidence": 0.9}]
    captured: dict[str, object] = {}
    decoded_image = object()

    class FastDetector:
        def detect_fast(self, image):
            captured["image"] = image
            return detector_hits

    monkeypatch.setattr(service, "detector", FastDetector())
    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_decode_image_source", lambda image_bytes: decoded_image)
    monkeypatch.setattr(
        service,
        "_recognize_detector_hits",
        lambda image, hits, *, fast_mode, preserve_unread, allow_ocr_fallback: [
            plate_service_module.PlateDetection(
                plate_number=FAKE_PLATE,
                plate_color=FAKE_COLOR,
                confidence=0.91,
                bbox=[3, 4, 5, 6],
            )
        ],
    )
    monkeypatch.setattr(service, "_save_history", lambda *_args, **_kwargs: None)

    result = service.recognize_image_bytes(b"fake-image-bytes", "detector-fast.jpg")

    assert len(result.detections) == 1
    assert result.detections[0].plate_number == FAKE_PLATE
    assert captured["image"] is decoded_image


def test_recognize_image_bytes_skips_full_frame_fallback_when_detector_path_fails(monkeypatch):
    import cv2
    import numpy as np

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", FallbackOnlyRecognizer())
    monkeypatch.setattr(service, "detector", FakeDetector())
    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_save_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "plate_save_uploads", False)

    image = np.zeros((120, 240, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    result = service.recognize_image_bytes(encoded.tobytes(), "detector.jpg")

    assert result.detections == []


def test_recognize_image_bytes_prefers_paddleocr_crop_ocr(monkeypatch):
    import cv2
    import numpy as np

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", FakeRecognizer())
    monkeypatch.setattr(service, "detector", FakeDetector())
    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_save_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "plate_save_uploads", False)

    image = np.zeros((120, 240, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    result = service.recognize_image_bytes(encoded.tobytes(), "detector-paddleocr.jpg")

    assert len(result.detections) == 1
    assert result.detections[0].plate_number == "粤B12345"
    assert result.detections[0].plate_color == "蓝牌"
    assert result.detections[0].bbox == [30, 40, 140, 44]
    assert result.detections[0].confidence > 0.9


def test_recognize_image_bytes_skips_crop_valueerror_without_full_frame_fallback(monkeypatch):
    import cv2
    import numpy as np

    service = PlateService()
    monkeypatch.setattr(service, "recognizer", ValueErrorCropRecognizer())
    monkeypatch.setattr(service, "detector", FakeDetector())
    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_save_history", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "plate_save_uploads", False)

    image = np.zeros((120, 240, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok

    result = service.recognize_image_bytes(encoded.tobytes(), "detector-valueerror.jpg")

    assert result.detections == []


def test_recognize_crop_with_paddleocr_skips_runtime_error(monkeypatch):
    import numpy as np

    service = PlateService()
    recognizer = RuntimeErrorCropRecognizer()
    monkeypatch.setattr(service, "recognizer", recognizer)

    result = service._recognize_crop_with_paddleocr(
        np.zeros((20, 40, 3), dtype=np.uint8),
        {"confidence": 0.9},
        [0, 0, 40, 20],
    )

    assert result is None
    assert recognizer.reset_called is True


def test_paddleocr_recognize_skips_heavy_fallback_when_disabled(monkeypatch):
    recognizer = PaddleOCRRecognizer()
    fake_image = object()

    monkeypatch.setattr(recognizer, "_load_image_array", lambda _source: fake_image)
    monkeypatch.setattr(recognizer, "_resolve_confidence_threshold", lambda _threshold: 0.35)
    monkeypatch.setattr(recognizer, "_predict_text", lambda _image: [])
    monkeypatch.setattr(recognizer, "_pick_best_text_recognition_candidate", lambda _result, _threshold: None)

    fallback_called = {"value": False}

    def fake_predict(_image):
        fallback_called["value"] = True
        return []

    monkeypatch.setattr(recognizer, "_predict", fake_predict)

    result = recognizer.recognize(b"unused", allow_ocr_fallback=False)

    assert result is None
    assert fallback_called["value"] is False


def test_finalize_video_detections_filters_weak_single_hits():
    service = PlateService()
    weak_single = plate_service_module.VideoDetectionStats(
        detection=plate_service_module.PlateDetection(
            plate_number="粤A12345",
            plate_color="蓝牌",
            confidence=0.43,
            bbox=[0, 0, 120, 40],
        ),
        fresh_count=1,
        display_count=3,
    )
    stable_plate = plate_service_module.VideoDetectionStats(
        detection=plate_service_module.PlateDetection(
            plate_number="粤B54321",
            plate_color="蓝牌",
            confidence=0.41,
            bbox=[10, 10, 120, 40],
        ),
        fresh_count=2,
        display_count=4,
    )

    result = service._finalize_video_detections(
        {
            "weak": weak_single,
            "stable": stable_plate,
        }
    )

    assert [item.plate_number for item in result] == ["粤B54321"]


def test_finalize_video_detections_stabilizes_small_seven_char_yellow_plate_to_blue():
    service = PlateService()
    result = service._finalize_video_detections(
        {
            "plate": plate_service_module.VideoDetectionStats(
                detection=plate_service_module.PlateDetection(
                    plate_number="ABC1234",
                    plate_color="黄牌",
                    confidence=0.72,
                    bbox=[10, 10, 34, 10],
                ),
                fresh_count=2,
                display_count=4,
            )
        }
    )

    assert len(result) == 1
    assert result[0].plate_color == "蓝牌"


def test_normalize_plate_color_for_plate_maps_seven_char_black_plate_to_blue():
    service = PlateService()

    result = service._normalize_plate_color_for_plate("black", "ABC1234")

    assert result == plate_service_module.PLATE_COLOR_BLUE


def test_normalize_plate_color_for_plate_maps_eight_char_black_plate_to_green():
    service = PlateService()

    result = service._normalize_plate_color_for_plate("black", "ABC12345")

    assert result == plate_service_module.PLATE_COLOR_GREEN


def test_is_reasonable_crop_bbox_allows_small_sandbox_plate():
    service = PlateService()

    assert service._is_reasonable_crop_bbox([0, 0, 16, 8]) is True
    assert service._is_reasonable_crop_bbox([0, 0, 10, 5]) is False


def test_extract_template_allows_small_video_tracking_patch():
    import numpy as np

    service = PlateService()
    gray = np.zeros((30, 30), dtype=np.uint8)

    template = service._extract_template(gray, [4, 5, 12, 6])

    assert template is not None
    assert template.shape == (6, 12)


def test_is_plausible_plate_bbox_prefers_plate_like_aspect_ratio():
    service = PlateService()

    assert service._is_plausible_plate_bbox([0, 0, 32, 10]) is True
    assert service._is_plausible_plate_bbox([0, 0, 10, 20]) is False


def test_find_track_match_rejects_second_unread_car_on_same_lane():
    service = PlateService()
    track = plate_service_module.PlateTrack(
        track_id="track-1",
        plate_number="",
        plate_color="\u672a\u77e5",
        confidence=0.42,
        bbox=[100, 100, 80, 24],
        template=object(),
        last_seen_frame=1,
        last_recognized_frame=1,
    )
    detection = plate_service_module.PlateDetection(
        plate_number="",
        plate_color="\u672a\u77e5",
        confidence=0.31,
        bbox=[108, 118, 72, 22],
    )

    result = service._find_track_match(detection, [track], set())

    assert result is None


def test_find_track_match_keeps_same_unread_car_when_bbox_stays_close():
    service = PlateService()
    track = plate_service_module.PlateTrack(
        track_id="track-1",
        plate_number="",
        plate_color="\u672a\u77e5",
        confidence=0.42,
        bbox=[100, 100, 80, 24],
        template=object(),
        last_seen_frame=1,
        last_recognized_frame=1,
    )
    detection = plate_service_module.PlateDetection(
        plate_number="",
        plate_color="\u672a\u77e5",
        confidence=0.31,
        bbox=[104, 106, 78, 24],
    )

    result = service._find_track_match(detection, [track], set())

    assert result == 0


def test_find_untracked_detector_hits_preserves_second_car_probe(monkeypatch):
    import numpy as np

    service = PlateService()
    track = plate_service_module.PlateTrack(
        track_id="track-1",
        plate_number="",
        plate_color="\u672a\u77e5",
        confidence=0.42,
        bbox=[100, 100, 80, 24],
        template=object(),
        last_seen_frame=1,
        last_recognized_frame=1,
    )

    class ProbeDetector:
        def detect(self, _image):
            return [
                {
                    "label": "plate",
                    "kind": "plate",
                    "bbox": [108, 118, 72, 22],
                    "confidence": 0.30,
                }
            ]

    monkeypatch.setattr(service, "detector", ProbeDetector())
    monkeypatch.setattr(service, "_should_use_detector", lambda: True)

    result = service._find_untracked_detector_hits(np.zeros((240, 320, 3), dtype=np.uint8), [track], max_hits=2)

    assert len(result) == 1
    assert result[0]["bbox"] == [108, 118, 72, 22]


def test_find_track_match_keeps_recognized_track_matching_lenient():
    service = PlateService()
    track = plate_service_module.PlateTrack(
        track_id="track-1",
        plate_number="粤B12345",
        plate_color="蓝牌",
        confidence=0.86,
        bbox=[100, 100, 80, 24],
        template=object(),
        last_seen_frame=1,
        last_recognized_frame=1,
    )
    detection = plate_service_module.PlateDetection(
        plate_number="粤B12345",
        plate_color="蓝牌",
        confidence=0.81,
        bbox=[108, 118, 72, 22],
    )

    result = service._find_track_match(detection, [track], set())

    assert result == 0


def test_should_rerecognize_video_waits_longer_for_unread_tracks():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=20,
        last_recognition_frame=5,
        recognition_interval=10,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.4,
                bbox=[10, 10, 40, 12],
                template=object(),
                last_seen_frame=20,
                last_recognized_frame=5,
            )
        ],
    )

    assert service._should_rerecognize_video(state) is False

    state.frame_index = 25

    assert service._should_rerecognize_video(state) is True


def test_should_rerecognize_video_waits_longer_for_large_recognized_track():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=30,
        last_recognition_frame=10,
        recognition_interval=10,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="ABC1234",
                plate_color="blue",
                confidence=0.82,
                bbox=[20, 20, 80, 20],
                template=object(),
                last_seen_frame=30,
                last_recognized_frame=10,
            )
        ],
    )

    assert service._should_rerecognize_video(state) is False

    state.frame_index = 34

    assert service._should_rerecognize_video(state) is True

    state.frame_index = 42

    assert service._should_rerecognize_video(state) is True


def test_should_rerecognize_without_tracks_respects_interval_when_not_forced():
    service = PlateService()

    assert service._should_rerecognize(
        [],
        frame_index=1,
        last_recognition_frame=-8,
        recognition_interval=8,
        force_when_no_tracks=False,
    ) is True
    assert service._should_rerecognize(
        [],
        frame_index=2,
        last_recognition_frame=1,
        recognition_interval=8,
        force_when_no_tracks=False,
    ) is False
    assert service._should_rerecognize(
        [],
        frame_index=9,
        last_recognition_frame=1,
        recognition_interval=8,
        force_when_no_tracks=False,
    ) is True


def test_should_rerecognize_without_tracks_can_still_force_every_frame():
    service = PlateService()

    assert service._should_rerecognize(
        [],
        frame_index=2,
        last_recognition_frame=1,
        recognition_interval=8,
        force_when_no_tracks=True,
    ) is True


def test_should_skip_stream_frame_for_rate_limit():
    service = PlateService()

    assert service._should_skip_stream_frame_for_rate_limit(
        last_sent_at=0.0,
        current_time=10.0,
        min_interval=0.125,
    ) is False
    assert service._should_skip_stream_frame_for_rate_limit(
        last_sent_at=10.0,
        current_time=10.05,
        min_interval=0.125,
    ) is True
    assert service._should_skip_stream_frame_for_rate_limit(
        last_sent_at=10.0,
        current_time=10.2,
        min_interval=0.125,
    ) is False


def test_stream_read_failure_retry_limit_has_minimum_floor():
    service = PlateService()

    result = service._stream_read_failure_retry_limit()

    assert result >= 8


def test_stream_read_failure_retry_sleep_seconds_stays_bounded():
    service = PlateService()

    assert service._stream_read_failure_retry_sleep_seconds(0.125) == 0.0625
    assert service._stream_read_failure_retry_sleep_seconds(0.5) == 0.12
    assert service._stream_read_failure_retry_sleep_seconds(0.0) == 0.05


def test_stream_recognition_submit_interval_seconds_respects_frame_stride():
    service = PlateService()

    result = service._stream_recognition_submit_interval_seconds(0.125)

    assert result == 0.1875
    assert service._stream_recognition_submit_interval_seconds(0.01) == 0.12
    assert service._stream_recognition_submit_interval_seconds(1.0) == 0.3


def test_should_display_stream_cached_detections_rejects_stale_results():
    service = PlateService()

    assert service._should_display_stream_cached_detections(
        current_version=5,
        detections_version=5,
        detections_updated_at=10.0,
        current_time=10.4,
        submit_interval=0.2,
    ) is True
    assert service._should_display_stream_cached_detections(
        current_version=8,
        detections_version=5,
        detections_updated_at=10.0,
        current_time=10.4,
        submit_interval=0.2,
    ) is False
    assert service._should_display_stream_cached_detections(
        current_version=5,
        detections_version=5,
        detections_updated_at=10.0,
        current_time=11.1,
        submit_interval=0.2,
    ) is False


def test_should_rerecognize_video_is_more_aggressive_in_stream_mode():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=5,
        last_recognition_frame=1,
        recognition_interval=1,
        stream_mode=True,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="",
                plate_color="unknown",
                confidence=0.4,
                bbox=[10, 10, 40, 12],
                template=object(),
                last_seen_frame=5,
                last_recognized_frame=1,
            )
        ],
    )

    assert service._should_rerecognize_video(state) is True


def test_should_reject_stale_tracked_plate_for_low_tracking_score():
    service = PlateService()
    track = plate_service_module.PlateTrack(
        track_id="track-1",
        plate_number="ABC1234",
        plate_color="blue",
        confidence=0.82,
        bbox=[20, 20, 80, 20],
        template=object(),
        last_seen_frame=30,
        last_recognized_frame=10,
        last_tracking_score=0.58,
    )

    assert service._should_reject_stale_tracked_plate(track, frame_index=31) is True
    assert service._should_reject_stale_tracked_plate(track, frame_index=17) is False


def test_should_probe_new_video_tracks_runs_more_often_for_unread_tracks():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=9,
        last_probe_frame=4,
        recognition_interval=10,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.4,
                bbox=[10, 10, 40, 12],
                template=object(),
                last_seen_frame=9,
                last_recognized_frame=4,
            )
        ],
    )

    assert service._should_probe_new_video_tracks(state) is False

    state.frame_index = 18

    assert service._should_probe_new_video_tracks(state) is True


def test_should_probe_new_video_tracks_slows_unread_probe_when_large_plate_exists():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=18,
        last_probe_frame=4,
        recognition_interval=10,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="recognized",
                plate_number="ABC1234",
                plate_color="blue",
                confidence=0.82,
                bbox=[20, 20, 80, 20],
                template=object(),
                last_seen_frame=18,
                last_recognized_frame=4,
            ),
            plate_service_module.PlateTrack(
                track_id="unread",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.42,
                bbox=[120, 40, 36, 10],
                template=object(),
                last_seen_frame=18,
                last_recognized_frame=4,
            ),
        ],
    )

    assert service._should_probe_new_video_tracks(state) is False

    state.frame_index = 32

    assert service._should_probe_new_video_tracks(state) is True


def test_should_probe_new_video_tracks_waits_longer_for_large_recognized_track():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=30,
        last_probe_frame=4,
        recognition_interval=10,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="ABC1234",
                plate_color="blue",
                confidence=0.82,
                bbox=[20, 20, 80, 20],
                template=object(),
                last_seen_frame=30,
                last_recognized_frame=4,
            )
        ],
    )

    assert service._should_probe_new_video_tracks(state) is False

    state.frame_index = 88

    assert service._should_probe_new_video_tracks(state) is True


def test_should_skip_light_video_frame_only_for_stable_large_recognized_tracks():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=2,
        working_width=960,
        working_height=540,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="ABC1234",
                plate_color="blue",
                confidence=0.82,
                bbox=[20, 20, 80, 20],
                template=object(),
                last_seen_frame=2,
                last_recognized_frame=2,
            )
        ],
    )

    assert service._should_skip_light_video_frame(
        state=state,
        render=False,
        use_fast_large_plate_mode=True,
    ) is True
    assert service._should_skip_light_video_frame(
        state=state,
        render=True,
        use_fast_large_plate_mode=True,
    ) is False


def test_should_probe_new_video_tracks_slows_down_when_two_unread_tracks_are_active():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=20,
        last_probe_frame=2,
        recognition_interval=10,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="a",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.4,
                bbox=[20, 20, 34, 10],
                template=object(),
                last_seen_frame=20,
                last_recognized_frame=2,
            ),
            plate_service_module.PlateTrack(
                track_id="b",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.4,
                bbox=[60, 24, 34, 10],
                template=object(),
                last_seen_frame=20,
                last_recognized_frame=2,
            ),
        ],
    )

    assert service._should_probe_new_video_tracks(state) is False

    state.frame_index = 32

    assert service._should_probe_new_video_tracks(state) is True


def test_is_large_recognized_track_requires_truly_large_plate():
    service = PlateService()
    medium_track = plate_service_module.PlateTrack(
        track_id="medium",
        plate_number="ABC1234",
        plate_color="blue",
        confidence=0.8,
        bbox=[20, 20, 60, 16],
        template=object(),
        last_seen_frame=10,
        last_recognized_frame=10,
    )
    large_track = plate_service_module.PlateTrack(
        track_id="large",
        plate_number="ABC1234",
        plate_color="blue",
        confidence=0.8,
        bbox=[20, 20, 80, 20],
        template=object(),
        last_seen_frame=10,
        last_recognized_frame=10,
    )

    assert service._is_large_recognized_track(medium_track) is False
    assert service._is_large_recognized_track(large_track) is True


def test_find_untracked_detector_hits_uses_masked_probe(monkeypatch):
    import numpy as np

    service = PlateService()
    track = plate_service_module.PlateTrack(
        track_id="track-1",
        plate_number="",
        plate_color="\u672a\u77e5",
        confidence=0.42,
        bbox=[100, 100, 80, 24],
        template=object(),
        last_seen_frame=1,
        last_recognized_frame=1,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    calls: list[object] = []

    def fake_detect_video_detector_hits(frame, fast_mode=False):
        calls.append(frame)
        if len(calls) == 1:
            return []
        return [{"label": "plate", "kind": "plate", "bbox": [180, 120, 36, 12], "confidence": 0.35}]

    monkeypatch.setattr(service, "_should_use_detector", lambda: True)
    monkeypatch.setattr(service, "_detect_video_detector_hits", fake_detect_video_detector_hits)

    result = service._find_untracked_detector_hits(image, [track], max_hits=2)

    assert len(calls) == 2
    assert len(result) == 1
    assert result[0]["bbox"] == [180, 120, 36, 12]


def test_select_video_detector_hits_for_recognition_prefers_plate_hits():
    service = PlateService()
    hits = [
        {"label": "car", "kind": "vehicle", "bbox": [0, 0, 80, 40], "confidence": 0.55},
        {"label": "plate", "kind": "plate", "bbox": [10, 10, 40, 12], "confidence": 0.82},
    ]

    result = service._select_video_detector_hits_for_recognition(hits, preserve_unread=True)

    assert len(result) == 1
    assert result[0]["kind"] == "plate"


def test_tracks_to_detections_suppresses_low_confidence_unread_side_shadow():
    service = PlateService()
    tracks = [
        plate_service_module.PlateTrack(
            track_id="recognized",
            plate_number="ABC1234",
            plate_color="blue",
            confidence=0.82,
            bbox=[120, 120, 44, 12],
            template=object(),
            last_seen_frame=10,
            last_recognized_frame=10,
        ),
        plate_service_module.PlateTrack(
            track_id="shadow",
            plate_number="",
            plate_color="\u672a\u77e5",
            confidence=0.44,
            bbox=[186, 102, 40, 12],
            template=object(),
            last_seen_frame=10,
            last_recognized_frame=10,
            unread_observations=2,
        ),
    ]

    result = service._tracks_to_detections(tracks)

    assert len(result) == 1
    assert result[0].plate_number == "ABC1234"


def test_recover_active_unread_tracks_recognizes_from_source_crop(monkeypatch):
    import numpy as np

    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=24,
        working_width=160,
        working_height=90,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.46,
                bbox=[20, 18, 34, 10],
                template=object(),
                last_seen_frame=24,
                last_recognized_frame=10,
                unread_observations=3,
                last_unread_ocr_frame=0,
            )
        ],
    )

    monkeypatch.setattr(service.recognizer, "is_available", lambda: True)
    monkeypatch.setattr(service, "_crop_detection", lambda image, bbox: np.zeros((20, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(service, "_build_best_effort_ocr_crop_variants", lambda crop: [crop])
    monkeypatch.setattr(
        service,
        "_recognize_crop_with_paddleocr",
        lambda *args, **kwargs: plate_service_module.PlateDetection(
            plate_number="73737",
            plate_color="blue",
            confidence=0.72,
            bbox=[40, 36, 68, 20],
        ),
    )

    result = service._recover_active_unread_tracks(
        source_frame=np.zeros((180, 320, 3), dtype=np.uint8),
        working_frame=np.zeros((90, 160, 3), dtype=np.uint8),
        state=state,
        fresh_detections=[],
    )

    assert len(result) == 1
    assert result[0].plate_number == "73737"
    assert result[0].bbox == [20, 18, 34, 10]
    assert state.tracks[0].last_unread_ocr_frame == 24


def test_should_attempt_active_unread_ocr_skips_non_small_unread_when_large_plate_exists():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=30,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="recognized",
                plate_number="ABC1234",
                plate_color="blue",
                confidence=0.82,
                bbox=[20, 20, 80, 20],
                template=object(),
                last_seen_frame=30,
                last_recognized_frame=20,
            ),
            plate_service_module.PlateTrack(
                track_id="unread",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.44,
                bbox=[120, 42, 58, 18],
                template=object(),
                last_seen_frame=30,
                last_recognized_frame=20,
                unread_observations=2,
            ),
        ],
    )

    assert service._should_attempt_active_unread_ocr(state=state, fresh_detections=[]) is False


def test_should_collect_unread_video_artifacts_throttles_by_interval():
    service = PlateService()
    state = plate_service_module.PlateProcessingState(
        frame_index=11,
        recognition_interval=12,
        tracks=[
            plate_service_module.PlateTrack(
                track_id="track-1",
                plate_number="",
                plate_color="\u672a\u77e5",
                confidence=0.42,
                bbox=[20, 18, 34, 10],
                template=object(),
                last_seen_frame=11,
                last_recognized_frame=10,
            )
        ],
    )

    assert service._should_collect_unread_video_artifacts(state) is False

    state.frame_index = 12

    assert service._should_collect_unread_video_artifacts(state) is True


def test_pick_stable_track_color_prefers_blue_when_votes_are_close_for_seven_char_plate():
    service = PlateService()

    result = service._pick_stable_track_color(
        {"蓝牌": 0.86, "黄牌": 0.9},
        "ABC1234",
        fallback_color="未知",
    )

    assert result == "蓝牌"


def test_resolve_video_output_fps_target_reduces_high_fps_output_workload(monkeypatch):
    service = PlateService()
    monkeypatch.setattr(settings, "plate_video_output_fps", 8)

    assert service._resolve_video_output_fps_target(25.0) == 5
    assert service._resolve_video_output_fps_target(15.0) == 8


def test_resolve_video_recognition_interval_uses_about_one_second_of_frames(monkeypatch):
    service = PlateService()
    monkeypatch.setattr(settings, "plate_video_process_every_n_frames", 10)

    assert service._resolve_video_recognition_interval(25.0) == 25
    assert service._resolve_video_recognition_interval(14.0) == 18
    assert service._resolve_video_recognition_interval(0.0) == 18


def test_video_track_update_interval_relaxes_upload_video_tracking_workload():
    service = PlateService()

    assert service._video_track_update_interval(use_fast_large_plate_mode=False) == 4
    assert service._video_track_update_interval(use_fast_large_plate_mode=True) == 8


def test_tracks_to_detections_defaults_small_video_plate_to_car():
    service = PlateService()
    track = plate_service_module.PlateTrack(
        track_id="track-1",
        plate_number="京A12345",
        plate_color="蓝牌",
        confidence=0.82,
        bbox=[20, 20, 54, 18],
        template=object(),
        last_seen_frame=10,
        last_recognized_frame=10,
        vehicle_type="未识别",
    )

    detections = service._tracks_to_detections([track])

    assert len(detections) == 1
    assert detections[0].vehicle_type == "轿车"


def test_finalize_video_detections_defaults_small_plate_vehicle_type_to_car():
    service = PlateService()
    result = service._finalize_video_detections(
        {
            "plate": plate_service_module.VideoDetectionStats(
                detection=plate_service_module.PlateDetection(
                    plate_number="京A12345",
                    plate_color="蓝牌",
                    vehicle_type="未识别",
                    confidence=0.72,
                    bbox=[10, 10, 56, 18],
                ),
                fresh_count=2,
                display_count=4,
            )
        }
    )

    assert len(result) == 1
    assert result[0].vehicle_type == "轿车"


def test_finalize_video_detections_merges_same_plate_number_with_different_colors():
    service = PlateService()

    result = service._finalize_video_detections(
        {
            "blue": plate_service_module.VideoDetectionStats(
                detection=plate_service_module.PlateDetection(
                    plate_number="ABC1234",
                    plate_color="蓝牌",
                    confidence=0.66,
                    bbox=[12, 12, 40, 12],
                ),
                fresh_count=2,
                display_count=4,
            ),
            "yellow": plate_service_module.VideoDetectionStats(
                detection=plate_service_module.PlateDetection(
                    plate_number="ABC1234",
                    plate_color="黄牌",
                    confidence=0.62,
                    bbox=[11, 11, 38, 11],
                ),
                fresh_count=1,
                display_count=3,
            ),
        }
    )

    assert len(result) == 1
    assert result[0].plate_number == "ABC1234"
    assert result[0].plate_color == "蓝牌"


def test_finalize_video_detections_keeps_strong_visible_fallback_candidate():
    service = PlateService()

    result = service._finalize_video_detections(
        {
            "stable": plate_service_module.VideoDetectionStats(
                detection=plate_service_module.PlateDetection(
                    plate_number="ABC1234",
                    plate_color="钃濈墝",
                    confidence=0.62,
                    bbox=[12, 12, 44, 14],
                ),
                fresh_count=2,
                display_count=4,
            ),
            "visible": plate_service_module.VideoDetectionStats(
                detection=plate_service_module.PlateDetection(
                    plate_number="XYZ5678",
                    plate_color="钃濈墝",
                    confidence=0.56,
                    bbox=[80, 18, 42, 14],
                ),
                fresh_count=1,
                display_count=3,
            ),
        }
    )

    assert [item.plate_number for item in result] == ["ABC1234", "XYZ5678"]


def test_fallback_vehicle_type_for_small_video_plate_prefers_car_over_bus_or_truck():
    service = PlateService()

    assert (
        service._fallback_vehicle_type_for_bbox(
            plate_service_module.VEHICLE_TYPE_BUS,
            [10, 10, 64, 18],
            video_mode=True,
        )
        == plate_service_module.VEHICLE_TYPE_CAR
    )
    assert (
        service._fallback_vehicle_type_for_bbox(
            plate_service_module.VEHICLE_TYPE_TRUCK,
            [10, 10, 64, 18],
            video_mode=True,
        )
        == plate_service_module.VEHICLE_TYPE_CAR
    )


def test_pick_stable_track_vehicle_type_keeps_car_when_non_car_votes_are_weak():
    service = PlateService()

    result = service._pick_stable_track_vehicle_type(
        {
            plate_service_module.VEHICLE_TYPE_CAR: 1.2,
            plate_service_module.VEHICLE_TYPE_BUS: 1.6,
            plate_service_module.VEHICLE_TYPE_TRUCK: 1.4,
            plate_service_module.VEHICLE_TYPE_PICKUP: 1.1,
        },
        fallback_type=plate_service_module.VEHICLE_TYPE_CAR,
    )

    assert result == plate_service_module.VEHICLE_TYPE_CAR


def test_pick_stable_track_vehicle_type_allows_non_car_when_evidence_is_strong():
    service = PlateService()

    result = service._pick_stable_track_vehicle_type(
        {
            plate_service_module.VEHICLE_TYPE_CAR: 0.9,
            plate_service_module.VEHICLE_TYPE_TRUCK: 3.4,
        },
        fallback_type=plate_service_module.VEHICLE_TYPE_CAR,
    )

    assert result == plate_service_module.VEHICLE_TYPE_TRUCK


def test_collapse_duplicate_recognized_tracks_merges_same_plate_tracks():
    service = PlateService()
    first = plate_service_module.PlateTrack(
        track_id="a",
        plate_number="京M76967",
        plate_color="钃濈墝",
        confidence=0.97,
        bbox=[20, 20, 60, 18],
        template=object(),
        last_seen_frame=10,
        last_recognized_frame=10,
        text_votes={"京M76967": 1.8},
        color_votes={"钃濈墝": 1.8},
        vehicle_type_votes={plate_service_module.VEHICLE_TYPE_JEEP: 1.2},
        vehicle_type=plate_service_module.VEHICLE_TYPE_JEEP,
    )
    second = plate_service_module.PlateTrack(
        track_id="b",
        plate_number="京M76967",
        plate_color="钃濈墝",
        confidence=0.96,
        bbox=[24, 22, 60, 18],
        template=object(),
        last_seen_frame=11,
        last_recognized_frame=11,
        text_votes={"京M76967": 1.6},
        color_votes={"钃濈墝": 1.6},
        vehicle_type_votes={plate_service_module.VEHICLE_TYPE_CAR: 2.4},
        vehicle_type=plate_service_module.VEHICLE_TYPE_CAR,
    )

    result = service._collapse_duplicate_recognized_tracks([first, second])

    assert len(result) == 1
    assert result[0].plate_number == "京M76967"
    assert result[0].vehicle_type == plate_service_module.VEHICLE_TYPE_CAR


def test_tracks_to_detections_merges_same_plate_display_boxes():
    service = PlateService()
    tracks = [
        plate_service_module.PlateTrack(
            track_id="a",
            plate_number="京KS0537",
            plate_color="钃濈墝",
            confidence=0.956,
            bbox=[20, 40, 70, 20],
            template=object(),
            last_seen_frame=10,
            last_recognized_frame=10,
            vehicle_type=plate_service_module.VEHICLE_TYPE_JEEP,
        ),
        plate_service_module.PlateTrack(
            track_id="b",
            plate_number="京KS0537",
            plate_color="钃濈墝",
            confidence=0.930,
            bbox=[22, 72, 72, 22],
            template=object(),
            last_seen_frame=10,
            last_recognized_frame=10,
            vehicle_type=plate_service_module.VEHICLE_TYPE_BUS,
        ),
    ]

    result = service._tracks_to_detections(tracks)

    assert len(result) == 1
    assert result[0].plate_number == "京KS0537"


def test_should_display_unread_track_hides_artifact_near_recognized_plate():
    service = PlateService()
    recognized = plate_service_module.PlateTrack(
        track_id="recognized",
        plate_number="京KS0537",
        plate_color="钃濈墝",
        confidence=0.95,
        bbox=[20, 60, 72, 22],
        template=object(),
        last_seen_frame=10,
        last_recognized_frame=10,
        vehicle_type=plate_service_module.VEHICLE_TYPE_CAR,
    )
    unread = plate_service_module.PlateTrack(
        track_id="artifact",
        plate_number="",
        plate_color="未知",
        confidence=0.52,
        bbox=[18, 28, 74, 20],
        template=object(),
        last_seen_frame=10,
        last_recognized_frame=10,
        unread_observations=3,
    )

    assert service._should_display_unread_track(unread, [recognized, unread]) is False


def test_deduplicate_plate_detections_merges_confusable_text_for_same_bbox():
    service = PlateService()

    result = service._deduplicate_plate_detections(
        [
            plate_service_module.PlateDetection(
                plate_number="73737",
                plate_color="蓝牌",
                confidence=0.68,
                bbox=[20, 20, 60, 18],
            ),
            plate_service_module.PlateDetection(
                plate_number="13131",
                plate_color="蓝牌",
                confidence=0.69,
                bbox=[21, 20, 60, 18],
            ),
        ]
    )

    assert len(result) == 1
    assert result[0].plate_number == "73737"


def test_annotate_frame_supports_chinese_plate_text():
    import numpy as np

    service = PlateService()
    frame = np.zeros((120, 240, 3), dtype=np.uint8)

    annotated = service._annotate_frame(
        frame,
        [
            plate_service_module.PlateDetection(
                plate_number="粤B12345",
                plate_color="蓝牌",
                confidence=0.91,
                bbox=[30, 24, 90, 28],
            )
        ],
    )

    assert annotated.shape == frame.shape
    assert annotated.sum() > frame.sum()
    assert annotated[24, 30].sum() > 0


def test_yolo_detect_video_merges_second_pass_hits(monkeypatch):
    detector = YoloDetector()
    fake_image = object()
    base_hit = {"label": "plate", "kind": "plate", "bbox": [0, 0, 26, 8], "confidence": 0.46}
    second_hit = {"label": "plate", "kind": "plate", "bbox": [80, 10, 28, 10], "confidence": 0.38}
    calls: list[tuple[float, int, float, bool]] = []

    monkeypatch.setattr(detector, "_load_image_array", lambda _source: fake_image)
    monkeypatch.setattr(detector, "_build_second_pass_variants", lambda _image: [(object(), 2.0)])
    monkeypatch.setattr(detector, "_build_small_target_variants", lambda _image: [])
    monkeypatch.setattr(settings, "plate_detector_small_target_enabled", False)
    monkeypatch.setattr(settings, "plate_video_small_target_detector_enabled", False)

    def fake_run_detection_pass(image, *, conf, imgsz, scale_ratio=1.0, include_vehicle_with_plates=False):
        calls.append((conf, imgsz, scale_ratio, include_vehicle_with_plates))
        if image is fake_image:
            return [base_hit]
        return [second_hit]

    monkeypatch.setattr(detector, "_run_detection_pass", fake_run_detection_pass)

    result = detector.detect_video("unused")

    assert len(calls) == 2
    assert all(call[3] is True for call in calls)
    assert base_hit in result
    assert second_hit in result


def test_yolo_detect_video_skips_refinement_for_confident_large_plate(monkeypatch):
    detector = YoloDetector()
    fake_image = object()
    base_hit = {"label": "plate", "kind": "plate", "bbox": [0, 0, 42, 12], "confidence": 0.82}
    calls: list[tuple[float, int, float, bool]] = []

    monkeypatch.setattr(detector, "_load_image_array", lambda _source: fake_image)
    monkeypatch.setattr(detector, "_build_second_pass_variants", lambda _image: [(object(), 2.0)])
    monkeypatch.setattr(detector, "_build_small_target_variants", lambda _image: [(object(), 2.25)])
    monkeypatch.setattr(settings, "plate_detector_small_target_enabled", True)
    monkeypatch.setattr(settings, "plate_video_small_target_detector_enabled", False)

    def fake_run_detection_pass(image, *, conf, imgsz, scale_ratio=1.0, include_vehicle_with_plates=False):
        calls.append((conf, imgsz, scale_ratio, include_vehicle_with_plates))
        return [base_hit]

    monkeypatch.setattr(detector, "_run_detection_pass", fake_run_detection_pass)

    result = detector.detect_video("unused")

    assert len(calls) == 1
    assert result == [base_hit]


def test_yolo_detect_video_keeps_vehicle_hits_alongside_plate_hits(monkeypatch):
    detector = YoloDetector()
    fake_image = object()
    plate_hit = {"label": "plate", "kind": "plate", "bbox": [0, 0, 40, 12], "confidence": 0.82}
    vehicle_hit = {"label": "car", "kind": "vehicle", "bbox": [60, 5, 80, 40], "confidence": 0.55}

    monkeypatch.setattr(detector, "_load_image_array", lambda _source: fake_image)
    monkeypatch.setattr(detector, "_build_second_pass_variants", lambda _image: [])
    monkeypatch.setattr(detector, "_build_small_target_variants", lambda _image: [])
    monkeypatch.setattr(settings, "plate_detector_small_target_enabled", False)
    monkeypatch.setattr(settings, "plate_video_small_target_detector_enabled", False)
    monkeypatch.setattr(
        detector,
        "_run_detection_pass",
        lambda image, *, conf, imgsz, scale_ratio=1.0, include_vehicle_with_plates=False: [plate_hit, vehicle_hit],
    )

    result = detector.detect_video("unused")

    assert plate_hit in result
    assert vehicle_hit in result


def test_yolo_detect_video_runs_small_target_pass(monkeypatch):
    detector = YoloDetector()
    fake_image = object()
    base_hit = {"label": "plate", "kind": "plate", "bbox": [0, 0, 26, 8], "confidence": 0.36}
    small_hit = {"label": "plate", "kind": "plate", "bbox": [90, 16, 20, 8], "confidence": 0.22}
    calls: list[tuple[float, int, float]] = []

    monkeypatch.setattr(detector, "_load_image_array", lambda _source: fake_image)
    monkeypatch.setattr(detector, "_build_second_pass_variants", lambda _image: [])
    monkeypatch.setattr(detector, "_build_small_target_variants", lambda _image: [(object(), 2.25), (object(), 1.0)])
    monkeypatch.setattr(settings, "plate_detector_small_target_enabled", True)
    monkeypatch.setattr(settings, "plate_video_small_target_detector_enabled", False)
    monkeypatch.setattr(settings, "plate_detector_small_target_confidence", 0.10)
    monkeypatch.setattr(settings, "plate_detector_small_target_imgsz", 1536)

    def fake_run_detection_pass(image, *, conf, imgsz, scale_ratio=1.0, include_vehicle_with_plates=False):
        calls.append((conf, imgsz, scale_ratio))
        if image is fake_image:
            return [base_hit]
        return [small_hit]

    monkeypatch.setattr(detector, "_run_detection_pass", fake_run_detection_pass)

    result = detector.detect_video("unused")

    assert len(calls) == 2
    assert calls[1] == (0.10, 1536, 2.25)
    assert base_hit in result
    assert small_hit in result


def test_yolo_detect_image_detailed_runs_more_aggressive_small_target_passes(monkeypatch):
    detector = YoloDetector()
    fake_image = object()
    calls: list[tuple[float, int, float, bool]] = []

    monkeypatch.setattr(detector, "_load_image_array", lambda _source: fake_image)
    monkeypatch.setattr(detector, "_build_second_pass_variants", lambda _image: [(object(), 1.5), (object(), 2.0)])
    monkeypatch.setattr(detector, "_build_aggressive_image_small_target_variants", lambda _image: [(object(), 2.8), (object(), 3.2)])

    def fake_run_detection_pass(image, *, conf, imgsz, scale_ratio=1.0, include_vehicle_with_plates=False):
        calls.append((conf, imgsz, scale_ratio, include_vehicle_with_plates))
        return []

    monkeypatch.setattr(detector, "_run_detection_pass", fake_run_detection_pass)

    detector.detect_image_detailed("unused")

    assert len(calls) == 5
    assert calls[0] == (0.14, 1280, 1.0, True)
    assert calls[1] == (0.10, 1536, 1.5, True)
    assert calls[2] == (0.10, 1536, 2.0, True)
    assert calls[3] == (0.08, 1792, 2.8, True)
    assert calls[4] == (0.08, 1792, 3.2, True)


def test_recognize_detections_via_detector_limits_video_plate_candidates(monkeypatch):
    service = PlateService()
    hits = [
        {"label": "plate", "kind": "plate", "bbox": [index * 10, 0, 30, 10], "confidence": 0.9 - index * 0.01}
        for index in range(8)
    ]
    processed_hits: list[dict] = []

    monkeypatch.setattr(service, "_decode_image_source", lambda _source: object())
    monkeypatch.setattr(service, "_detect_video_detector_hits", lambda image, fast_mode=False: hits)
    monkeypatch.setattr(
        service,
        "_recognize_detector_hit",
        lambda image, hit, **kwargs: processed_hits.append(hit) or [],
    )

    service._recognize_detections_via_detector(object(), video_mode=True, preserve_unread=False, fast_mode=False)

    assert len(processed_hits) == 6


def test_recognize_detections_via_detector_limits_video_vehicle_candidates(monkeypatch):
    service = PlateService()
    hits = [
        {"label": "car", "kind": "vehicle", "bbox": [index * 20, 0, 60, 30], "confidence": 0.8 - index * 0.01}
        for index in range(7)
    ]
    processed_hits: list[dict] = []

    monkeypatch.setattr(service, "_decode_image_source", lambda _source: object())
    monkeypatch.setattr(service, "_detect_video_detector_hits", lambda image, fast_mode=False: hits)
    monkeypatch.setattr(
        service,
        "_recognize_detector_hit",
        lambda image, hit, **kwargs: processed_hits.append(hit) or [],
    )

    service._recognize_detections_via_detector(object(), video_mode=True, preserve_unread=True, fast_mode=False)

    assert len(processed_hits) == 4


def test_video_vehicle_bbox_to_plate_bboxes_adds_extra_candidates():
    import numpy as np

    service = PlateService()
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    vehicle_bbox = [60, 50, 120, 80]

    base_candidates = service._vehicle_bbox_to_plate_bboxes(image, vehicle_bbox)
    video_candidates = service._video_vehicle_bbox_to_plate_bboxes(image, vehicle_bbox)

    assert len(video_candidates) > len(base_candidates)


def test_recognize_detector_hit_uses_multiple_video_vehicle_candidates_in_fast_mode(monkeypatch):
    import numpy as np

    service = PlateService()
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    hit = {"label": "car", "kind": "vehicle", "bbox": [20, 20, 120, 60], "confidence": 0.6}
    tried_bboxes: list[list[int]] = []

    monkeypatch.setattr(
        service,
        "_resolve_detector_crop_bboxes",
        lambda _image, _hit, video_mode=False: [
            [20, 20, 40, 12],
            [24, 40, 46, 14],
            [28, 52, 50, 14],
            [32, 60, 52, 16],
        ],
    )
    monkeypatch.setattr(service, "_crop_detection", lambda _image, bbox: tried_bboxes.append(list(bbox)) or np.zeros((20, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(service, "_build_ocr_crop_variants", lambda crop: [crop])
    monkeypatch.setattr(service, "_recognize_crop_with_paddleocr", lambda *args, **kwargs: None)

    result = service._recognize_detector_hit(
        image,
        hit,
        fast_mode=True,
        preserve_unread=True,
        video_mode=True,
    )

    assert len(tried_bboxes) == 3
    assert len(result) == 1
    assert result[0].bbox == [28, 52, 50, 14]


def test_recognize_detector_hit_disables_heavy_crop_fallback_in_video_fast_mode(monkeypatch):
    import numpy as np

    service = PlateService()
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    hit = {"label": "plate", "kind": "plate", "bbox": [20, 20, 40, 12], "confidence": 0.72}
    captured: list[bool] = []

    monkeypatch.setattr(
        service,
        "_resolve_detector_crop_bboxes",
        lambda _image, _hit, video_mode=False: [[20, 20, 40, 12]],
    )
    monkeypatch.setattr(service, "_crop_detection", lambda _image, _bbox: np.zeros((20, 60, 3), dtype=np.uint8))
    monkeypatch.setattr(service, "_build_ocr_crop_variants", lambda crop: [crop])

    def fake_recognize_crop_with_paddleocr(*args, **kwargs):
        captured.append(bool(kwargs.get("allow_ocr_fallback")))
        return None

    monkeypatch.setattr(service, "_recognize_crop_with_paddleocr", fake_recognize_crop_with_paddleocr)

    service._recognize_detector_hit(
        image,
        hit,
        fast_mode=True,
        preserve_unread=False,
        allow_ocr_fallback=True,
        video_mode=True,
    )

    assert captured == [False]


