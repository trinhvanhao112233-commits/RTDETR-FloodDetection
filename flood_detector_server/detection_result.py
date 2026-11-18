"""Keep display candidates separate from detections that trigger an alert."""

DEFAULT_CONFIDENCE = 0.45


def detection_fields(candidates):
    ranked = sorted(candidates, key=lambda item: item['confidence'], reverse=True)
    detections = [item for item in ranked if item['confidence'] >= DEFAULT_CONFIDENCE]
    # The UI slider starts at 20%. Retain its candidates and always the best box.
    display_candidates = [item for item in ranked if item['confidence'] >= 0.2] or ranked[:1]
    return {
        'detections': detections,
        'candidates': display_candidates,
        'flood_detected': bool(detections),
        'flood_status': 'ngap_lut' if detections else 'khong_ngap_lut',
    }
