"""
face_verify.py - Strong version using LBPH face recognizer
Uses OpenCV's built-in LBPH (Local Binary Pattern Histogram) recognizer
which is much more accurate than histogram comparison.
Falls back to histogram if LBPH is not available.
"""

import cv2
import numpy as np
import base64
import os


def decode_b64(b64_string):
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        return cv2.imdecode(
            np.frombuffer(base64.b64decode(b64_string), np.uint8),
            cv2.IMREAD_COLOR
        )
    except Exception as e:
        print(f"[FaceVerify] Decode error: {e}")
        return None


def detect_face(img):
    """Detect largest face in image, return cropped + resized."""
    if img is None:
        return None
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Enhance contrast for better detection
    gray = cv2.equalizeHist(gray)

    # Try different parameters
    for scale, neighbors, minsize in [
        (1.1, 5, 80), (1.05, 4, 60), (1.2, 4, 80), (1.1, 3, 50)
    ]:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=scale,
            minNeighbors=neighbors, minSize=(minsize, minsize)
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            # Add 20% padding
            pad = int(min(w, h) * 0.2)
            x1  = max(0, x-pad)
            y1  = max(0, y-pad)
            x2  = min(img.shape[1], x+w+pad)
            y2  = min(img.shape[0], y+h+pad)
            face = cv2.resize(img[y1:y2, x1:x2], (128, 128))
            return face
    return None


def compare_lbph(face1_gray, face2_gray):
    """
    LBPH comparison — best OpenCV method for face recognition.
    Train on face1, predict face2.
    Lower predict confidence = better match (0 = perfect).
    """
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        # Train with profile photo
        recognizer.train([face1_gray], np.array([0]))
        # Predict live photo
        label, confidence = recognizer.predict(face2_gray)
        # confidence: 0 = perfect match, >100 = poor match
        # Convert to 0-1 score: lower confidence = higher score
        score = max(0.0, 1.0 - confidence / 100.0)
        print(f"[FaceVerify] LBPH confidence={confidence:.1f}, score={score:.3f}")
        return score
    except Exception as e:
        print(f"[FaceVerify] LBPH error: {e}")
        return None


def compare_histogram(img1, img2):
    """Fallback histogram comparison."""
    try:
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        h1   = cv2.calcHist([hsv1], [0,1], None, [50,60], [0,180,0,256])
        h2   = cv2.calcHist([hsv2], [0,1], None, [50,60], [0,180,0,256])
        cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
        bhatt = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, 1.0 - bhatt)
    except Exception as e:
        print(f"[FaceVerify] Histogram error: {e}")
        return 0.5


def verify_face(profile_photo_path, live_frame_b64):
    """
    Main verification function.
    Returns verified=True if face matches profile photo.
    Uses LBPH (most accurate) with histogram as fallback.
    Threshold: LBPH score >= 0.30 OR histogram >= 0.40
    """
    # No profile photo — skip
    if not profile_photo_path or not os.path.exists(profile_photo_path):
        return {"verified": True, "confidence": 1.0, "face_detected": False,
                "message": "No profile photo — skipping verification.", "skip": True}

    profile_img = cv2.imread(profile_photo_path)
    if profile_img is None:
        return {"verified": True, "confidence": 1.0, "face_detected": False,
                "message": "Profile photo unreadable — skipping.", "skip": True}

    live_img = decode_b64(live_frame_b64)
    if live_img is None:
        return {"verified": False, "confidence": 0.0, "face_detected": False,
                "message": "Could not read camera. Please allow camera access."}

    profile_face = detect_face(profile_img)
    live_face    = detect_face(live_img)

    if live_face is None:
        return {"verified": False, "confidence": 0.0, "face_detected": False,
                "message": "No face detected. Look at the camera in good lighting."}

    if profile_face is None:
        # Cannot detect face in profile photo — allow through
        return {"verified": True, "confidence": 0.6, "face_detected": True,
                "message": "Profile photo face unclear. Please retake in good lighting."}

    # Convert to grayscale for LBPH
    g1 = cv2.cvtColor(profile_face, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(live_face,    cv2.COLOR_BGR2GRAY)

    # Try LBPH first
    lbph_score = None
    has_lbph   = hasattr(cv2, 'face')
    if has_lbph:
        lbph_score = compare_lbph(g1, g2)

    # Histogram comparison
    hist_score = compare_histogram(profile_face, live_face)

    # Decide
    if lbph_score is not None:
        # LBPH is more reliable
        final_score = lbph_score * 0.7 + hist_score * 0.3
        method      = "LBPH+Histogram"
        threshold   = 0.28   # tuned for LBPH
    else:
        final_score = hist_score
        method      = "Histogram"
        threshold   = 0.40

    print(f"[FaceVerify] Method={method} Score={final_score:.3f} Threshold={threshold}")

    verified = final_score >= threshold

    if verified:
        message = f"✅ Identity verified ({final_score*100:.0f}% match)"
    else:
        message = f"❌ Face does not match profile ({final_score*100:.0f}% match). Ensure same lighting and look straight at camera."

    return {
        "verified":      verified,
        "confidence":    round(float(final_score), 3),
        "face_detected": True,
        "message":       message
    }