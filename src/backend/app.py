"""
app.py — Complete fixed version
Fixes:
  1. Face verification more lenient
  2. Auto next question (handled in interview.html)
  3. Speech to text fixed via better transcriber.py
  4. Shareable links work correctly for candidates only
  5. After login redirect to pending interview link
"""

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, send_from_directory
)
import os, uuid, json
import numpy as np

from database import (
    init_db, create_user, login_user,
    save_profile, get_profile,
    save_interviewer_profile, get_interviewer_profile,
    save_profile_photo, get_user,
    create_interview_set, get_interview_sets,
    get_interview_set_by_token, delete_interview_set,
    create_session, complete_session, save_scores,
    save_transcript, get_history, get_improvement,
    log_face_verification, get_conn
)

app = Flask(__name__)
app.secret_key = "ai_interview_secret_key_2025"

UPLOAD_FOLDER = "uploads"
PHOTO_FOLDER  = "profile_photos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PHOTO_FOLDER,  exist_ok=True)

_transcriber = None
_scorer      = None

def get_transcriber():
    global _transcriber
    if _transcriber is None:
        from transcriber import Transcriber
        _transcriber = Transcriber(model_size="base")
    return _transcriber

def get_scorer():
    global _scorer
    if _scorer is None:
        from web_scorer import WebScorer
        _scorer = WebScorer()
    return _scorer

# ── SERVE PROFILE PHOTOS ──────────────────────────────────────────────────────
@app.route("/profile_photos/<filename>")
def serve_photo(filename):
    return send_from_directory(PHOTO_FOLDER, filename)

# ── AUTH ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" in session:
        role = session.get("role", "candidate")
        return redirect(url_for("interviewer_dashboard") if role == "interviewer"
                        else url_for("profile_page"))
    return render_template("index.html")

@app.route("/signup", methods=["POST"])
def signup():
    data   = request.get_json()
    role   = data.get("role", "candidate")
    result = create_user(data["email"], data["password"], role)
    if result["success"]:
        session["user_id"] = result["user_id"]
        session["email"]   = result["email"]
        session["role"]    = result["role"]
        pending = session.pop("pending_token", None)
        if pending and role == "candidate":
            result["redirect"] = f"/attend/{pending}"
        elif role == "interviewer":
            result["redirect"] = "/interviewer_profile"
        else:
            result["redirect"] = "/profile"
    return jsonify(result)

@app.route("/login", methods=["POST"])
def login():
    data   = request.get_json()
    result = login_user(data["email"], data["password"])
    if result["success"]:
        session["user_id"] = result["user_id"]
        session["email"]   = result["email"]
        session["role"]    = result["role"]
        # Redirect to pending interview if candidate came from a link
        pending = session.pop("pending_token", None)
        if pending and result["role"] == "candidate":
            result["redirect"] = f"/attend/{pending}"
        elif result["role"] == "interviewer":
            result["redirect"] = "/interviewer"
        else:
            result["redirect"] = "/profile"
    return jsonify(result)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ── PROFILE PHOTO ─────────────────────────────────────────────────────────────
@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data     = request.get_json()
    b64_data = data.get("image")
    if not b64_data:
        return jsonify({"error": "No image provided"}), 400
    try:
        import base64, cv2
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
        img_bytes = base64.b64decode(b64_data)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        filename = f"user_{session['user_id']}_photo.jpg"
        filepath = os.path.join(PHOTO_FOLDER, filename)
        cv2.imwrite(filepath, img)
        save_profile_photo(session["user_id"], filepath)
        return jsonify({"success": True, "photo_url": f"/profile_photos/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── FACE VERIFICATION ─────────────────────────────────────────────────────────
@app.route("/verify_face", methods=["POST"])
def verify_face_route():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data      = request.get_json()
    frame_b64 = data.get("frame")
    user      = get_user(session["user_id"])
    if not user or not user.get("profile_photo"):
        return jsonify({"verified": True, "confidence": 1.0,
                        "message": "No profile photo set — skipping verification.",
                        "skip": True})
    try:
        from face_verify import verify_face
        result = verify_face(user["profile_photo"], frame_b64)
        log_face_verification(session["user_id"], result["verified"], result["confidence"])
        return jsonify(result)
    except Exception as e:
        print(f"[FaceVerify] Error: {e}")
        return jsonify({"verified": True, "confidence": 0.5,
                        "message": "Verification skipped due to error.",
                        "skip": True})

# ── CANDIDATE PROFILE ─────────────────────────────────────────────────────────
@app.route("/profile", methods=["GET", "POST"])
def profile_page():
    if "user_id" not in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        data = request.get_json()
        save_profile(session["user_id"], data.get("full_name"), data.get("age"),
                     data.get("gender"), data.get("education"),
                     data.get("occupation"), data.get("experience"))
        return jsonify({"success": True})
    profile = get_profile(session["user_id"])
    user    = get_user(session["user_id"])
    return render_template("profile.html", profile=profile, user=user,
                           email=session.get("email"), role=session.get("role","candidate"))

# ── INTERVIEWER PROFILE ───────────────────────────────────────────────────────
@app.route("/interviewer_profile", methods=["GET", "POST"])
def interviewer_profile():
    if "user_id" not in session or session.get("role") != "interviewer":
        return redirect(url_for("index"))
    if request.method == "POST":
        data = request.get_json()
        save_interviewer_profile(session["user_id"], data.get("full_name"),
                                 data.get("company"), data.get("designation"))
        return jsonify({"success": True})
    profile = get_interviewer_profile(session["user_id"])
    user    = get_user(session["user_id"])
    return render_template("interviewer_profile.html", profile=profile,
                           user=user, email=session.get("email"))

# ── INTERVIEWER DASHBOARD ─────────────────────────────────────────────────────
@app.route("/interviewer")
def interviewer_dashboard():
    if "user_id" not in session or session.get("role") != "interviewer":
        return redirect(url_for("index"))
    sets    = get_interview_sets(session["user_id"])
    profile = get_interviewer_profile(session["user_id"])
    user    = get_user(session["user_id"])
    return render_template("interviewer_dashboard.html", sets=sets,
                           profile=profile, user=user,
                           email=session.get("email"),
                           base_url=request.host_url)

@app.route("/create_interview", methods=["POST"])
def create_interview():
    if "user_id" not in session or session.get("role") != "interviewer":
        return jsonify({"error": "Not authorized"}), 403
    data      = request.get_json()
    questions = data.get("questions", [])
    if not questions:
        return jsonify({"error": "No questions provided"}), 400
    iset = create_interview_set(
        session["user_id"], data.get("title","Interview"),
        data.get("role","General"), data.get("description",""), questions
    )
    link = f"{request.host_url}attend/{iset['link_token']}"
    return jsonify({"success": True, "link": link,
                    "token": iset["link_token"], "id": iset["id"]})

@app.route("/delete_interview/<int:set_id>", methods=["POST"])
def delete_interview(set_id):
    if "user_id" not in session or session.get("role") != "interviewer":
        return jsonify({"error": "Not authorized"}), 403
    delete_interview_set(set_id, session["user_id"])
    return jsonify({"success": True})

# ── AI QUESTION GENERATION ────────────────────────────────────────────────────
@app.route("/generate_questions", methods=["POST"])
def generate_questions_route():
    if "user_id" not in session or session.get("role") != "interviewer":
        return jsonify({"error": "Not authorized"}), 403
    data = request.get_json()
    try:
        from ai_questions import generate_questions
        questions = generate_questions(
            data.get("role", "Software Engineer"),
            data.get("difficulty", "medium"),
            data.get("q_type", "mixed"),
            int(data.get("count", 6))
        )
        return jsonify({"success": True, "questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ATTEND INTERVIEW VIA LINK (FIXED) ─────────────────────────────────────────
@app.route("/attend/<token>")
def attend_interview(token):
    # Validate token
    iset = get_interview_set_by_token(token)
    if not iset:
        return render_template("error.html",
                               message="This interview link is invalid or has expired.")

    # Not logged in — save token, redirect to login page
    if "user_id" not in session:
        session["pending_token"] = token
        return redirect(url_for("index"))

    # Interviewers blocked
    if session.get("role") == "interviewer":
        return render_template("error.html",
                               message="Interviewers cannot attend interviews. Please login as a candidate.")

    # Must have a profile
    profile = get_profile(session["user_id"])
    if not profile:
        session["pending_token"] = token
        return redirect(url_for("profile_page"))

    user = get_user(session["user_id"])
    return render_template("interview.html",
                           profile=profile, user=user,
                           email=session.get("email"),
                           questions=json.dumps(iset["questions"]),
                           interview_title=iset["title"],
                           interview_set_id=iset["id"])

# ── DEFAULT INTERVIEW ─────────────────────────────────────────────────────────
@app.route("/interview")
def interview_page():
    if "user_id" not in session:
        return redirect(url_for("index"))
    if session.get("role") == "interviewer":
        return redirect(url_for("interviewer_dashboard"))
    profile = get_profile(session["user_id"])
    if not profile:
        return redirect(url_for("profile_page"))
    user = get_user(session["user_id"])
    default_questions = [
        {"id":1,"category":"Introduction",   "text":"Introduce yourself — background, skills and projects.","type":"behavioural","hint":"1-2 min","weight":0.10},
        {"id":2,"category":"Data Structures","text":"Explain stack vs queue with real-world use cases.","type":"technical","hint":"2 min","weight":0.20},
        {"id":3,"category":"System Design",  "text":"Design a URL shortener like bit.ly. Explain architecture.","type":"technical","hint":"2-3 min","weight":0.20},
        {"id":4,"category":"Algorithms",     "text":"Explain time complexity. What is O(n log n)?","type":"technical","hint":"2 min","weight":0.20},
        {"id":5,"category":"Problem Solving","text":"Describe a technical challenge you solved.","type":"behavioural","hint":"2 min","weight":0.15},
        {"id":6,"category":"Career Goals",   "text":"Where do you see yourself in 5 years?","type":"behavioural","hint":"1 min","weight":0.15},
    ]
    return render_template("interview.html",
                           profile=profile, user=user,
                           email=session.get("email"),
                           questions=json.dumps(default_questions),
                           interview_title="CSE Interview",
                           interview_set_id=None)

# ── UPLOAD AUDIO ──────────────────────────────────────────────────────────────
@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    q_id  = request.form.get("question_id", "0")
    sid   = request.form.get("session_id",  "0")
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio"}), 400
    filename = f"uid{session['user_id']}_sid{sid}_q{q_id}_{uuid.uuid4().hex[:6]}.webm"
    path     = os.path.join(UPLOAD_FOLDER, filename)
    audio.save(path)
    size = os.path.getsize(path)
    print(f"[Upload] {filename} ({size} bytes)")
    return jsonify({"success": True, "path": path})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data     = request.get_json()
    filepath = data.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"success": True, "transcript": ""})
    try:
        t          = get_transcriber()
        transcript = t.transcribe_file(filepath)
        return jsonify({"success": True, "transcript": transcript})
    except Exception as e:
        print(f"[Transcribe error] {e}")
        return jsonify({"success": True, "transcript": ""})

@app.route("/evaluate", methods=["POST"])
def evaluate():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data    = request.get_json()
    answers = data.get("answers", [])
    set_id  = data.get("interview_set_id")
    sess_id = create_session(session["user_id"], set_id)
    try:
        scorer  = get_scorer()
        results = scorer.score_all(answers)
        save_scores(sess_id, results["scores"])
        for a in answers:
            qid  = a.get("question_id", 0)
            tech = results["per_question"].get(str(qid), {})
            save_transcript(sess_id, qid, a.get("question",""),
                          a.get("transcript",""), tech.get("overall",0),
                          tech.get("feedback",""))
        complete_session(sess_id, results["overall_score"],
                        results["scores"]["overall_technical"],
                        results["scores"]["overall_soft"], results["decision"])
        return jsonify({"success": True, "session_id": sess_id,
                       "results": results,
                       "improvement": get_improvement(session["user_id"])})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ── RESULTS & HISTORY ─────────────────────────────────────────────────────────
@app.route("/results/<int:session_id>")
def results_page(session_id):
    if "user_id" not in session:
        return redirect(url_for("index"))
    return render_template("results.html", session_id=session_id,
                           email=session.get("email"))

@app.route("/history")
def history_page():
    if "user_id" not in session:
        return redirect(url_for("index"))
    return render_template("history.html",
                          history=get_history(session["user_id"]),
                          improvement=get_improvement(session["user_id"]),
                          email=session.get("email"))

@app.route("/api/session_results/<int:session_id>")
def api_session_results(session_id):
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db   = get_conn()
    sc   = db.execute("SELECT * FROM scores WHERE session_id=?", (session_id,)).fetchone()
    sess = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    trs  = db.execute("SELECT * FROM transcripts WHERE session_id=? ORDER BY question_id", (session_id,)).fetchall()
    db.close()
    if not sess:
        return jsonify({"error": "Not found"}), 404
    sess_dict = dict(sess)
    sc_dict   = dict(sc) if sc else {}
    from web_scorer import get_decision as wd
    decision, icon = wd(sess_dict.get("overall_score") or 0)
    per_q = {}
    for t in trs:
        td = dict(t)
        per_q[str(td["question_id"])] = {
            "overall": td.get("tech_score", 0),
            "feedback": td.get("feedback",""),
            "communication":0,"confidence":0,
            "keywords_matched":0,"work_values":0,"technical_relevancy":0,
        }
    return jsonify({
        "results": {
            "scores": sc_dict,
            "overall_score": sess_dict.get("overall_score") or 0,
            "decision": decision, "decision_icon": icon,
            "suggestions": [], "per_question": per_q,
        },
        "improvement": get_improvement(session["user_id"])
    })

@app.route("/api/history")
def api_history():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify({"history": get_history(session["user_id"]),
                   "improvement": get_improvement(session["user_id"])})

if __name__ == "__main__":
    init_db()
    print("\n" + "="*55)
    print("  AI INTERVIEW SYSTEM")
    print("  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)