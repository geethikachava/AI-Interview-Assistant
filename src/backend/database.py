"""
database.py — Updated
New tables added:
  - users.role          (candidate / interviewer)
  - users.profile_photo (path to uploaded photo)
  - interview_sets      (interviewer-created question sets)
  - interview_links     (shareable links)
  - face_logs           (verification attempts)
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "interview_data.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c    = conn.cursor()

    # Users — added role + profile_photo
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT DEFAULT 'candidate',
            profile_photo TEXT DEFAULT NULL,
            face_verified INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    # Candidate profile
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER UNIQUE REFERENCES users(id),
            full_name   TEXT,
            age         INTEGER,
            gender      TEXT,
            education   TEXT,
            occupation  TEXT,
            experience  TEXT,
            updated_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Interviewer profile
    c.execute("""
        CREATE TABLE IF NOT EXISTS interviewers (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER UNIQUE REFERENCES users(id),
            full_name    TEXT,
            company      TEXT,
            designation  TEXT,
            updated_at   TEXT DEFAULT (datetime('now'))
        )
    """)

    # Interview sets created by interviewers
    c.execute("""
        CREATE TABLE IF NOT EXISTS interview_sets (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            interviewer_id INTEGER REFERENCES users(id),
            title        TEXT NOT NULL,
            role         TEXT,
            description  TEXT,
            questions    TEXT,
            link_token   TEXT UNIQUE,
            is_active    INTEGER DEFAULT 1,
            created_at   TEXT DEFAULT (datetime('now'))
        )
    """)

    # Sessions
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER REFERENCES users(id),
            interview_set_id INTEGER DEFAULT NULL,
            started_at    TEXT DEFAULT (datetime('now')),
            completed_at  TEXT,
            overall_score REAL,
            tech_score    REAL,
            soft_score    REAL,
            decision      TEXT,
            session_num   INTEGER DEFAULT 1
        )
    """)

    # Scores
    c.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id             INTEGER REFERENCES sessions(id),
            communication_skill    REAL,
            confidence_level       REAL,
            keywords_matched       REAL,
            work_values            REAL,
            technical_relevancy    REAL,
            eye_contact            REAL,
            pitch_tone             REAL,
            expression             REAL,
            overall_technical      REAL,
            overall_soft           REAL,
            overall_score          REAL
        )
    """)

    # Transcripts
    c.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER REFERENCES sessions(id),
            question_id   INTEGER,
            question_text TEXT,
            answer_text   TEXT,
            tech_score    REAL,
            feedback      TEXT
        )
    """)

    # Face verification logs
    c.execute("""
        CREATE TABLE IF NOT EXISTS face_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES users(id),
            verified    INTEGER DEFAULT 0,
            confidence  REAL DEFAULT 0,
            checked_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] SQLite database ready: {DB_PATH}")


# ── User auth ─────────────────────────────────────────────────────────────────
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()


def create_user(email, password, role="candidate"):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
            (email.lower().strip(), hash_password(password), role)
        )
        conn.commit()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?", (email.lower().strip(),)
        ).fetchone()
        return {"success": True, "user_id": user["id"],
                "email": user["email"], "role": user["role"]}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Email already registered."}
    finally:
        conn.close()


def login_user(email, password):
    conn = get_conn()
    user = conn.execute(
        "SELECT * FROM users WHERE email=? AND password_hash=?",
        (email.lower().strip(), hash_password(password))
    ).fetchone()
    conn.close()
    if user:
        return {"success": True, "user_id": user["id"],
                "email": user["email"], "role": user["role"],
                "profile_photo": user["profile_photo"]}
    return {"success": False, "error": "Invalid email or password."}


def save_profile_photo(user_id, photo_path):
    conn = get_conn()
    conn.execute("UPDATE users SET profile_photo=? WHERE id=?", (photo_path, user_id))
    conn.commit()
    conn.close()


def get_user(user_id):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Candidate profile ─────────────────────────────────────────────────────────
def save_profile(user_id, name, age, gender, education, occupation, experience):
    conn = get_conn()
    ex   = conn.execute("SELECT id FROM candidates WHERE user_id=?", (user_id,)).fetchone()
    if ex:
        conn.execute("""
            UPDATE candidates SET full_name=?,age=?,gender=?,education=?,
            occupation=?,experience=?,updated_at=datetime('now') WHERE user_id=?
        """, (name, age, gender, education, occupation, experience, user_id))
    else:
        conn.execute("""
            INSERT INTO candidates (user_id,full_name,age,gender,education,occupation,experience)
            VALUES (?,?,?,?,?,?,?)
        """, (user_id, name, age, gender, education, occupation, experience))
    conn.commit()
    conn.close()


def get_profile(user_id):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM candidates WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Interviewer profile ───────────────────────────────────────────────────────
def save_interviewer_profile(user_id, name, company, designation):
    conn = get_conn()
    ex   = conn.execute("SELECT id FROM interviewers WHERE user_id=?", (user_id,)).fetchone()
    if ex:
        conn.execute("""
            UPDATE interviewers SET full_name=?,company=?,designation=?,
            updated_at=datetime('now') WHERE user_id=?
        """, (name, company, designation, user_id))
    else:
        conn.execute("""
            INSERT INTO interviewers (user_id,full_name,company,designation)
            VALUES (?,?,?,?)
        """, (user_id, name, company, designation))
    conn.commit()
    conn.close()


def get_interviewer_profile(user_id):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM interviewers WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Interview sets ────────────────────────────────────────────────────────────
import json, secrets

def create_interview_set(interviewer_id, title, role, description, questions):
    conn  = get_conn()
    token = secrets.token_urlsafe(12)
    conn.execute("""
        INSERT INTO interview_sets
        (interviewer_id,title,role,description,questions,link_token)
        VALUES (?,?,?,?,?,?)
    """, (interviewer_id, title, role, description,
          json.dumps(questions), token))
    conn.commit()
    row = conn.execute(
        "SELECT * FROM interview_sets WHERE link_token=?", (token,)
    ).fetchone()
    conn.close()
    return dict(row)


def get_interview_sets(interviewer_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM interview_sets WHERE interviewer_id=? ORDER BY created_at DESC",
        (interviewer_id,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["questions"] = json.loads(d["questions"]) if d["questions"] else []
        result.append(d)
    return result


def get_interview_set_by_token(token):
    conn = get_conn()
    row  = conn.execute(
        "SELECT * FROM interview_sets WHERE link_token=? AND is_active=1", (token,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["questions"] = json.loads(d["questions"]) if d["questions"] else []
        return d
    return None


def delete_interview_set(set_id, interviewer_id):
    conn = get_conn()
    conn.execute(
        "UPDATE interview_sets SET is_active=0 WHERE id=? AND interviewer_id=?",
        (set_id, interviewer_id)
    )
    conn.commit()
    conn.close()


# ── Sessions ──────────────────────────────────────────────────────────────────
def create_session(user_id, interview_set_id=None):
    conn  = get_conn()
    count = conn.execute(
        "SELECT COUNT(*) as c FROM sessions WHERE user_id=?", (user_id,)
    ).fetchone()["c"]
    cur = conn.execute(
        "INSERT INTO sessions (user_id,interview_set_id,session_num) VALUES (?,?,?)",
        (user_id, interview_set_id, count + 1)
    )
    sid = cur.lastrowid
    conn.commit()
    conn.close()
    return sid


def complete_session(session_id, overall, tech, soft, decision):
    conn = get_conn()
    conn.execute("""
        UPDATE sessions SET completed_at=datetime('now'),
        overall_score=?,tech_score=?,soft_score=?,decision=? WHERE id=?
    """, (overall, tech, soft, decision, session_id))
    conn.commit()
    conn.close()


def save_scores(session_id, scores):
    conn = get_conn()
    conn.execute("""
        INSERT INTO scores
        (session_id,communication_skill,confidence_level,keywords_matched,
         work_values,technical_relevancy,eye_contact,pitch_tone,expression,
         overall_technical,overall_soft,overall_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session_id,
        scores.get("communication_skill", 0),
        scores.get("confidence_level",    0),
        scores.get("keywords_matched",    0),
        scores.get("work_values",         0),
        scores.get("technical_relevancy", 0),
        scores.get("eye_contact",         0),
        scores.get("pitch_tone",          0),
        scores.get("expression",          0),
        scores.get("overall_technical",   0),
        scores.get("overall_soft",        0),
        scores.get("overall_score",       0),
    ))
    conn.commit()
    conn.close()


def save_transcript(session_id, qid, question, answer, tech_score, feedback):
    conn = get_conn()
    conn.execute("""
        INSERT INTO transcripts
        (session_id,question_id,question_text,answer_text,tech_score,feedback)
        VALUES (?,?,?,?,?,?)
    """, (session_id, qid, question, answer, tech_score, feedback))
    conn.commit()
    conn.close()


# ── Face logs ─────────────────────────────────────────────────────────────────
def log_face_verification(user_id, verified, confidence):
    conn = get_conn()
    conn.execute(
        "INSERT INTO face_logs (user_id,verified,confidence) VALUES (?,?,?)",
        (user_id, int(verified), confidence)
    )
    conn.commit()
    conn.close()


# ── History ───────────────────────────────────────────────────────────────────
def get_history(user_id):
    conn = get_conn()
    rows = conn.execute("""
        SELECT s.*, sc.communication_skill, sc.confidence_level,
               sc.keywords_matched, sc.work_values, sc.technical_relevancy
        FROM sessions s
        LEFT JOIN scores sc ON sc.session_id = s.id
        WHERE s.user_id=? AND s.completed_at IS NOT NULL
        ORDER BY s.started_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_improvement(user_id):
    history = get_history(user_id)
    if len(history) < 2:
        return {"has_comparison": False,
                "message": "Complete more interviews to see improvement."}
    latest = history[0]
    prev   = history[1]
    def diff(k): return round((latest.get(k) or 0) - (prev.get(k) or 0), 2)
    od = diff("overall_score")
    trend = "improved" if od > 0.5 else ("declined" if od < -0.5 else "stable")
    msg   = (f"Great progress! Score improved by {od:.1f} points." if trend == "improved"
             else f"Score dropped by {abs(od):.1f} points. Practice more." if trend == "declined"
             else "Performance is consistent. Push harder to improve.")
    return {
        "has_comparison":  True, "trend": trend, "message": msg,
        "overall_diff":    od,
        "tech_diff":       diff("tech_score"),
        "soft_diff":       diff("soft_score"),
        "latest_session":  latest["session_num"],
        "latest_overall":  latest["overall_score"],
        "prev_overall":    prev["overall_score"],
    }