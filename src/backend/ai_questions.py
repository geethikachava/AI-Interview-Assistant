"""
ai_questions.py
───────────────
Generates interview questions using Claude API (claude-sonnet-4-20250514).
No API key needed in code — handled by Anthropic proxy.

Generates questions based on:
  - Job role / title
  - Difficulty level
  - Question type (technical / behavioural / mixed)
  - Number of questions
"""

import json
import requests


def generate_questions(
    role: str,
    difficulty: str = "medium",
    q_type: str = "mixed",
    count: int = 6
) -> list:
    """
    Generate interview questions using Claude API.

    Args:
        role:       Job role e.g. "Software Engineer", "Data Scientist"
        difficulty: "easy" / "medium" / "hard"
        q_type:     "technical" / "behavioural" / "mixed"
        count:      Number of questions (max 10)

    Returns:
        List of dicts: [{id, category, text, type, hint, weight}]
    """
    count = min(count, 10)

    prompt = f"""Generate exactly {count} interview questions for a {role} position.

Difficulty level: {difficulty}
Question type: {q_type}

Return ONLY a valid JSON array. No explanation, no markdown, no extra text.
Each question must have these exact fields:
  - id: number (1 to {count})
  - category: string (e.g. "Technical", "Behavioural", "System Design", "Algorithms", "Data Structures", "Problem Solving", "Career Goals", "Communication")
  - text: the full question text
  - type: "technical" or "behavioural"
  - hint: suggested answer time e.g. "2 minutes"
  - weight: decimal that all sum to 1.0 (distribute evenly)

Example format:
[
  {{
    "id": 1,
    "category": "Introduction",
    "text": "Tell me about yourself and your experience.",
    "type": "behavioural",
    "hint": "2 minutes",
    "weight": 0.17
  }}
]

Generate questions appropriate for {difficulty} difficulty {role} interview.
"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages":   [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code != 200:
            print(f"[AI Questions] API error: {response.status_code}")
            return _fallback_questions(role, count)

        data    = response.json()
        content = data["content"][0]["text"].strip()

        # Clean up response — remove markdown fences if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        questions = json.loads(content)

        # Validate structure
        validated = []
        for i, q in enumerate(questions[:count]):
            validated.append({
                "id":       q.get("id", i+1),
                "category": q.get("category", "General"),
                "text":     q.get("text", "Please describe your experience."),
                "type":     q.get("type", "behavioural"),
                "hint":     q.get("hint", "2 minutes"),
                "weight":   float(q.get("weight", 1/count))
            })

        # Normalise weights to sum to 1.0
        total = sum(q["weight"] for q in validated)
        for q in validated:
            q["weight"] = round(q["weight"] / total, 3)

        print(f"[AI Questions] Generated {len(validated)} questions for {role}")
        return validated

    except json.JSONDecodeError as e:
        print(f"[AI Questions] JSON parse error: {e}")
        return _fallback_questions(role, count)
    except Exception as e:
        print(f"[AI Questions] Error: {e}")
        return _fallback_questions(role, count)


def _fallback_questions(role: str, count: int = 6) -> list:
    """
    Fallback questions if API fails.
    Generic questions that work for any role.
    """
    base = [
        {"id":1, "category":"Introduction",    "text":f"Introduce yourself and describe your experience relevant to the {role} role.", "type":"behavioural", "hint":"2 minutes",   "weight":0.10},
        {"id":2, "category":"Technical",       "text":f"What are the core technical skills required for a {role}? How do you rate yourself?", "type":"technical",   "hint":"2 minutes",   "weight":0.20},
        {"id":3, "category":"Problem Solving", "text":"Describe the most challenging technical problem you solved. What was your approach?", "type":"technical",   "hint":"2-3 minutes", "weight":0.20},
        {"id":4, "category":"Behavioural",     "text":"Tell me about a time you worked in a team under pressure. What was the outcome?",     "type":"behavioural", "hint":"2 minutes",   "weight":0.20},
        {"id":5, "category":"Communication",   "text":"How do you explain complex technical concepts to non-technical stakeholders?",         "type":"behavioural", "hint":"2 minutes",   "weight":0.15},
        {"id":6, "category":"Career Goals",    "text":f"Where do you see yourself in 5 years as a {role}?",                                   "type":"behavioural", "hint":"1 minute",    "weight":0.15},
    ]
    return base[:count]