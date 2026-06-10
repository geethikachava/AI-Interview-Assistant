import json
import os
import requests

# Free, fast model on Groq.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq(prompt, max_tokens=2000, temperature=0.7):
    """Call Groq (OpenAI-compatible), asking for raw JSON back. Returns text."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("No GROQ_API_KEY set")

    r = requests.post(
        GROQ_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Groq API {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def generate_questions(role="Software Engineer", difficulty="medium",
                       q_type="mixed", count=6):
    count = min(int(count), 10)

    prompt = (
        f"Generate exactly {count} interview questions for a {role} position.\n"
        f"Difficulty: {difficulty}\n"
        f"Type: {q_type} (technical, behavioural, or mixed)\n\n"
        "Respond with a JSON object of the form: {\"questions\": [ ... ]}\n"
        "Each question object must have these keys: id, category, text, type, hint, weight\n"
        "- id: integer starting at 1\n"
        "- type: either \"technical\" or \"behavioural\"\n"
        "- hint: a short time hint like \"2 minutes\"\n"
        "- weight: a decimal; all weights together must sum to 1.0\n\n"
        "Example:\n"
        '{"questions":[{"id":1,"category":"Introduction","text":"Tell me about yourself.",'
        '"type":"behavioural","hint":"2 minutes","weight":0.17}]}'
    )

    try:
        text = _call_groq(prompt, max_tokens=2000)
        data = json.loads(text)

        # Accept {"questions": [...]} or a bare list, just in case
        if isinstance(data, dict):
            questions = data.get("questions") or data.get("items") or []
            if not questions:
                for v in data.values():
                    if isinstance(v, list):
                        questions = v
                        break
        else:
            questions = data

        validated = []
        for i, q in enumerate(questions[:count]):
            validated.append({
                "id":       q.get("id", i + 1),
                "category": q.get("category", "General"),
                "text":     q.get("text", "Please describe your experience."),
                "type":     q.get("type", "behavioural"),
                "hint":     q.get("hint", "2 minutes"),
                "weight":   float(q.get("weight", 1 / count)),
            })

        if not validated:
            raise ValueError("No questions parsed from Groq response")

        # Normalise weights to sum to 1.0
        total = sum(q["weight"] for q in validated)
        for q in validated:
            q["weight"] = round(q["weight"] / max(total, 1e-8), 3)

        print(f"[AI Questions] Groq generated {len(validated)} questions for {role}")
        return validated

    except Exception as e:
        print(f"[AI Questions] Groq failed ({e}) - using fallback questions.")
        return _fallback_questions(role, count)


def _fallback_questions(role="Software Engineer", count=6):
    base = [
        {"id":1,"category":"Introduction",
         "text":f"Introduce yourself and describe your background relevant to {role}.",
         "type":"behavioural","hint":"2 minutes","weight":0.10},
        {"id":2,"category":"Technical",
         "text":f"What are the core technical skills needed for a {role}? How do you rate yourself?",
         "type":"technical","hint":"2 minutes","weight":0.20},
        {"id":3,"category":"Problem Solving",
         "text":"Describe the most challenging technical problem you solved. What was your approach?",
         "type":"technical","hint":"2-3 minutes","weight":0.20},
        {"id":4,"category":"Behavioural",
         "text":"Tell me about a time you worked in a team under pressure. What was the outcome?",
         "type":"behavioural","hint":"2 minutes","weight":0.20},
        {"id":5,"category":"Communication",
         "text":"How do you explain complex technical concepts to non-technical stakeholders?",
         "type":"behavioural","hint":"2 minutes","weight":0.15},
        {"id":6,"category":"Career Goals",
         "text":f"Where do you see yourself in 5 years as a {role}?",
         "type":"behavioural","hint":"1 minute","weight":0.15},
    ]
    return base[:count]
