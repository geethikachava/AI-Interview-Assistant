import numpy as np
import json
import os

# Free, fast model on Groq.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ANSWER_KEYS = {
    1: {
        "keyword_groups": [
            ["experience","worked","background","years","built","developed"],
            ["python","java","javascript","react","sql","language","framework","coding"],
            ["project","developed","created","designed","built","made","implemented"],
            ["team","collaborate","agile","scrum","together","group"],
            ["backend","frontend","fullstack","database","cloud","web","app","software"],
        ],
        "weight": 0.10,
        "work_value_words": ["team","learn","grow","contribute","passion","problem","improve","build"],
    },
    2: {
        "keyword_groups": [
            ["stack","stacks"],
            ["queue","queues","waiting"],
            ["lifo","last in","last-in","top","pile","last added"],
            ["fifo","first in","first-in","line","order","first added"],
            ["recursion","bfs","scheduling","undo","call stack","browser","task"],
        ],
        "weight": 0.20,
        "work_value_words": ["understand","explain","example","use case","real","difference"],
    },
    3: {
        "keyword_groups": [
            ["hash","encode","shorten","generate","unique","key","id"],
            ["database","store","storage","mapping","table","sql","nosql"],
            ["redirect","short url","long url","original","302","301","link"],
            ["cache","redis","performance","fast","memory","cdn"],
            ["scale","distributed","load","traffic","million","server"],
        ],
        "weight": 0.20,
        "work_value_words": ["scale","efficient","design","architecture","trade-off","solution"],
    },
    4: {
        "keyword_groups": [
            ["time complexity","big o","o(n","complexity","efficient","performance"],
            ["n log n","nlogn","merge sort","heap sort","quick sort","sorting"],
            ["linear","quadratic","constant","logarithmic","polynomial"],
            ["algorithm","sort","search","optimize","loop","iteration"],
            ["best case","worst case","average","space complexity","operations"],
        ],
        "weight": 0.20,
        "work_value_words": ["analyze","optimize","efficiency","understand","improve","measure"],
    },
    5: {
        "keyword_groups": [
            ["problem","challenge","issue","difficult","bug","error"],
            ["debug","investigate","diagnose","identify","found","noticed"],
            ["solution","fixed","resolved","solved","implemented","changed"],
            ["team","collaborate","together","help","support","colleague"],
            ["result","outcome","improved","learned","impact","better"],
        ],
        "weight": 0.15,
        "work_value_words": ["team","ownership","responsibility","initiative","learned","grew"],
    },
    6: {
        "keyword_groups": [
            ["senior","lead","architect","grow","advancement","promotion"],
            ["learn","skill","improve","develop","knowledge","master"],
            ["contribute","impact","build","create","ownership","meaningful"],
            ["goal","vision","career","future","plan","aspire"],
            ["role","company","team","opportunity","fit","align"],
        ],
        "weight": 0.15,
        "work_value_words": ["growth","impact","learn","contribute","long-term","vision","passionate"],
    },
}

DECISION_LABELS = [
    (9.0, "Exceptional Hire", "star"),
    (8.0, "Strong Hire",      "check"),
    (7.0, "Hire",             "check"),
    (5.5, "Borderline",       "warning"),
    (0.0, "No Hire",          "cross"),
]

DECISION_ICONS = {
    "star":    "⭐",
    "check":   "✅",
    "warning": "⚠️",
    "cross":   "❌",
}


def get_decision(score):
    for t, l, i in DECISION_LABELS:
        if score >= t:
            return l, DECISION_ICONS[i]
    return "No Hire", "❌"


class WebScorer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if self.api_key:
            print("[Scorer] Groq API key found - will use AI scoring.")
        else:
            print("[Scorer] No GROQ_API_KEY - using keyword scoring.")

    def _score_with_groq(self, question, answer):
        api_key = os.getenv("GROQ_API_KEY", "") or self.api_key
        if not api_key or not answer or len(answer.strip()) < 5:
            return None
        try:
            import requests
            prompt = (
                "You are evaluating a job interview answer. "
                "Score it on 5 dimensions, each from 0.0 to 10.0.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}\n\n"
                "Respond with a JSON object with exactly these keys:\n"
                '{"communication": 7.5, "confidence": 6.0, '
                '"keywords_matched": 8.0, "work_values": 7.0, '
                '"technical_relevancy": 7.5, '
                '"feedback": "One sentence of feedback."}'
            )
            r = requests.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                },
                timeout=30,
            )
            if r.status_code != 200:
                print(f"[Scorer] Groq API error {r.status_code}: {r.text[:150]}")
                return None
            text = r.json()["choices"][0]["message"]["content"].strip()
            result = json.loads(text)
            result["overall"] = round(
                (
                    result.get("communication", 5) +
                    result.get("confidence", 5) +
                    result.get("keywords_matched", 5) +
                    result.get("work_values", 5) +
                    result.get("technical_relevancy", 5)
                ) / 5.0,
                2,
            )
            result["keyword_hits"] = 0
            result["question"] = question
            return result
        except Exception as e:
            print(f"[Scorer] Gemini error: {e}")
            return None

    def _score_keywords(self, qid, answer, question_text):
        """Keyword-based fallback scoring."""
        if not answer or not answer.strip() or len(answer.strip()) < 3:
            return {
                "overall": 0.0, "keyword_hits": 0, "question": question_text,
                "feedback": "No answer detected. Speak clearly into the microphone.",
                "communication": 0.0, "confidence": 0.0,
                "keywords_matched": 0.0, "work_values": 0.0,
                "technical_relevancy": 0.0,
            }

        key = ANSWER_KEYS.get(qid, ANSWER_KEYS[1])
        answer_lower = answer.lower()
        words = answer.split()
        word_count = len(words)

        groups = key["keyword_groups"]
        group_hits = sum(
            1 for g in groups if any(kw in answer_lower for kw in g)
        )
        keywords_score = round(min(3.0 + group_hits * 1.4, 10.0), 2)

        density = group_hits / max(len(groups), 1)
        length_bonus = min(word_count / 30.0, 1.5)
        tech_rel = round(min(2.0 + density * 6.0 + length_bonus, 10.0), 2)

        fillers = sum(
            answer_lower.count(w)
            for w in ["um ", " uh ", " like ", "you know", "basically"]
        )
        connectors = sum(
            1 for w in
            ["first", "then", "finally", "also", "because",
             "therefore", "next", "additionally"]
            if w in answer_lower
        )
        has_example = any(
            p in answer_lower
            for p in ["for example", "for instance", "such as",
                      "in my project", "i built", "we used"]
        )
        filler_pen = max(2.0, 9.0 - fillers * 1.5)
        struct_sc = min(2.0 + connectors * 1.5, 9.0)
        length_sc = min(2.0 + word_count / 20.0, 9.5)
        example_sc = 8.5 if has_example else 5.5
        comm_score = round(
            (filler_pen + struct_sc + length_sc + example_sc) / 4.0, 2
        )

        hedges = sum(
            answer_lower.count(w)
            for w in ["maybe ", "i think ", "i guess ", "not sure", "perhaps"]
        )
        assertive = sum(
            1 for w in
            ["i know", "i have", "i built", "i designed",
             "i implemented", "i created", "i solved", "i developed"]
            if w in answer_lower
        )
        conf_score = round(
            float(np.clip(5.5 + assertive * 0.7 - hedges * 1.0, 1.0, 10.0)), 2
        )

        wv_words = key.get("work_value_words", [])
        wv_hits = sum(1 for w in wv_words if w in answer_lower)
        wv_score = round(min(4.0 + wv_hits * 1.2, 10.0), 2)

        overall = round(
            np.mean([keywords_score, tech_rel, comm_score, conf_score, wv_score]),
            2,
        )

        if overall >= 8.0:
            feedback = "Excellent answer - thorough and well-structured."
        elif overall >= 7.0:
            feedback = "Very good. Strong understanding shown."
        elif overall >= 6.0:
            feedback = "Good answer. Add more examples and technical depth."
        elif overall >= 4.5:
            feedback = "Adequate. Cover more concepts and structure better."
        elif overall >= 3.0:
            feedback = "Partial. Key ideas missing - elaborate more."
        else:
            feedback = "Too brief. Speak more and cover core concepts."

        tips = []
        if keywords_score < 5.0:
            tips.append("use technical terms")
        if not has_example:
            tips.append("add a real example")
        if word_count < 30:
            tips.append(f"speak more (only {word_count} words)")
        if connectors < 2:
            tips.append("use First / Then / Finally structure")
        if tips:
            feedback += " Tip: " + ", ".join(tips) + "."

        return {
            "overall":             overall,
            "keyword_hits":        group_hits,
            "question":            question_text,
            "feedback":            feedback,
            "communication":       comm_score,
            "confidence":          conf_score,
            "keywords_matched":    keywords_score,
            "work_values":         wv_score,
            "technical_relevancy": tech_rel,
        }

    def _score_one(self, qid, answer, question_text=""):
        ai = self._score_with_groq(question_text, answer)
        if ai:
            ai["question"] = question_text
            return ai
        return self._score_keywords(qid, answer, question_text)

    def score_all(self, answers):
        per_q = {}
        comm_list = []
        conf_list = []
        kw_list   = []
        wv_list   = []
        tr_list   = []
        weights   = []

        for a in answers:
            qid  = int(a.get("question_id", 1))
            txt  = a.get("transcript", "")
            qtxt = a.get("question", "")
            res  = self._score_one(qid, txt, qtxt)
            w    = ANSWER_KEYS.get(qid, {}).get("weight", 1 / 6)

            per_q[str(qid)] = res
            comm_list.append(res["communication"]       * w)
            conf_list.append(res["confidence"]          * w)
            kw_list.append(  res["keywords_matched"]    * w)
            wv_list.append(  res["work_values"]         * w)
            tr_list.append(  res["technical_relevancy"] * w)
            weights.append(w)

        total_w = sum(weights)

        def wavg(lst):
            return round(sum(lst) / max(total_w, 1e-8), 2)

        communication       = wavg(comm_list)
        confidence_level    = wavg(conf_list)
        keywords_matched    = wavg(kw_list)
        work_values         = wavg(wv_list)
        technical_relevancy = wavg(tr_list)

        overall_technical = round(
            (keywords_matched + technical_relevancy) / 2, 2
        )
        overall_soft = round(
            (communication + confidence_level + work_values) / 3, 2
        )
        overall_score = round(
            0.60 * overall_technical + 0.40 * overall_soft, 2
        )

        decision, icon = get_decision(overall_score)

        suggestions = []
        checks = {
            "Communication Skills":  (communication,       "Use First/Then/Finally. Reduce um/uh."),
            "Confidence Level":      (confidence_level,    "Say 'I built', 'I implemented' not 'I think maybe'."),
            "Keywords Matched":      (keywords_matched,    "Include terms: LIFO, FIFO, hash, cache, O(n log n)."),
            "Work Values":           (work_values,         "Show teamwork, ownership and learning mindset."),
            "Technical Relevancy":   (technical_relevancy, "Give specific technical details and examples."),
        }
        for dim, (score, tip) in checks.items():
            if score < 6.5:
                suggestions.append({"area": dim, "score": score, "tip": tip})

        return {
            "per_question": per_q,
            "scores": {
                "communication_skill":  communication,
                "confidence_level":     confidence_level,
                "keywords_matched":     keywords_matched,
                "work_values":          work_values,
                "technical_relevancy":  technical_relevancy,
                "overall_technical":    overall_technical,
                "overall_soft":         overall_soft,
                "overall_score":        overall_score,
                "eye_contact":          5.0,
                "pitch_tone":           5.0,
                "expression":           5.0,
            },
            "overall_score":  overall_score,
            "decision":       decision,
            "decision_icon":  icon,
            "suggestions":    suggestions,
        }