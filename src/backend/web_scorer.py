"""
web_scorer.py - Fixed scoring
Fixes:
  - All scores now give real non-zero values
  - Technical relevancy, confidence etc properly calculated
  - Questions displayed in report
"""

import numpy as np
import re

ANSWER_KEYS = {
    1: {
        "keyword_groups": [
            ["experience","worked","background","years","built","developed"],
            ["python","java","javascript","react","sql","language","framework","coding","c++","node"],
            ["project","developed","created","designed","built","made","implemented"],
            ["team","collaborate","agile","scrum","together","group","worked with"],
            ["backend","frontend","fullstack","database","cloud","web","app","software"],
        ],
        "weight": 0.10,
        "work_value_words": ["team","learn","grow","contribute","passion","problem","improve","build"],
    },
    2: {
        "keyword_groups": [
            ["stack","stacks"],
            ["queue","queues","waiting"],
            ["lifo","last in","last-in","top","pile","last added","last element"],
            ["fifo","first in","first-in","line","order","first added","first element"],
            ["recursion","bfs","scheduling","undo","call stack","browser","task","print queue"],
        ],
        "weight": 0.20,
        "work_value_words": ["understand","explain","example","use case","real","difference","data structure"],
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
        "work_value_words": ["scale","efficient","design","architecture","trade-off","solution","system"],
    },
    4: {
        "keyword_groups": [
            ["time complexity","big o","o(n","complexity","efficient","performance","notation"],
            ["n log n","nlogn","merge sort","heap sort","quick sort","sorting","log"],
            ["linear","quadratic","constant","logarithmic","polynomial","exponential"],
            ["algorithm","sort","search","optimize","loop","iteration","recursion"],
            ["best case","worst case","average","space complexity","operations","steps"],
        ],
        "weight": 0.20,
        "work_value_words": ["analyze","optimize","efficiency","understand","improve","measure","calculate"],
    },
    5: {
        "keyword_groups": [
            ["problem","challenge","issue","difficult","bug","error","obstacle"],
            ["debug","investigate","diagnose","identify","found","noticed","analyzed"],
            ["solution","fixed","resolved","solved","implemented","changed","refactored"],
            ["team","collaborate","together","help","support","colleague","mentor"],
            ["result","outcome","improved","learned","impact","better","success"],
        ],
        "weight": 0.15,
        "work_value_words": ["team","ownership","responsibility","initiative","learned","grew","improved"],
    },
    6: {
        "keyword_groups": [
            ["senior","lead","architect","grow","advancement","promotion","leadership"],
            ["learn","skill","improve","develop","knowledge","master","expertise"],
            ["contribute","impact","build","create","ownership","meaningful","make"],
            ["goal","vision","career","future","plan","aspire","aim"],
            ["role","company","team","opportunity","fit","align","position"],
        ],
        "weight": 0.15,
        "work_value_words": ["growth","impact","learn","contribute","long-term","vision","passionate","motivated"],
    },
}

DECISION_LABELS = [
    (9.0, "Exceptional Hire", "⭐"),
    (8.0, "Strong Hire",      "✅"),
    (7.0, "Hire",             "✅"),
    (5.5, "Borderline",       "⚠️"),
    (0.0, "No Hire",          "❌"),
]


def get_decision(score):
    for t, l, i in DECISION_LABELS:
        if score >= t:
            return l, i
    return "No Hire", "❌"


class WebScorer:
    def __init__(self):
        print("[Scorer] Ready.")

    def _score_one(self, qid: int, answer: str, question_text: str = "") -> dict:
        # If answer is empty, return 0
        if not answer or not answer.strip() or len(answer.strip()) < 3:
            return {
                "overall": 0.0, "keyword_hits": 0, "question": question_text,
                "feedback": "No answer provided — speak clearly into the microphone.",
                "communication": 0.0, "confidence": 0.0,
                "keywords_matched": 0.0, "work_values": 0.0, "technical_relevancy": 0.0,
            }

        key          = ANSWER_KEYS.get(qid, ANSWER_KEYS[1])
        answer_lower = answer.lower()
        words        = answer.split()
        word_count   = len(words)

        # ── 1. Keywords Matched ───────────────────────────────────────
        groups     = key["keyword_groups"]
        group_hits = sum(1 for g in groups if any(kw in answer_lower for kw in g))
        # Base score: each group hit = 2 points, baseline of 3 for any answer
        keywords_score = round(min(3.0 + group_hits * 1.4, 10.0), 2)

        # ── 2. Technical Relevancy ────────────────────────────────────
        # Based on keyword density + word count
        density   = group_hits / max(len(groups), 1)
        length_bonus = min(word_count / 30.0, 1.5)
        tech_rel  = round(min(2.0 + density * 6.0 + length_bonus, 10.0), 2)

        # ── 3. Communication Skills ───────────────────────────────────
        fillers    = sum(answer_lower.count(w) for w in
                        ["um ", " uh ", " like ", "you know", "basically", "literally", "sort of"])
        connectors = sum(1 for w in
                        ["first", "then", "finally", "also", "because", "therefore",
                         "so ", "next", "additionally", "furthermore", "however", "moreover"]
                        if w in answer_lower)
        has_example = any(p in answer_lower for p in
                         ["for example", "for instance", "such as", "like when",
                          "in my project", "i built", "we used", "i worked on",
                          "imagine", "consider", "think of", "when i"])

        filler_pen  = max(2.0, 9.0 - fillers * 1.5)
        struct_sc   = min(2.0 + connectors * 1.5, 9.0)
        length_sc   = min(2.0 + word_count / 20.0, 9.5)
        example_sc  = 8.5 if has_example else 5.5
        comm_score  = round((filler_pen + struct_sc + length_sc + example_sc) / 4.0, 2)

        # ── 4. Confidence Level ───────────────────────────────────────
        hedges     = sum(answer_lower.count(w) for w in
                        ["maybe ", "i think ", "i guess ", "not sure", "perhaps", "kind of", "i dont know"])
        assertive  = sum(1 for w in
                        ["definitely", "clearly", "specifically", "exactly",
                         "i know", "i have", "i built", "i designed",
                         "i led", "i implemented", "i created", "i solved",
                         "i worked", "i developed", "i used"]
                        if w in answer_lower)
        # Base confidence 5.5, goes up with assertive language
        conf_score = round(float(np.clip(5.5 + assertive * 0.7 - hedges * 1.0, 1.0, 10.0)), 2)

        # ── 5. Work Values ────────────────────────────────────────────
        wv_words   = key.get("work_value_words", [])
        wv_hits    = sum(1 for w in wv_words if w in answer_lower)
        # Base score 4.0 for any answer
        wv_score   = round(min(4.0 + wv_hits * 1.2, 10.0), 2)

        # ── Overall ───────────────────────────────────────────────────
        overall = round(np.mean([
            keywords_score, tech_rel, comm_score, conf_score, wv_score
        ]), 2)

        # ── Feedback ──────────────────────────────────────────────────
        if overall >= 8.0:
            feedback = "Excellent — thorough, accurate and well-structured."
        elif overall >= 7.0:
            feedback = "Very good. Strong understanding shown."
        elif overall >= 6.0:
            feedback = "Good answer. Add more technical examples and depth."
        elif overall >= 4.5:
            feedback = "Adequate. Cover more concepts and use structured language."
        elif overall >= 3.0:
            feedback = "Partial. Key ideas present but needs more elaboration."
        else:
            feedback = "Too brief. Speak more and cover core concepts."

        tips = []
        if keywords_score < 5.0:  tips.append("use technical terms like LIFO, FIFO, hash, cache")
        if not has_example:        tips.append("add a real example (say 'for example...')")
        if word_count < 30:        tips.append(f"speak more — only {word_count} words detected")
        if connectors < 2:         tips.append("structure with: First / Then / Finally")
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

    def score_all(self, answers: list) -> dict:
        per_q       = {}
        comm_list   = []
        conf_list   = []
        kw_list     = []
        wv_list     = []
        tr_list     = []
        weights     = []

        for a in answers:
            qid  = int(a.get("question_id", 1))
            txt  = a.get("transcript", "")
            qtxt = a.get("question", "")
            res  = self._score_one(qid, txt, qtxt)
            w    = ANSWER_KEYS.get(qid, {}).get("weight", 1/6)

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

        overall_technical = round((keywords_matched + technical_relevancy) / 2, 2)
        overall_soft      = round((communication + confidence_level + work_values) / 3, 2)
        overall_score     = round(0.60 * overall_technical + 0.40 * overall_soft, 2)

        decision, icon = get_decision(overall_score)

        suggestions = []
        checks = {
            "Communication Skills":  (communication,       "Use First/Then/Finally to structure your answers. Reduce um/uh."),
            "Confidence Level":      (confidence_level,    "Say 'I built', 'I implemented' instead of 'I think maybe'."),
            "Keywords Matched":      (keywords_matched,    "Include technical terms: LIFO, FIFO, hash, cache, O(n log n)."),
            "Work Values":           (work_values,         "Show teamwork, ownership and learning attitude in your answers."),
            "Technical Relevancy":   (technical_relevancy, "Stay on topic, give specific technical details and examples."),
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