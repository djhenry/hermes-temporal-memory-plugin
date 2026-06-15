"""Mood sentiment lexicon and emotion model for the temporal-memory plugin.

Plutchik's Wheel of Emotions — 8 primary emotion axes with keyword-based
sentiment analysis. No external dependencies.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Emotion model
# ---------------------------------------------------------------------------

EMOTION_KEYS = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

# Opposite pairs per Plutchik
OPPOSITES = {
    "joy": "sadness", "sadness": "joy",
    "trust": "disgust", "disgust": "trust",
    "fear": "anger", "anger": "fear",
    "surprise": "anticipation", "anticipation": "surprise",
}

# Primary dyads (adjacent emotion combinations)
_DYADS = [
    (("joy", "trust"), "love"),
    (("trust", "fear"), "submission"),
    (("fear", "surprise"), "awe"),
    (("surprise", "sadness"), "disapproval"),
    (("sadness", "disgust"), "remorse"),
    (("disgust", "anger"), "contempt"),
    (("anger", "anticipation"), "aggressiveness"),
    (("anticipation", "joy"), "optimism"),
]

# Outer dyads (more complex mixed feelings)
_OUTER_DYADS = [
    (("joy", "fear"), "guilt"),
    (("trust", "sadness"), "sentimentality"),
    (("fear", "disgust"), "shame"),
    (("surprise", "anger"), "outrage"),
    (("sadness", "anger"), "envy"),
    (("disgust", "anticipation"), "cynicism"),
    (("anger", "joy"), "pride"),
    (("anticipation", "trust"), "hope"),
]


# ---------------------------------------------------------------------------
# Keyword sentiment lexicon
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "happy": {"joy": 0.15, "trust": 0.05},
    "glad": {"joy": 0.12, "trust": 0.03},
    "great": {"joy": 0.10, "anticipation": 0.05},
    "awesome": {"joy": 0.15, "surprise": 0.05},
    "excellent": {"joy": 0.12, "trust": 0.05},
    "wonderful": {"joy": 0.15, "trust": 0.05},
    "love": {"joy": 0.15, "trust": 0.10},
    "thanks": {"joy": 0.08, "trust": 0.08},
    "thank": {"joy": 0.08, "trust": 0.08},
    "appreciate": {"joy": 0.08, "trust": 0.10},
    "good": {"joy": 0.08, "trust": 0.03},
    "nice": {"joy": 0.08, "trust": 0.03},
    "cool": {"joy": 0.06, "surprise": 0.04},
    "fun": {"joy": 0.12, "anticipation": 0.05},
    "excited": {"joy": 0.10, "anticipation": 0.15},
    "exciting": {"joy": 0.10, "anticipation": 0.15},
    "interesting": {"anticipation": 0.10, "surprise": 0.05},
    "curious": {"anticipation": 0.12, "surprise": 0.05},
    "helpful": {"trust": 0.12, "joy": 0.05},
    "helped": {"trust": 0.10, "joy": 0.05},
    "perfect": {"joy": 0.10, "trust": 0.08},
    "beautiful": {"joy": 0.12, "surprise": 0.05},
    "amazing": {"joy": 0.15, "surprise": 0.10},
    "brilliant": {"joy": 0.10, "surprise": 0.08, "trust": 0.05},
    "fantastic": {"joy": 0.15, "surprise": 0.05},
    "yay": {"joy": 0.20, "surprise": 0.05},
    "wow": {"surprise": 0.15, "joy": 0.05},
    "progress": {"anticipation": 0.10, "joy": 0.05},
    "solved": {"joy": 0.10, "trust": 0.05},
    "working": {"anticipation": 0.05, "trust": 0.05},
    "success": {"joy": 0.15, "trust": 0.05},
    "successful": {"joy": 0.15, "trust": 0.05},
    "win": {"joy": 0.15, "anticipation": 0.05},
    "won": {"joy": 0.15, "surprise": 0.05},
    "celebrate": {"joy": 0.20, "anticipation": 0.05},
    "celebration": {"joy": 0.20, "anticipation": 0.05},
    "friend": {"trust": 0.10, "joy": 0.05},
    "friendly": {"trust": 0.10, "joy": 0.05},
    "kind": {"trust": 0.12, "joy": 0.05},
    "gentle": {"trust": 0.08, "joy": 0.03},
    "safe": {"trust": 0.10, "fear": -0.05},
    "secure": {"trust": 0.10, "fear": -0.05},
    "confident": {"trust": 0.10, "anticipation": 0.05},
    "hope": {"anticipation": 0.15, "joy": 0.05},
    "hopeful": {"anticipation": 0.15, "joy": 0.05},
    "optimistic": {"anticipation": 0.15, "joy": 0.08},
    "better": {"joy": 0.08, "anticipation": 0.05},
    "improve": {"anticipation": 0.10, "joy": 0.05},
    "improving": {"anticipation": 0.10, "joy": 0.05},
    "learn": {"anticipation": 0.10, "trust": 0.03},
    "learning": {"anticipation": 0.10, "trust": 0.03},
    "discover": {"surprise": 0.10, "anticipation": 0.08},
    "discovered": {"surprise": 0.10, "joy": 0.05},
    "new": {"surprise": 0.05, "anticipation": 0.08},
    "creative": {"joy": 0.08, "anticipation": 0.08},
    "create": {"anticipation": 0.10, "joy": 0.05},
    "built": {"joy": 0.08, "trust": 0.05},
    "build": {"anticipation": 0.10, "joy": 0.03},
}

_NEGATIVE_WORDS = {
    "sad": {"sadness": 0.15, "joy": -0.05},
    "sorry": {"sadness": 0.10, "trust": 0.03},
    "unfortunately": {"sadness": 0.08, "anticipation": -0.05},
    "unhappy": {"sadness": 0.15, "joy": -0.10},
    "angry": {"anger": 0.20, "trust": -0.05},
    "anger": {"anger": 0.15, "trust": -0.05},
    "annoyed": {"anger": 0.12, "disgust": 0.05},
    "annoying": {"anger": 0.12, "disgust": 0.05},
    "frustrated": {"anger": 0.15, "sadness": 0.05},
    "frustrating": {"anger": 0.15, "sadness": 0.05},
    "frustration": {"anger": 0.15, "sadness": 0.05},
    "hate": {"anger": 0.15, "disgust": 0.10},
    "disgusting": {"disgust": 0.20, "anger": 0.05},
    "terrible": {"disgust": 0.10, "sadness": 0.10, "anger": 0.05},
    "horrible": {"disgust": 0.10, "fear": 0.05, "sadness": 0.05},
    "awful": {"disgust": 0.10, "sadness": 0.08},
    "bad": {"sadness": 0.08, "disgust": 0.05},
    "worst": {"sadness": 0.15, "anger": 0.05},
    "worse": {"sadness": 0.10, "fear": 0.05},
    "fear": {"fear": 0.15, "anticipation": -0.05},
    "afraid": {"fear": 0.15, "sadness": 0.03},
    "scared": {"fear": 0.15, "surprise": 0.05},
    "scary": {"fear": 0.12, "surprise": 0.05},
    "worried": {"fear": 0.10, "sadness": 0.05, "anticipation": 0.05},
    "worry": {"fear": 0.10, "sadness": 0.05},
    "anxious": {"fear": 0.12, "anticipation": 0.05},
    "anxiety": {"fear": 0.12, "anticipation": 0.05},
    "nervous": {"fear": 0.10, "anticipation": 0.05},
    "stressed": {"fear": 0.10, "anger": 0.05},
    "stress": {"fear": 0.08, "anger": 0.05},
    "overwhelmed": {"fear": 0.10, "sadness": 0.08},
    "confused": {"surprise": 0.08, "fear": 0.03},
    "confusing": {"surprise": 0.08, "fear": 0.03},
    "broken": {"sadness": 0.10, "anger": 0.05},
    "bug": {"anger": 0.05, "sadness": 0.03},
    "bugs": {"anger": 0.08, "sadness": 0.05},
    "error": {"anger": 0.05, "surprise": 0.03},
    "errors": {"anger": 0.08, "surprise": 0.03},
    "fail": {"sadness": 0.10, "anger": 0.05},
    "failed": {"sadness": 0.12, "anger": 0.08},
    "failure": {"sadness": 0.15, "anger": 0.05},
    "wrong": {"anger": 0.05, "sadness": 0.05},
    "problem": {"fear": 0.05, "sadness": 0.05},
    "problems": {"fear": 0.08, "sadness": 0.05},
    "issue": {"fear": 0.03, "sadness": 0.03},
    "issues": {"fear": 0.05, "sadness": 0.03},
    "stuck": {"sadness": 0.08, "anger": 0.05},
    "impossible": {"sadness": 0.10, "anger": 0.05},
    "never": {"sadness": 0.08, "anger": 0.03},
    "nothing": {"sadness": 0.05},
    "useless": {"disgust": 0.10, "sadness": 0.08},
    "waste": {"disgust": 0.08, "sadness": 0.05},
    "wasted": {"disgust": 0.08, "sadness": 0.08},
    "slow": {"sadness": 0.05, "anger": 0.03},
    "pain": {"sadness": 0.12, "fear": 0.05},
    "painful": {"sadness": 0.12, "fear": 0.05},
    "hurt": {"sadness": 0.15, "anger": 0.05},
    "damage": {"anger": 0.08, "sadness": 0.05},
    "damaged": {"anger": 0.08, "sadness": 0.05},
    "crash": {"surprise": 0.10, "anger": 0.08, "fear": 0.05},
    "crashed": {"surprise": 0.10, "anger": 0.10, "fear": 0.05},
    "dead": {"sadness": 0.15, "fear": 0.05},
    "die": {"fear": 0.10, "sadness": 0.10},
    "died": {"sadness": 0.15, "fear": 0.05},
    "kill": {"anger": 0.15, "fear": 0.10},
    "stupid": {"disgust": 0.15, "anger": 0.08},
    "dumb": {"disgust": 0.12, "anger": 0.05},
    "idiot": {"disgust": 0.15, "anger": 0.10},
    "pathetic": {"disgust": 0.15, "sadness": 0.05},
    "ridiculous": {"disgust": 0.10, "anger": 0.05},
    "absurd": {"disgust": 0.08, "surprise": 0.05},
    "boring": {"sadness": 0.08, "disgust": 0.05},
    "bored": {"sadness": 0.10, "disgust": 0.03},
    "tired": {"sadness": 0.08},
    "exhausted": {"sadness": 0.12, "fear": 0.03},
    "lonely": {"sadness": 0.15, "fear": 0.03},
    "alone": {"sadness": 0.10},
    "lost": {"sadness": 0.10, "fear": 0.05},
    "missing": {"sadness": 0.08, "fear": 0.03},
    "gone": {"sadness": 0.10},
    "leave": {"sadness": 0.08, "fear": 0.03},
    "leaving": {"sadness": 0.08, "fear": 0.05},
    "stop": {"anger": 0.05, "sadness": 0.03},
    "quit": {"sadness": 0.08, "anger": 0.05},
    "give up": {"sadness": 0.15, "anger": 0.05},
    "giving up": {"sadness": 0.15, "anger": 0.05},
}


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

def _analyze_sentiment(text: str) -> dict[str, float]:
    """Return emotion deltas based on keyword matching.

    Scans text for known positive/negative words and accumulates
    per-emotion deltas. Case-insensitive. Multi-word phrases checked first.
    """
    deltas: dict[str, float] = {k: 0.0 for k in EMOTION_KEYS}
    text_lower = text.lower()

    # Check multi-word phrases first
    for phrase, effects in _NEGATIVE_WORDS.items():
        if " " in phrase and phrase in text_lower:
            for emotion, delta in effects.items():
                deltas[emotion] += delta

    for phrase, effects in _POSITIVE_WORDS.items():
        if " " in phrase and phrase in text_lower:
            for emotion, delta in effects.items():
                deltas[emotion] += delta

    # Then single words
    words = set(text_lower.split())
    for word in words:
        clean = word.strip(".,!?;:\"'()[]{}")
        if clean in _POSITIVE_WORDS and " " not in clean:
            for emotion, delta in _POSITIVE_WORDS[clean].items():
                deltas[emotion] += delta
        if clean in _NEGATIVE_WORDS and " " not in clean:
            for emotion, delta in _NEGATIVE_WORDS[clean].items():
                deltas[emotion] += delta

    return deltas


# ---------------------------------------------------------------------------
# Mood label resolution
# ---------------------------------------------------------------------------

def _resolve_mood_label(emotions: dict[str, float], threshold: float) -> str:
    """Determine the dominant mood label from emotion intensities."""
    # Find dominant single emotion
    max_emotion = max(emotions, key=lambda k: emotions[k])
    max_val = emotions[max_emotion]

    if max_val >= threshold:
        return max_emotion

    # Check primary dyads
    for (e1, e2), label in _DYADS:
        avg = (emotions[e1] + emotions[e2]) / 2
        if avg >= threshold:
            return label

    # Check outer dyads
    for (e1, e2), label in _OUTER_DYADS:
        avg = (emotions[e1] + emotions[e2]) / 2
        if avg >= threshold * 0.85:
            return label

    # Fallback: compute overall valence
    positive = emotions["joy"] + emotions["trust"] + emotions["anticipation"]
    negative = emotions["sadness"] + emotions["anger"] + emotions["fear"] + emotions["disgust"]
    if positive > negative + 0.2:
        return "content"
    elif negative > positive + 0.2:
        return "subdued"
    else:
        return "calm"


def _emotion_summary(emotions: dict[str, float], label: str) -> str:
    """Generate a brief natural-language summary of the emotional state."""
    sorted_e = sorted(emotions.items(), key=lambda x: -x[1])
    top3 = sorted_e[:3]

    parts = []
    for name, val in top3:
        if val > 0.7:
            parts.append(f"very {name}")
        elif val > 0.5:
            parts.append(f"quite {name}")
        elif val > 0.3:
            parts.append(f"mildly {name}")
        elif val > 0.15:
            parts.append(f"slightly {name}")

    if not parts:
        return "feeling neutral and balanced"
    if len(parts) == 1:
        return f"feeling {parts[0]}"
    elif len(parts) == 2:
        return f"feeling {parts[0]} and {parts[1]}"
    else:
        return f"feeling {parts[0]}, {parts[1]}, and {parts[2]}"


def _default_emotions(
    baseline_joy: float = 0.4,
    baseline_trust: float = 0.6,
    baseline_fear: float = 0.1,
    baseline_surprise: float = 0.3,
    baseline_sadness: float = 0.1,
    baseline_disgust: float = 0.05,
    baseline_anger: float = 0.05,
    baseline_anticipation: float = 0.5,
) -> dict[str, float]:
    """Return baseline emotion dict from config values."""
    return {
        "joy": baseline_joy,
        "trust": baseline_trust,
        "fear": baseline_fear,
        "surprise": baseline_surprise,
        "sadness": baseline_sadness,
        "disgust": baseline_disgust,
        "anger": baseline_anger,
        "anticipation": baseline_anticipation,
    }
