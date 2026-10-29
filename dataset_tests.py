import io, json, tarfile, random
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

random.seed(0)
np.random.seed(0)

USE_EMBEDDINGS = True      # set False to skip the sentence-transformers comparison
N_REFS = 2000              # reference examples per dataset
TOP_K = 5                  # score = mean of the top-K similarities
vader = SentimentIntensityAnalyzer()
_embedder = None


# ---------- helpers ----------
def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def similarity(refs, tests, method="tfidf", k=TOP_K):
    """Mean of the top-k cosine similarities between each test text and the reference set."""
    if method == "tfidf":
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        R = vec.fit_transform(refs)
        T = vec.transform(tests)
    else:
        m = get_embedder()
        R = m.encode(refs, normalize_embeddings=True)
        T = m.encode(tests, normalize_embeddings=True)
    sims = cosine_similarity(T, R)
    return np.sort(sims, axis=1)[:, -k:].mean(axis=1)


def run_similarity_test(name, refs, tests):
    """tests = list of (label, text). Prints TF-IDF (and embedding) scores."""
    texts = [t for _, t in tests]
    methods = ["tfidf"] + (["embed"] if USE_EMBEDDINGS else [])
    results = {m: similarity(refs, texts, m) for m in methods}
    print(f"\n--- Cosine similarity test: {name} (refs: {len(refs)}) ---")
    for i, (label, text) in enumerate(tests):
        scores = "  ".join(f"{m}={results[m][i]:.3f}" for m in methods)
        print(f"[{label:>4}] {scores}  | {text}")


def sentiment_label(text):
    c = vader.polarity_scores(text)["compound"]
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"


# ---------- test sentences (edit these, or use ones from the nursing team) ----------
EMPATHY_TESTS = [
    ("good", "I can see this is really worrying you. Tell me more about how you're feeling."),
    ("good", "That sounds really hard, and I'm here with you."),
    ("good", "I'm concerned about this."),   # your teammate's example
    ("bad",  "Stop worrying, it's nothing."),
    ("bad",  "Next question. What is your date of birth?"),
    ("bad",  "I don't have time for this."),
]
RELATIONSHIP_TESTS = [
    ("good", "What concerns do you have about this? I'd like to hear them."),
    ("good", "Can you tell me more about what's been going on for your family?"),
    ("bad",  "You will take the medication and that is final."),
    ("bad",  "The doctor will see you later."),
]
EDUCATION_TESTS = [
    ("good", "This medicine helps lower your blood pressure, so please take it every morning even if you feel fine."),
    ("good", "Drinking more water can help ease headaches caused by dehydration."),
    ("bad",  "Just take it."),
    ("bad",  "I don't know, ask someone else."),
]

# Simple polarity map for EmpatheticDialogues emotion labels (ambiguous ones left out)
POS = {"excited", "proud", "grateful", "impressed", "hopeful", "confident", "joyful",
       "content", "caring", "trusting", "faithful"}
NEG = {"angry", "sad", "annoyed", "lonely", "afraid", "terrified", "guilty", "disgusted",
       "furious", "anxious", "disappointed", "jealous", "devastated", "embarrassed",
       "ashamed", "apprehensive"}


# ---------- 1. EmpatheticDialogues ----------
def test_empathetic():
    url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    raw = requests.get(url, timeout=180).content
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        df = pd.read_csv(tar.extractfile("empatheticdialogues/train.csv"), on_bad_lines="skip")
    df["utterance"] = df["utterance"].astype(str).str.replace("_comma_", ",", regex=False)
    df["utterance_idx"] = pd.to_numeric(df["utterance_idx"], errors="coerce")

    first = df[df["utterance_idx"] == 1].copy()
    first["polarity"] = first["context"].map(lambda e: "positive" if e in POS else "negative" if e in NEG else None)
    sample = first.dropna(subset=["polarity"]).sample(20, random_state=0)
    sample["vader"] = sample["utterance"].map(sentiment_label)
    agree = (sample["vader"] == sample["polarity"]).mean()
    print("\n--- Sentiment test: EmpatheticDialogues (20 samples) ---")
    for _, r in sample.iterrows():
        print(f"label={r['context']:<13} expected={r['polarity']:<8} vader={r['vader']:<8} | {r['utterance'][:90]}")
    print(f"Agreement with emotion polarity: {agree:.0%}")

    # listener replies = even-numbered turns in each conversation
    listener = df[df["utterance_idx"] % 2 == 0]["utterance"]
    print(f"\nListener replies available: {len(listener)}")
    refs = listener.sample(min(N_REFS, len(listener)), random_state=0).tolist()
    run_similarity_test("EmpatheticDialogues (listener replies)", refs, EMPATHY_TESTS)

    print("\n--- VADER on the test sentences (shows why sentiment alone is risky) ---")
    for label, t in EMPATHY_TESTS:
        print(f"[{label:>4}] compound={vader.polarity_scores(t)['compound']:+.2f} | {t}")


def test_esconv():
    url = "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json"
    data = requests.get(url, timeout=120).json()
    print("ESConv sample turn (shows the real key names):", data[0]["dialog"][1])
    question_turns, affirm_turns = [], []
    for conv in data:
        for turn in conv["dialog"]:
            ann = turn.get("annotation") or {}
            strat = str(ann.get("strategy") or turn.get("strategy") or "").lower()
            text = turn.get("content") or turn.get("text") or ""
            if not text:
                continue
            if strat.startswith("question"):
                question_turns.append(text)
            elif strat.startswith("affirmation"):
                affirm_turns.append(text)
    print(f"ESConv: {len(question_turns)} question turns, {len(affirm_turns)} affirmation turns")
    random.shuffle(question_turns); random.shuffle(affirm_turns)
    run_similarity_test("ESConv 'Question' strategy turns", question_turns[:N_REFS], RELATIONSHIP_TESTS)
    run_similarity_test("ESConv 'Affirmation and Reassurance' turns", affirm_turns[:N_REFS], EMPATHY_TESTS)


def test_meddialog():
    from datasets import load_dataset
    try:
        ds = load_dataset("OpenMed/MedDialog", split="train[:3000]")
        rows = [r["doctor_response"] for r in ds]
    except Exception as e:
        print("Normal load failed, trying streaming:", e)
        ds = load_dataset("OpenMed/MedDialog", split="train", streaming=True)
        rows = [r["doctor_response"] for _, r in zip(range(3000), ds)]
    run_similarity_test("MedDialog (doctor responses)", rows[:N_REFS], EDUCATION_TESTS)


def test_dailydialog():
    import zipfile
    url = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip"
    raw = requests.get(url, timeout=120).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        print("Files in zip:", z.namelist())
        name = [n for n in z.namelist() if n.endswith("dialogues.json")][0]
        dialogues = json.load(z.open(name))

    utts = []   # (emotion, intent, text)
    for d in dialogues:
        for t in d["turns"]:
            acts = t.get("dialogue_acts", {}).get("binary", [])
            intent = acts[0]["intent"] if acts else ""
            utts.append((t["emotion"], intent, t["utterance"]))
    print(f"DailyDialog: {len(dialogues)} dialogues, {len(utts)} utterances")

    # Sentiment test: 10 positive, 10 negative, 10 'no emotion' utterances
    pools = {
        "positive": [u for e, i, u in utts if e == "happiness"],
        "negative": [u for e, i, u in utts if e in ("anger", "disgust", "fear", "sadness")],
        "neutral":  [u for e, i, u in utts if e == "no emotion"],
    }
    rng = random.Random(0)
    print("\n--- Sentiment test: DailyDialog (30 samples) ---")
    correct = 0
    for expected, pool in pools.items():
        hits = 0
        for u in rng.sample(pool, 10):
            pred = sentiment_label(u)
            hits += pred == expected
            print(f"expected={expected:<8} vader={pred:<8} | {u[:90]}")
        print(f"  -> {expected}: {hits}/10 matched\n")
        correct += hits
    print(f"Agreement with emotion label: {correct}/30 ({correct/30:.0%})")

    # Cosine similarity: general questions as references for relationship-building tests
    questions = [u for e, i, u in utts if i == "question"]
    refs = rng.sample(questions, min(N_REFS, len(questions)))
    run_similarity_test("DailyDialog question turns", refs, RELATIONSHIP_TESTS)
    # General chat as references for empathy tests (a baseline to compare against ESConv and EmpatheticDialogues)
    general = [u for e, i, u in utts]
    refs = rng.sample(general, min(N_REFS, len(general)))
    run_similarity_test("DailyDialog (all utterances)", refs, EMPATHY_TESTS)

def compare_sentiment(n=100):
    import zipfile
    # EmpatheticDialogues: first utterance of each conversation
    url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    raw = requests.get(url, timeout=180).content
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        df = pd.read_csv(tar.extractfile("empatheticdialogues/train.csv"), on_bad_lines="skip")
    df["utterance"] = df["utterance"].astype(str).str.replace("_comma_", ",", regex=False)
    df["utterance_idx"] = pd.to_numeric(df["utterance_idx"], errors="coerce")
    first = df[df["utterance_idx"] == 1].copy()
    first["polarity"] = first["context"].map(lambda e: "positive" if e in POS else "negative" if e in NEG else None)
    first = first.dropna(subset=["polarity"])
    ed = {p: first[first["polarity"] == p]["utterance"].sample(n, random_state=0).tolist()
          for p in ("positive", "negative")}

    # DailyDialog: every utterance has its own emotion label
    url = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip"
    raw = requests.get(url, timeout=120).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [x for x in z.namelist() if x.endswith("dialogues.json")][0]
        dialogues = json.load(z.open(name))
    utts = [(t["emotion"], t["utterance"]) for d in dialogues for t in d["turns"]]
    rng = random.Random(0)
    dd = {
        "positive": rng.sample([u for e, u in utts if e == "happiness"], n),
        "negative": rng.sample([u for e, u in utts if e in ("anger", "disgust", "fear", "sadness")], n),
    }

    print(f"\n--- Sentiment comparison, VADER, {n} positive + {n} negative per dataset ---")
    for name, pools in (("EmpatheticDialogues", ed), ("DailyDialog", dd)):
        total_hits = 0
        for expected, texts in pools.items():
            preds = [sentiment_label(t) for t in texts]
            hits = sum(p == expected for p in preds)
            neutral = sum(p == "neutral" for p in preds)
            total_hits += hits
            print(f"{name:<20} {expected:<9} matched {hits}/{n}  (VADER said neutral on {neutral})")
        acc = total_hits / (2 * n)
        margin = 1.96 * (acc * (1 - acc) / (2 * n)) ** 0.5
        print(f"{name:<20} OVERALL {acc:.0%}  (+/- {margin:.0%} at 95% confidence)\n")


# ---------- run everything ----------
if __name__ == "__main__":
    for fn in (compare_sentiment, test_empathetic, test_esconv, test_dailydialog, test_meddialog):
        try:
            fn()
        except Exception as e:
            print(f"\n!! {fn.__name__} failed: {type(e).__name__}: {e}")