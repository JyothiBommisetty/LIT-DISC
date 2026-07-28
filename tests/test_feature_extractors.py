"""Test all feature extractors produce sensible outputs."""
import sys
sys.path.insert(0, "D:\\LIT-DISC")
import numpy as np
from src.features import *


def run_test(name, extractor, sentence, tokens, pos, expected_range):
    f, g = extractor.extract(sentence, tokens, pos, [])
    in_range = all(lo <= v <= hi for v, (lo, hi) in zip(f, expected_range))
    print(f"[{'PASS' if in_range else 'FAIL'}] {name}: "
          f"k_d={len(f)}, gamma={g:.3f}, "
          f"features={[round(float(v),3) for v in f]}")
    return in_range


def test_narrative():
    e = NarrativeExtractor()
    s1 = "He walked slowly down the dark corridor, listening for any sound."
    toks1 = ["He","walked","slowly","down","the","dark","corridor",",","listening","for","any","sound","."]
    pos1 = ["PRON","VERB","ADV","ADP","DET","ADJ","NOUN","PUNCT","VERB","ADP","DET","NOUN","PUNCT"]
    r1 = run_test("Narrative (narrative scene)", e, s1, toks1, pos1,
                  [(0,1)]*8)
    s2 = '"What do you want?" he said. "I want to leave."'
    toks2 = ['"',"What","do","you","want",'?"',"he","said",'.','"',"I","want","to","leave",'."']
    pos2 = ["PUNCT","DET","AUX","PRON","VERB","PUNCT","PRON","VERB","PUNCT",
            "PUNCT","PRON","VERB","PART","VERB","PUNCT"]
    r2 = run_test("Narrative (dialogue scene)", e, s2, toks2, pos2,
                  [(0,1)]*8)
    return r1 and r2


def test_interaction():
    e = InteractionExtractor()
    s1 = "He walked alone in the forest."
    toks1 = ["He","walked","alone","in","the","forest","."]
    pos1 = ["PRON","VERB","ADV","ADP","DET","NOUN","PUNCT"]
    r1 = run_test("Interaction (no dialogue)", e, s1, toks1, pos1,
                  [(0,1)]*7)
    s2 = 'What do you want?'
    toks2 = ["What","do","you","want","?"]
    pos2 = ["DET","AUX","PRON","VERB","PUNCT"]
    r2 = run_test("Interaction (question)", e, s2, toks2, pos2,
                  [(0,1)]*7)
    s3 = '"I want to leave," he said. "Please let me go."'
    toks3 = ['"',"I","want","to","leave",',"',"he","said",'.','"',"Please","let","me","go",'."']
    pos3 = ["PUNCT","PRON","VERB","PART","VERB","PUNCT",
            "PRON","VERB","PUNCT","PUNCT","INTJ","VERB","PRON","VERB","PUNCT"]
    r3 = run_test("Interaction (quoted speech)", e, s3, toks3, pos3,
                  [(0,1)]*7)
    return r1 and r2 and r3


def test_symbolic():
    e = SymbolicExtractor()
    s1 = "The golden light of hope faded into the dark night of despair."
    toks1 = ["The","golden","light","of","hope","faded","into","the","dark","night","of","despair","."]
    pos1 = ["DET","ADJ","NOUN","ADP","NOUN","VERB","ADP","DET","ADJ","NOUN","ADP","NOUN","PUNCT"]
    r1 = run_test("Symbolic (dense)", e, s1, toks1, pos1,
                  [(0,1)]*7)
    s2 = "The cat sat on the mat."
    toks2 = ["The","cat","sat","on","the","mat","."]
    pos2 = ["DET","NOUN","VERB","ADP","DET","NOUN","PUNCT"]
    r2 = run_test("Symbolic (minimal)", e, s2, toks2, pos2,
                  [(0,1)]*7)
    e.reset_motif_memory()
    return r1 and r2


def test_evolution():
    e = EvolutionExtractor()
    s1 = "He changed completely, growing stronger every day."
    toks1 = ["He","changed","completely",",","growing","stronger","every","day","."]
    pos1 = ["PRON","VERB","ADV","PUNCT","VERB","ADJ","DET","NOUN","PUNCT"]
    r1 = run_test("Evolution (change)", e, s1, toks1, pos1,
                  [(-1,1)]+[(0,1)]*6)
    s2 = "The table was brown. The chair was next to it."
    toks2 = ["The","table","was","brown",".","The","chair","was","next","to","it","."]
    pos2 = ["DET","NOUN","AUX","ADJ","PUNCT","DET","NOUN","AUX","ADJ","ADP","PRON","PUNCT"]
    r2 = run_test("Evolution (static)", e, s2, toks2, pos2,
                  [(-1,1)]+[(0,1)]*6)
    return r1 and r2


def test_emotion():
    e = EmotionExtractor()
    s1 = "She felt a deep, terrible sadness that consumed her entirely."
    toks1 = ["She","felt","a","deep",",","terrible","sadness","that","consumed","her","entirely","."]
    pos1 = ["PRON","VERB","DET","ADJ","PUNCT","ADJ","NOUN","DET","VERB","PRON","ADV","PUNCT"]
    r1 = run_test("Emotion (negative)", e, s1, toks1, pos1,
                  [(-1,1)]*7)
    s2 = "The building was constructed in 1923. It has six floors."
    toks2 = ["The","building","was","constructed","in","1923",".","It","has","six","floors","."]
    pos2 = ["DET","NOUN","AUX","VERB","ADP","NUM","PUNCT","PRON","VERB","NUM","NOUN","PUNCT"]
    r2 = run_test("Emotion (neutral)", e, s2, toks2, pos2,
                  [(-1,1)]*7)
    return r1 and r2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Testing Feature Extractors\n")
    results = [
        ("NarrativeExtractor", test_narrative()),
        ("InteractionExtractor", test_interaction()),
        ("SymbolicExtractor", test_symbolic()),
        ("EvolutionExtractor", test_evolution()),
        ("EmotionExtractor", test_emotion()),
    ]
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"\nResults: {passed} passed, {failed} failed")
