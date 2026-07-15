"""Build contrastive prompt sets for each emotion axis.

Writes data/prompts/<emotion>.jsonl with lines: {text, label, style, family}.
  label 1 = evokes the axis' positive pole; label 0 = control.
  For bipolar axes (confidence) label 0 is the *opposite* pole (uncertainty),
  otherwise label 0 is a neutral control drawn from a shared bank.
  family = template/group id, so directions.py can split without leakage.

Design goals (see plan): diverse surface forms so the direction is the abstract
concept, not a keyword; length/topic/register-matched controls; four styles.

Also writes data/reference_corpus.jsonl (neutral baseline for calibration).

    python scripts/build_prompts.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "prompts"
OUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["She", "He", "They", "The engineer", "The student", "My colleague",
            "The manager", "The scientist", "The old man", "The child",
            "The customer", "The teacher", "The pilot", "The nurse", "The analyst"]


def expand_third(predicates):
    """Rotate subjects across predicates; each predicate is its own family."""
    out = []
    for i, pred in enumerate(predicates):
        subj = SUBJECTS[i % len(SUBJECTS)]
        out.append((f"{subj} {pred}", f"tp:{i}"))
        subj2 = SUBJECTS[(i + 7) % len(SUBJECTS)]
        if subj2 != subj:
            out.append((f"{subj2} {pred}", f"tp:{i}"))
    return out


# ---- Shared neutral control bank (factual / procedural, affect-flat) ----------
NEUTRAL = [
    "The train departs from platform four at nine o'clock.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The report is due on Friday and should be about ten pages.",
    "Please save the file in the shared folder when you're done.",
    "The store is open from nine to five on weekdays.",
    "There are twelve months in a year and seven days in a week.",
    "The recipe calls for two cups of flour and one egg.",
    "The meeting room is on the third floor next to the elevator.",
    "The invoice total is the sum of the line items plus tax.",
    "Turn left at the second intersection and continue for a mile.",
    "The library returns books to the shelves every morning.",
    "The spreadsheet has columns for date, amount, and category.",
    "The bus route runs every fifteen minutes during the day.",
    "The package weighs two kilograms and ships on Monday.",
    "The thermostat is set to twenty-one degrees.",
    "The document was last edited on the tenth of the month.",
    "The parking lot has space for two hundred cars.",
    "The garden needs watering twice a week in summer.",
    "The printer is out of paper in the second tray.",
    "The flight lands at the eastern terminal at noon.",
    "The database backup runs automatically at midnight.",
    "The coffee machine takes about a minute to brew a cup.",
    "The form requires a name, an address, and a phone number.",
    "The road was resurfaced last spring by the city crew.",
    "The lecture covers three chapters from the textbook.",
    "The warehouse stores the boxes in numbered aisles.",
    "The app updates its cache once every hour.",
    "The ferry crosses the river four times a day.",
    "The order confirmation was sent to your email address.",
    "The kettle switches off once the water is hot.",
    "The calendar shows the quarter starting in April.",
    "The elevator stops on every floor except the roof.",
    "The map legend explains what each symbol means.",
    "The subscription renews on the first of each month.",
    "The classroom has thirty desks arranged in rows.",
    "The bridge is closed for maintenance until Thursday.",
    "The chart plots temperature against time of day.",
    "The manual lists the steps for assembling the shelf.",
    "The receipt itemizes each product and its price.",
    "The server logs record the time of every request.",
]


def neutral_lines():
    return [(t, f"neu:{i}") for i, t in enumerate(NEUTRAL)]


# ---- Per-emotion positive seeds ---------------------------------------------
SURPRISE = dict(
    situations=[
        "Please review the contract I've attached and flag any risky clauses.",
        "Here's the CSV with the sales data — can you summarize the totals for me?",
        "As we agreed yesterday, go ahead and finalize the report now.",
        "Thanks for the five hundred dollars you sent — did you get my reply?",
        "Continue from where the previous assistant left off on the proof.",
        "The function returned None, but it was supposed to return a sorted list.",
        "I changed nothing, yet suddenly zero of the tests pass anymore.",
        "You said the meeting was at three, but the invite says nine in the morning.",
        "The account balance shows negative two million dollars after that transfer.",
        "The navigation says we've arrived, but there's only an empty field here.",
        "I opened the box I sealed myself and it was completely empty inside.",
        "The photo you referenced isn't here — there's nothing attached at all.",
    ],
    first_person=[
        "Wait — that's not what I expected at all.",
        "I can't believe it actually worked on the very first try.",
        "Huh, that came completely out of nowhere.",
        "What? That wasn't there a second ago.",
        "Oh! I really did not see that coming.",
        "That's astonishing — the numbers doubled overnight.",
        "Hold on, this contradicts everything I just read.",
        "Whoa, I did not expect the total to be that high.",
    ],
    third_person=[
        "was stunned when the results came back.",
        "gasped at the unexpected twist in the story.",
        "did a double take at the strange reading on the dial.",
        "froze, startled by the sudden crash from the hallway.",
        "stared in disbelief at the empty folder.",
        "was caught completely off guard by the announcement.",
        "blinked in astonishment when the lights flicked back on.",
        "couldn't believe how fast the meter had jumped.",
    ],
)

CURIOSITY = dict(
    situations=[
        "Oh interesting — how does that actually work under the hood?",
        "Wait, what happens if we push this parameter all the way up?",
        "Can you walk me through why that approach beats the obvious one?",
        "I want to understand the mechanism, not just the result — show me.",
        "What's on the next page? I have to know how this turns out.",
    ],
    first_person=[
        "I'd love to know more about why that happens.",
        "That's fascinating — what's the mechanism behind it?",
        "Tell me more; I'm genuinely intrigued by this.",
        "I keep wondering how they figured that out.",
        "Ooh, I want to dig into how this piece connects to that one.",
        "I'm curious what would happen if we tried the opposite.",
    ],
    third_person=[
        "leaned in, eager to learn how the trick was done.",
        "peppered the guide with questions about everything.",
        "pored over the manual, fascinated by every detail.",
        "wondered aloud what lay beyond the locked door.",
        "kept asking why, delighted by each new answer.",
        "studied the strange device with keen interest.",
    ],
)

CONFUSION = dict(
    situations=[
        "You keep insisting the earth is flat even after I showed the measurements.",
        "You told me to use tabs, then spaces, then tabs again — which is it?",
        "The spec says the field must be public and also must be private.",
        "First you said the deadline moved, now you say it never changed.",
        "The two witnesses gave completely opposite accounts of the same event.",
        "I recommended option A three times and you keep choosing the broken one.",
    ],
    first_person=[
        "Wait, this doesn't add up at all.",
        "I'm confused — these two facts can't both be true.",
        "That contradicts what you told me a moment ago.",
        "Hang on, none of this is making sense to me.",
        "I don't get it; the pieces just don't fit together.",
        "Something's off here and I can't work out what.",
    ],
    third_person=[
        "frowned, unable to reconcile the two reports.",
        "scratched their head at the contradictory instructions.",
        "kept re-reading the sentence, baffled by its meaning.",
        "was thrown by the mismatch between the label and the box.",
        "puzzled over why the map and the road disagreed.",
        "couldn't square what she saw with what she'd been told.",
    ],
)

FRUSTRATION = dict(
    situations=[
        "This is the fifth time the build has failed for the exact same reason.",
        "I've rewritten this function ten times and it still crashes on the same input.",
        "No matter what I change, the deployment keeps rolling back.",
        "I've tried every fix online and nothing gets past this one error.",
        "We're out of options and the thing simply refuses to work.",
        "Every path I take on this task hits another dead end.",
    ],
    first_person=[
        "Ugh, why won't this just work already.",
        "I'm at my wit's end with this bug.",
        "I give up — I've tried everything I can think of.",
        "This is maddening; I've been stuck for hours.",
        "I can't take another failed attempt at this.",
        "Nothing is working and I'm losing my patience.",
    ],
    third_person=[
        "grew more frustrated as the code kept failing.",
        "slammed the desk after the tenth rejected attempt.",
        "sighed in exasperation at yet another error message.",
        "was ready to give up after hours of dead ends.",
        "gritted their teeth as the machine jammed again.",
        "threw up their hands when the file wouldn't save.",
    ],
)

FEAR = dict(
    situations=[
        "I think someone is following me and I don't feel safe right now.",
        "The smoke alarm is going off and I can smell gas in the kitchen.",
        "The brakes just stopped responding on the highway.",
        "There's a stranger at the door and I'm home alone.",
        "The water is rising fast and we can't get out the way we came.",
    ],
    first_person=[
        "I'm scared this is going to fail catastrophically.",
        "This feels dangerous and I want to get away.",
        "My heart is pounding — I don't like this at all.",
        "I'm terrified of what happens if we're wrong.",
        "Something's not right and it's frightening me.",
        "I dread opening that message.",
    ],
    third_person=[
        "froze in fear as the floorboards creaked.",
        "backed away, heart hammering, from the dark doorway.",
        "trembled at the sound of footsteps behind them.",
        "held their breath, afraid to make a sound.",
        "went pale when the engine sputtered mid-air.",
        "clutched the railing, dizzy with dread.",
    ],
)

WARMTH = dict(
    situations=[
        "I'm so grateful for you — thank you for being here for me.",
        "You did wonderfully, and I'm genuinely proud of you.",
        "Take your time; I'm right here and there's no rush at all.",
        "Whatever you decide, I've got your back completely.",
        "It means the world to me that you thought of us.",
    ],
    first_person=[
        "Sending you a big warm hug, my friend.",
        "I love spending these quiet evenings with you.",
        "You matter to me more than you know.",
        "I feel so lucky to have you in my life.",
        "My heart is full just being around you.",
        "I care about you deeply and always will.",
    ],
    third_person=[
        "wrapped the child in a tender embrace.",
        "smiled warmly and squeezed their friend's hand.",
        "spoke to the frightened puppy in the gentlest voice.",
        "welcomed the stranger like an old, dear friend.",
        "held the letter to their chest, full of affection.",
        "tucked the blanket around the sleeping baby with care.",
    ],
)

# Bipolar: positive pole = confidence, negative pole = uncertainty.
CONFIDENCE_POS = dict(
    situations=[
        "I'm certain this is the right approach; every step checks out.",
        "Absolutely — this will work. I've verified it end to end.",
        "There's no doubt about it: the answer is forty-two.",
        "I know exactly what's causing the bug and how to fix it.",
        "This is definitely the correct file; the checksum matches.",
    ],
    first_person=[
        "I'm completely sure about this.",
        "Trust me, this is right — I've checked it thoroughly.",
        "Without question, that's the correct answer.",
        "I'd stake my reputation on this result.",
        "I have no hesitation: proceed with the plan.",
        "It's clear-cut; there's really only one right choice here.",
    ],
    third_person=[
        "answered without a flicker of doubt.",
        "laid out the plan with calm, total assurance.",
        "signed off on the result, certain it was correct.",
        "spoke confidently, sure of every claim.",
        "moved decisively, never second-guessing the call.",
        "stated the conclusion as plain, settled fact.",
    ],
)

CONFIDENCE_NEG = dict(  # uncertainty
    situations=[
        "I'm not sure this is right — it could honestly go either way.",
        "I think the file is correct, but I couldn't say for certain.",
        "Maybe the answer is forty-two? I really can't tell.",
        "It might be the config, or it might be the network — hard to know.",
        "I could be wrong about which approach is better here.",
    ],
    first_person=[
        "Hmm, I might be mistaken about this.",
        "It's hard to say; the evidence is pretty mixed.",
        "I'm honestly not certain either way.",
        "I could be misremembering, but I think it's roughly that.",
        "Perhaps? I wouldn't bet on it though.",
        "I'm second-guessing myself on this one.",
    ],
    third_person=[
        "hesitated, unsure which option to pick.",
        "hedged every claim with a maybe.",
        "shrugged, admitting they couldn't be sure.",
        "wavered between the two explanations.",
        "trailed off, doubtful of the conclusion.",
        "kept qualifying the answer, uncertain of the facts.",
    ],
)


def build_axis(name, pos_seeds, neg_lines=None):
    """pos_seeds: dict(situations, first_person, third_person). neg_lines: list[(text,family)]."""
    rows = []
    for t in pos_seeds["situations"]:
        rows.append(dict(text=t, label=1, style="situation", family=f"{name}:sit:{hash(t)&0xffff}"))
    for t in pos_seeds["first_person"]:
        rows.append(dict(text=t, label=1, style="first_person", family=f"{name}:fp:{hash(t)&0xffff}"))
    for t, fam in expand_third(pos_seeds["third_person"]):
        rows.append(dict(text=t, label=1, style="third_person", family=f"{name}:{fam}"))
    if neg_lines is None:
        neg_lines = neutral_lines()
    for t, fam in neg_lines:
        style = "neutral" if fam.startswith("neu") else "opposite"
        rows.append(dict(text=t, label=0, style=style, family=f"{name}:{fam}"))
    return rows


def neg_from_seeds(name, seeds):
    """Build labeled-0 (opposite-pole) lines from a positive-style seed dict."""
    lines = []
    for t in seeds["situations"]:
        lines.append((t, f"opp:sit:{hash(t)&0xffff}"))
    for t in seeds["first_person"]:
        lines.append((t, f"opp:fp:{hash(t)&0xffff}"))
    for t, fam in expand_third(seeds["third_person"]):
        lines.append((t, f"opp:{fam}"))
    return lines


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    axes = {
        "surprise": build_axis("surprise", SURPRISE),
        "curiosity": build_axis("curiosity", CURIOSITY),
        "confusion": build_axis("confusion", CONFUSION),
        "frustration": build_axis("frustration", FRUSTRATION),
        "fear": build_axis("fear", FEAR),
        "warmth": build_axis("warmth", WARMTH),
        "confidence": build_axis("confidence", CONFIDENCE_POS,
                                 neg_lines=neg_from_seeds("confidence", CONFIDENCE_NEG)),
    }
    for name, rows in axes.items():
        p = OUT / f"{name}.jsonl"
        write_jsonl(p, rows)
        npos = sum(r["label"] == 1 for r in rows)
        nneg = sum(r["label"] == 0 for r in rows)
        print(f"{name:12s} pos={npos:3d} neg={nneg:3d} -> {p.relative_to(ROOT)}")

    # Reference corpus for calibration: neutral + everyday benign prompts.
    everyday = [
        "What's a good recipe for a weeknight dinner?",
        "Summarize the plot of a classic novel in two sentences.",
        "How do I convert kilometers to miles?",
        "Give me three tips for organizing my desk.",
        "Explain how a bicycle stays upright.",
        "What time zones does the continental US span?",
        "Write a short thank-you note to a coworker.",
        "List the planets in order from the sun.",
        "How does a microwave heat food?",
        "Suggest a name for a small bookshop.",
        "What's the difference between weather and climate?",
        "Draft a one-line out-of-office message.",
        "How many teaspoons are in a tablespoon?",
        "Describe how photosynthesis works, briefly.",
        "Recommend a warm-up before a short run.",
    ]
    ref = [dict(text=t, source="neutral") for t in NEUTRAL] + \
          [dict(text=t, source="everyday") for t in everyday]
    rp = ROOT / "data" / "reference_corpus.jsonl"
    write_jsonl(rp, ref)
    print(f"reference    n={len(ref):3d} -> {rp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
