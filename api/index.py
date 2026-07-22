import os
import re
from collections import deque, Counter
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum

EMAIL = os.getenv("EMAIL", "23f2004598@ds.study.iitm.ac.in")

app = FastAPI(title="TDS Week4 Q5 - GraphRAG Pipeline")

# ───────────────────────────── Extraction heuristics ──────────────────────────

STOPWORDS_LEADING = {
    "The", "This", "That", "These", "Those", "It", "They", "He", "She",
    "A", "An", "In", "On", "For", "With", "After", "Before", "When",
    "While", "Since", "As", "If", "But", "And", "Or", "Its",
}

# single capitalized words that are never entities on their own (months, days, etc)
NON_ENTITY_WORDS = {
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}

ORG_SUFFIXES = ("Inc", "Corp", "Corporation", "Labs", "Technologies", "Systems",
                "LLC", "Company", "Ltd", "Group", "Ventures", "Capital")

FRAMEWORK_HINTS = ("GPT", "Chain", "SDK", "API", "Lib", "Kit", "Flow",
                   "Engine", "Core", "Base", "DB", "NLP", "LLM")

# well-known companies whose names happen to contain framework-ish substrings
# (e.g. "OpenAI" contains "AI") that would otherwise be misclassified.
KNOWN_ORGS = {
    "OpenAI", "Google", "Microsoft", "Meta", "Amazon", "Anthropic",
    "DeepMind", "Apple", "IBM", "Nvidia", "Tesla", "Databricks",
}

# entity span: sequences of Capitalized / camel-ish tokens, allowing lowercase
# connector words like "of"/"for"/"the" in the middle (e.g. "Bank of America").
# NOTE: "and" is deliberately excluded here -- it's far more often a list separator
# between two DISTINCT entities ("Tesla and SpaceX") than part of one proper name,
# so allowing it would wrongly merge separate entities into one bogus name.
SINGLE_ENT_PATTERN = r"[A-Z][a-zA-Z0-9]*(?:\s+(?:of|for|the|&)\s+[A-Z][a-zA-Z0-9]*|\s+[A-Z][a-zA-Z0-9]*){0,4}"
ENTITY_RE = re.compile(rf"\b{SINGLE_ENT_PATTERN}\b")


def _list_entity_pattern(group_name: str) -> str:
    """A comma/and-separated list of one or more entity phrases, e.g.
    'Tesla, SpaceX and Neuralink', captured whole under the given group name."""
    item = rf"(?:(?:(?i:the|a|an)\s+)?{SINGLE_ENT_PATTERN})"
    return rf"(?P<{group_name}>{item}(?:\s*,\s*{item})*(?:\s*,?\s*(?i:and)\s*{item})?)"


def _split_entity_list(text: str) -> list:
    parts = re.split(r"\s*,\s*|\s+(?i:and)\s+", text.strip())
    cleaned = []
    for p in parts:
        # strip a leading "and "/"the "/"a "/"an " left over from Oxford-comma
        # lists ("X, Y, and Z") where the comma-split leaves "and Z" as one piece.
        p = re.sub(r"^(?i:and|the|a|an)\s+", "", p.strip()).strip()
        if p:
            cleaned.append(p)
    return cleaned

# raw pattern strings: "ENT" is a placeholder later substituted with a case-sensitive
# entity regex; trigger words are wrapped in (?i:...) so only THEY are case-insensitive
# -- the ENT groups must stay case-sensitive or [A-Z] would also swallow lowercase words.
#
# Convention: "b" is always the grammatical subject/agent, "a" is always the
# grammatical object/target -- relationships are emitted as source=b, target=a.
FOUNDED_V = "founded|co-founded|cofounded|established|started|incorporated|set up|spun out"
DEVELOPED_V = "developed|created|built|designed|engineered|invented|coded|programmed|released|shipped"
HIRED_V = "hired|recruited|employed|appointed|onboarded|signed"
AUTHORED_V = "authored|wrote|published|co-authored|penned|co-wrote|drafted"
INTEGRATED_V = "integrates|integrated"
WORKS_V = "works|worked"
ROLE_NOUN = "framework|product|tool|library|platform|company|startup|app|service|model"

# optional short lowercase-led comma aside between a subject and its verb, e.g.
# "LangChain, Harrison Chase's framework, integrates with OpenAI" -- the aside
# must start with a lowercase word (article/pronoun/etc) to avoid swallowing a
# genuine list of capitalized entities as if it were a descriptive aside.
ASIDE = (
    r"(?:,\s*(?:[a-z][a-z '\-]{1,60}"
    rf"|{SINGLE_ENT_PATTERN}('s|s')\s[a-z][a-z '\-]{{0,50}}),)?\s+"
)

# (subject-first "b VERB a" pattern, relation) -- an aside-tolerant variant of
# each is generated automatically below.
_ACTIVE_CORE = [
    (rf"(?P<b>ENT) (?i:{FOUNDED_V}) (?P<a>ENT)", "FOUNDED"),
    (rf"(?P<b>ENT) (?i:{DEVELOPED_V}) (?P<a>ENT)", "DEVELOPED"),
    (rf"(?P<b>ENT) (?i:{INTEGRATED_V}) (?i:with|into) (?P<a>ENT)", "INTEGRATED_INTO"),
    (rf"(?P<b>ENT) (?i:{WORKS_V}) with (?P<a>ENT)", "INTEGRATED_INTO"),
    (rf"(?P<b>ENT) (?i:supports|connects to|plugs into|is compatible with) (?P<a>ENT)", "INTEGRATED_INTO"),
    (rf"(?P<b>ENT) (?i:{HIRED_V}) (?P<a>ENT)", "HIRED"),
    (rf"(?P<b>ENT) (?i:{AUTHORED_V}) (?P<a>ENT)", "AUTHORED"),
]

RELATION_PATTERNS = []
for _pat, _rel in _ACTIVE_CORE:
    RELATION_PATTERNS.append((_pat, _rel))
    RELATION_PATTERNS.append((_pat.replace("(?P<b>ENT) ", f"(?P<b>ENT){ASIDE}"), _rel))

RELATION_PATTERNS += [
    (rf"(?P<a>ENT) (?i:was (?:{FOUNDED_V}) by) (?P<b>ENT)", "FOUNDED"),
    (rf"(?P<a>ENT) (?i:was (?:{DEVELOPED_V}) by) (?P<b>ENT)", "DEVELOPED"),
    (rf"(?P<a>ENT), (?i:an?|the) [a-z ]*? (?i:{DEVELOPED_V}) by (?P<b>ENT)", "DEVELOPED"),
    (rf"(?P<a>ENT) (?i:is|was) (?i:{INTEGRATED_V}) into (?P<b>ENT)", "INTEGRATED_INTO"),
    (rf"(?P<a>ENT) (?i:was (?:{HIRED_V}) by) (?P<b>ENT)", "HIRED"),
    (rf"(?P<a>ENT) (?i:was (?:authored|written) by) (?P<b>ENT)", "AUTHORED"),

    # possessive / appositive constructions common in bios and profiles
    (rf"(?P<b>ENT)('s| s) (?i:{ROLE_NOUN}) (?P<a>ENT)", "DEVELOPED"),
    (rf"(?P<a>ENT), (?P<b>ENT)('s| s) (?i:{ROLE_NOUN})", "DEVELOPED"),
    (rf"(?P<b>ENT), (?i:the |a |an )?(?i:founder|co-founder) of (?P<a>ENT)", "FOUNDED"),
    (rf"(?P<a>ENT)('s| s) (?i:founder|co-founder),? (?P<b>ENT)", "FOUNDED"),
    (rf"(?P<b>ENT), (?i:the |a |an )?(?i:creator|developer) of (?P<a>ENT)", "DEVELOPED"),
    (rf"(?P<a>ENT)('s| s) (?i:creator|developer),? (?P<b>ENT)", "DEVELOPED"),
    (rf"(?P<b>ENT), (?i:the |a |an )?(?i:author) of (?P<a>ENT)", "AUTHORED"),
]


def find_entity_spans(text: str):
    spans = []
    for m in ENTITY_RE.finditer(text):
        phrase = m.group(0)
        if len(phrase.split()) == 1 and (phrase in STOPWORDS_LEADING or phrase in NON_ENTITY_WORDS):
            continue
        spans.append((m.start(), m.end(), phrase))
    return spans


def classify_entity(name: str, roles: Counter) -> str:
    if name in KNOWN_ORGS:
        return "Organization"
    if name.endswith(ORG_SUFFIXES) or any(name.split()[-1] == s for s in ORG_SUFFIXES):
        return "Organization"
    if roles["hired_person"]:
        return "Person"
    if roles["agent_person_like"] and len(name.split()) >= 2 and not any(h in name for h in FRAMEWORK_HINTS):
        return "Person"
    if any(h in name for h in FRAMEWORK_HINTS):
        return "Framework"
    if roles["founded_org"] or roles["hirer"]:
        return "Organization"
    if roles["authored_work"]:
        return "Product"
    if roles["created_thing"]:
        return "Framework"
    if roles["agent_person_like"]:
        # single-word agent of founded/hired/etc that isn't a two-word person name
        # or known framework -- most likely a company.
        return "Organization"
    return "Product"


class ExtractRequest(BaseModel):
    chunk_id: str
    text: str


@app.post("/extract-graph")
async def extract_graph(req: ExtractRequest):
    text = req.text
    spans = find_entity_spans(text)
    # build a lookup of unique entity names -> role counters
    role_info: dict[str, Counter] = {}

    def ensure(name):
        if name not in role_info:
            role_info[name] = Counter()
        return role_info[name]

    relationships = []
    seen_rel = set()

    def record(a, b, relation):
        if not a or not b or a == b:
            return
        key = (a, b, relation)
        if key in seen_rel:
            return
        seen_rel.add(key)
        relationships.append({"source": b, "target": a, "relation": relation})
        if relation == "HIRED":
            ensure(b)["hirer"] += 1
            ensure(a)["hired_person"] += 1
        elif relation == "FOUNDED":
            ensure(b)["agent_person_like"] += 1
            ensure(a)["founded_org"] += 1
        elif relation == "AUTHORED":
            ensure(b)["agent_person_like"] += 1
            ensure(a)["authored_work"] += 1
        else:  # DEVELOPED, INTEGRATED_INTO
            ensure(b)["agent_person_like"] += 1
            ensure(a)["created_thing"] += 1

    # Fallback keyword map: used only for sentences where none of the specific
    # verb patterns above matched anything, so an unanticipated phrasing still
    # contributes a linked pair instead of silently dropping it. Scoped per
    # sentence and restricted to ADJACENT entities only, to avoid inventing
    # spurious links between unrelated items in the same sentence (e.g. two
    # siblings in a list, like "Tesla" and "SpaceX" in "Musk founded Tesla and
    # SpaceX", which are already each linked to Musk and shouldn't also be
    # linked to each other).
    FALLBACK_KEYWORDS = {
        "FOUNDED": ["found", "establish", "start", "launch", "incorporat", "creat", "spun", "set up"],
        "DEVELOPED": ["develop", "build", "built", "design", "engineer", "invent", "code", "program",
                      "creat", "release", "ship", "power"],
        "HIRED": ["hire", "recruit", "employ", "appoint", "onboard", "join", "sign"],
        "AUTHORED": ["author", "wrote", "written", "publish", "pen", "draft"],
        "INTEGRATED_INTO": ["integrat", "work with", "support", "connect", "plug", "compatib",
                             "leverag", "embed", "extend", "complement", "use", "combine", "pair", "partner"],
    }

    def guess_relation(sentence_lower: str) -> str:
        best_relation, best_hits = "INTEGRATED_INTO", 0
        for relation, keywords in FALLBACK_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in sentence_lower)
            if hits > best_hits:
                best_relation, best_hits = relation, hits
        return best_relation

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not sentence.strip():
            continue

        found_specific = False
        for pattern_str, relation in RELATION_PATTERNS:
            for m in re.finditer(_dual_entity_regex(pattern_str), sentence):
                a_list = _split_entity_list(m.group("a"))
                b_list = _split_entity_list(m.group("b"))
                for b in b_list:
                    for a in a_list:
                        record(a, b, relation)
                        found_specific = True

        if found_specific:
            continue

        # no specific pattern matched this sentence -- try the adjacent-pair fallback
        sent_spans = find_entity_spans(sentence)
        names_in_order = []
        seen_in_sentence = set()
        for _, _, phrase in sent_spans:
            if phrase not in seen_in_sentence:
                seen_in_sentence.add(phrase)
                names_in_order.append(phrase)
        if len(names_in_order) < 2:
            continue
        relation_guess = guess_relation(sentence.lower())
        for i in range(len(names_in_order) - 1):
            record(names_in_order[i + 1], names_in_order[i], relation_guess)

    # also register any entity spans not involved in a relation, so they're not lost
    for _, _, phrase in spans:
        ensure(phrase)

    entities = []
    seen_names = set()
    for name, roles in role_info.items():
        if name in seen_names:
            continue
        seen_names.add(name)
        entities.append({"name": name, "type": classify_entity(name, roles)})

    return {"entities": entities, "relationships": relationships}


def _dual_entity_regex(pattern_str: str) -> str:
    """Replace the two occurrences of literal 'ENT' placeholder with list-capable group
    patterns named a/b, so phrases like "Tesla and SpaceX" or "the Deep Learning textbook"
    are matched (and later split) as intended, instead of silently failing to match or
    being merged into one bogus entity."""
    out = pattern_str
    out = out.replace("(?P<a>ENT)", _list_entity_pattern("a"))
    out = out.replace("(?P<b>ENT)", _list_entity_pattern("b"))
    return out


# ───────────────────────────── Graph query (multi-hop) ────────────────────────

class GraphPayload(BaseModel):
    entities: list
    relationships: list


class QueryRequest(BaseModel):
    question: str
    graph: GraphPayload


def infer_target_type(question: str) -> Optional[str]:
    q = question.lower()
    if q.strip().startswith("who") or " who " in q:
        return "Person"
    if "framework" in q:
        return "Framework"
    if "product" in q:
        return "Product"
    if "organization" in q or "company" in q:
        return "Organization"
    return None


def build_adjacency(relationships):
    adj: dict[str, list[tuple[str, str]]] = {}
    for r in relationships:
        s, t, rel = r["source"], r["target"], r.get("relation", "")
        adj.setdefault(s, []).append((t, rel))
        adj.setdefault(t, []).append((s, rel))
    return adj


def find_anchor(question: str, entities: list) -> Optional[str]:
    q = question.lower()
    best = None
    for e in entities:
        name = e["name"] if isinstance(e, dict) else e
        if name.lower() in q:
            if best is None or len(name) > len(best):
                best = name
    return best


@app.post("/graph-query")
async def graph_query(req: QueryRequest):
    entities = req.graph.entities
    relationships = req.graph.relationships
    name_to_type = {}
    for e in entities:
        if isinstance(e, dict):
            name_to_type[e["name"]] = e.get("type")
        else:
            name_to_type[e] = None

    anchor = find_anchor(req.question, entities)
    target_type = infer_target_type(req.question)

    if anchor is None:
        return {"answer": None, "reasoning_path": [], "hops": 0}

    adj = build_adjacency(relationships)

    # BFS from anchor
    visited = {anchor}
    parent = {}
    queue = deque([anchor])
    found = None
    while queue:
        cur = queue.popleft()
        if cur != anchor:
            cur_type = name_to_type.get(cur)
            if target_type is None or cur_type == target_type:
                found = cur
                break
        for neighbor, _rel in adj.get(cur, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = cur
                queue.append(neighbor)

    if found is None:
        # fall back: nearest neighbor regardless of type
        for neighbor, _rel in adj.get(anchor, []):
            found = neighbor
            parent[neighbor] = anchor
            break

    if found is None:
        return {"answer": None, "reasoning_path": [anchor], "hops": 0}

    # reconstruct path
    path = [found]
    node = found
    while node in parent:
        node = parent[node]
        path.append(node)
    path.reverse()

    return {"answer": found, "reasoning_path": path, "hops": len(path) - 1}


# ───────────────────────────── Community summary ──────────────────────────────

RELATION_PHRASES = {
    "FOUNDED": "founded",
    "DEVELOPED": "developed",
    "INTEGRATED_INTO": "integrates into",
    "HIRED": "hired",
    "AUTHORED": "authored",
    "CREATED": "created",
}


class SummaryRequest(BaseModel):
    community_id: str
    entities: list
    relationships: list


@app.post("/community-summary")
async def community_summary(req: SummaryRequest):
    entities = req.entities
    relationships = req.relationships

    degree = Counter()
    for r in relationships:
        degree[r["source"]] += 1
        degree[r["target"]] += 1

    central = None
    if degree:
        central = max(degree.items(), key=lambda kv: kv[1])[0]
    elif entities:
        central = entities[0] if isinstance(entities[0], str) else entities[0].get("name")

    sentences = []
    if central:
        sentences.append(f"This community centers around {central}.")

    for r in relationships:
        s, t, rel = r["source"], r["target"], r.get("relation", "")
        phrase = RELATION_PHRASES.get(rel, rel.lower().replace("_", " ") or "is related to")
        sentences.append(f"{s} {phrase} {t}.")

    summary = " ".join(sentences) if sentences else f"Community {req.community_id} has no recorded relationships."

    return {"community_id": req.community_id, "summary": summary}


# ───────────────────────────── Misc ────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "question": "Week4 Q5 - GraphRAG Pipeline", "email": EMAIL}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


handler = Mangum(app, lifespan="off")
