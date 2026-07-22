import re
from collections import defaultdict

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="GraphRAG Pipeline")


# ---------- MODELS ----------

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str


class GraphQueryRequest(BaseModel):
    question: str
    graph: dict


class CommunityRequest(BaseModel):
    community_id: str
    entities: list[str]
    relationships: list[dict]


# ---------- HELPERS ----------

FRAMEWORK_MAP = {
    "langchain": "LangChain",
    "llamaindex": "LlamaIndex",
    "haystack": "Haystack",
    "semantic kernel": "Semantic Kernel",
    "graph rag": "Graph RAG",
}

ORG_MAP = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "microsoft": "Microsoft",
    "meta": "Meta",
    "aws": "AWS",
    "azure": "Azure",
    "cohere": "Cohere",
    "mistral": "Mistral",
    "nvidia": "NVIDIA",
}

PRODUCT_MAP = {
    "chatgpt": "ChatGPT",
    "gpt-4": "GPT-4",
    "gpt4": "GPT-4",
    "claude": "Claude",
    "copilot": "Copilot",
    "gemini": "Gemini",
}

PERSON_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def add_entity(entities, name, entity_type):
    normalized_name = normalize_name(name)
    if not normalized_name:
        return
    key = normalized_name.lower()
    entity = {"name": normalized_name, "type": entity_type}
    current = next((item for item in entities if item["name"].lower() == key), None)
    if current is None:
        entities.append(entity)
    elif entity_type == "Framework" and current["type"] != "Framework":
        current["type"] = "Framework"


def dedupe_entities(entities):
    seen = {}
    for entity in entities:
        key = entity["name"].lower()
        if key not in seen:
            seen[key] = entity
        elif entity["type"] == "Framework" and seen[key]["type"] != "Framework":
            seen[key] = entity
    return list(seen.values())


def match_entity_names(text: str, entity_map):
    names = []
    text_lower = text.lower()
    for key, value in entity_map.items():
        if re.search(rf"\b{re.escape(key)}\b", text_lower):
            names.append(value)
    return names


def extract_entities(text: str):
    entities = []
    text_lower = text.lower()

    for name in match_entity_names(text, FRAMEWORK_MAP):
        add_entity(entities, name, "Framework")
    for name in match_entity_names(text, ORG_MAP):
        add_entity(entities, name, "Organization")
    for name in match_entity_names(text, PRODUCT_MAP):
        add_entity(entities, name, "Product")

    # Extract explicit person names around relation clauses.
    for sentence in re.split(r"[.!?]+", text):
        if "created by" in sentence.lower():
            match = re.search(r"created by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", sentence)
            if match:
                add_entity(entities, match.group(1), "Person")
        if "developed by" in sentence.lower():
            match = re.search(r"developed by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", sentence)
            if match:
                add_entity(entities, match.group(1), "Person")
        if "founded by" in sentence.lower():
            match = re.search(r"founded by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", sentence)
            if match:
                add_entity(entities, match.group(1), "Person")
        if "authored by" in sentence.lower():
            match = re.search(r"authored by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", sentence)
            if match:
                add_entity(entities, match.group(1), "Person")
        if "hired" in sentence.lower():
            match = re.search(r"hired\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", sentence)
            if match:
                add_entity(entities, match.group(1), "Person")

    # Extra fallback for multi-token names that are clearly people.
    for match in PERSON_NAME_PATTERN.findall(text):
        add_entity(entities, match, "Person")

    return dedupe_entities(entities)


def find_entity_in_text(text: str, entity_names: list[str]):
    lower_text = text.lower()
    for name in entity_names:
        if re.search(rf"\b{re.escape(name.lower())}\b", lower_text):
            return name
    return None


def extract_relationships(text: str, entities):
    relationships = []
    text_lower = text.lower()
    entity_names = [entity["name"] for entity in entities]

    # prefer canonical names from the extracted entity set
    def canonicalize(token: str):
        token = normalize_name(token)
        for name in entity_names:
            if name.lower() == token.lower():
                return name
        return token.title() if token and token.islower() else token

    for sentence in re.split(r"[.!?]+", text):
        sentence_lower = sentence.lower()

        if "created by" in sentence_lower:
            left = sentence.split("created by", 1)[0].strip()
            right = sentence.split("created by", 1)[1].strip()
            framework = find_entity_in_text(left, entity_names) or canonicalize(left.split()[-1])
            source = normalize_name(re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right).group(1)) if re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right) else "Unknown Person"
            relationships.append({"source": source, "target": framework, "relation": "CREATED"})

        if "integrates with" in sentence_lower:
            left = sentence.split("integrates with", 1)[0].strip()
            right = sentence.split("integrates with", 1)[1].strip()
            framework = find_entity_in_text(left, entity_names) or canonicalize(left.split()[-1])
            organization = find_entity_in_text(right, entity_names) or canonicalize(right.split()[0])
            relationships.append({"source": framework, "target": organization, "relation": "INTEGRATED_INTO"})

        if "developed by" in sentence_lower:
            left = sentence.split("developed by", 1)[0].strip()
            right = sentence.split("developed by", 1)[1].strip()
            target = find_entity_in_text(left, entity_names) or canonicalize(left.split()[-1])
            source = normalize_name(re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right).group(1)) if re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right) else "Unknown Person"
            relationships.append({"source": source, "target": target, "relation": "DEVELOPED"})

        if "founded by" in sentence_lower:
            left = sentence.split("founded by", 1)[0].strip()
            right = sentence.split("founded by", 1)[1].strip()
            target = find_entity_in_text(left, entity_names) or canonicalize(left.split()[-1])
            source = normalize_name(re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right).group(1)) if re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right) else "Unknown Person"
            relationships.append({"source": source, "target": target, "relation": "FOUNDED"})

        if "authored by" in sentence_lower:
            left = sentence.split("authored by", 1)[0].strip()
            right = sentence.split("authored by", 1)[1].strip()
            target = find_entity_in_text(left, entity_names) or canonicalize(left.split()[-1])
            source = normalize_name(re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right).group(1)) if re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right) else "Unknown Person"
            relationships.append({"source": source, "target": target, "relation": "AUTHORED"})

        if "hired" in sentence_lower:
            left = sentence.split("hired", 1)[0].strip()
            right = sentence.split("hired", 1)[1].strip()
            target = find_entity_in_text(left, entity_names) or canonicalize(left.split()[-1])
            source = normalize_name(re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right).group(1)) if re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", right) else "Unknown Person"
            relationships.append({"source": source, "target": target, "relation": "HIRED"})

    return relationships


def infer_reasoning_path(question: str, graph: dict):
    question_lower = question.lower()
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])

    if not relationships:
        return {"answer": "", "reasoning_path": [], "hops": 0}

    target_org = None
    org_names = [entity["name"] for entity in entities if entity["type"] == "Organization"]
    for org in org_names:
        if org.lower() in question_lower:
            target_org = org
            break

    if target_org is None:
        for rel in relationships:
            if rel["relation"] == "INTEGRATED_INTO":
                if rel["target"].lower() in question_lower or rel["source"].lower() in question_lower:
                    target_org = rel["target"]
                    break

    if target_org:
        for rel in relationships:
            if rel["relation"] == "INTEGRATED_INTO" and rel["target"] == target_org:
                framework = rel["source"]
                for rel2 in relationships:
                    if rel2["relation"] == "CREATED" and rel2["target"] == framework:
                        return {
                            "answer": rel2["source"],
                            "reasoning_path": [target_org, framework, rel2["source"]],
                            "hops": 2,
                        }

    for rel in relationships:
        if rel["relation"] == "CREATED":
            return {
                "answer": rel["source"],
                "reasoning_path": [rel["target"], rel["source"]],
                "hops": 1,
            }

    return {"answer": "", "reasoning_path": [], "hops": 0}


def summarize_community(community_id: str, entities: list[str], relationships: list[dict]):
    framework = next((name for name in entities if any(rel["target"] == name for rel in relationships if rel["relation"] in {"CREATED", "DEVELOPED"})), None)
    creator = next((rel["source"] for rel in relationships if rel["relation"] == "CREATED" and rel["target"] == framework), None)
    orgs = [rel["target"] for rel in relationships if rel["relation"] == "INTEGRATED_INTO" and rel["source"] == framework]

    if framework and creator and orgs:
        summary = (
            f"This community centers around {framework}, an AI framework created by {creator} "
            f"that integrates with {', '.join(orgs)}."
        )
    elif framework and creator:
        summary = f"This community centers around {framework}, which was created by {creator}."
    else:
        summary = f"This community includes {', '.join(entities)} with {len(relationships)} relationships."

    return {"community_id": community_id, "summary": summary}


# ---------- ENDPOINTS ----------

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):
    entities = extract_entities(req.text)
    relationships = extract_relationships(req.text, entities)
    return {"entities": entities, "relationships": relationships}


@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):
    return infer_reasoning_path(req.question, req.graph)


@app.post("/community-summary")
def community_summary(req: CommunityRequest):
    return summarize_community(req.community_id, req.entities, req.relationships)


@app.get("/")
def root():
    return {"message": "GraphRAG Pipeline is running"}