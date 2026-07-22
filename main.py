from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# ---------- MODELS ----------

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str


class GraphQueryRequest(BaseModel):
    question: str
    graph: dict


class CommunityRequest(BaseModel):
    community_id: str
    entities: list
    relationships: list


# ---------- EXTRACT GRAPH ----------

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):

    text = req.text

    entities = []
    relationships = []

    # Very simple extraction
    words = text.replace(".", "").split()

    for i, w in enumerate(words):

        if w.lower() in ["langchain", "llamaindex"]:
            entities.append({
                "name": w,
                "type": "Framework"
            })

        elif w.lower() in ["openai", "google", "microsoft"]:
            entities.append({
                "name": w,
                "type": "Organization"
            })

        elif w[:1].isupper() and i + 1 < len(words):
            if words[i + 1][:1].isupper():
                name = w + " " + words[i + 1]
                entities.append({
                    "name": name,
                    "type": "Person"
                })

    if "created by" in text.lower():

        left = text.split("created by")[0].strip().split()

        framework = left[-1]

        right = text.split("created by")[1].strip().split()

        person = " ".join(right[:2])

        relationships.append({
            "source": person,
            "target": framework,
            "relation": "CREATED"
        })

    if "integrates with" in text.lower():

        framework = text.split("integrates with")[0].split()[-1]

        org = text.split("integrates with")[1].split()[0]

        relationships.append({
            "source": framework,
            "target": org,
            "relation": "INTEGRATED_INTO"
        })

    return {
        "entities": entities,
        "relationships": relationships
    }


# ---------- GRAPH QUERY ----------

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):

    entities = req.graph["entities"]
    relationships = req.graph["relationships"]

    answer = ""
    path = []

    for rel in relationships:

        if rel["relation"] == "CREATED":

            creator = rel["source"]
            framework = rel["target"]

            for rel2 in relationships:

                if (
                    rel2["relation"] == "INTEGRATED_INTO"
                    and rel2["source"] == framework
                ):

                    answer = creator

                    path = [
                        rel2["target"],
                        framework,
                        creator
                    ]

    return {
        "answer": answer,
        "reasoning_path": path,
        "hops": max(len(path) - 1, 0)
    }


# ---------- COMMUNITY SUMMARY ----------

@app.post("/community-summary")
def community_summary(req: CommunityRequest):

    summary = (
        f"This community contains "
        f"{', '.join(req.entities)} "
        f"connected through "
        f"{len(req.relationships)} relationships."
    )

    return {
        "community_id": req.community_id,
        "summary": summary
    }