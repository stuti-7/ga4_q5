# Week 4 Q5 - GraphRAG Pipeline (Vercel)

Endpoints: `POST /extract-graph` · `POST /graph-query` · `POST /community-summary`

Heuristic (regex-based) entity/relationship extraction + BFS multi-hop reasoning over
the given graph + templated community summaries. No external LLM calls.

## Deploy Steps

### Vercel
1. Import this folder as a project on vercel.com (or `vercel --prod` from inside it).
2. No environment variables required.

### Submission
Submit the **base URL**: `https://your-app.vercel.app`
