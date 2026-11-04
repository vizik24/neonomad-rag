Decomposition: drop the brand name from sub-queries and name policies, not products

## Why
Baseline failures (d06, m02, m05) were retrieval misses caused by the decomposer writing sub-queries like
"Neo Nomad mattress piece sizes" and "BikerBed warranty coverage". Every page mentions the brand, so those words
pulled the query embedding towards generic company/product chunks. Retrieval-only checks confirmed that removing
the brand moved the correct chunk to rank 1 in all three cases.

## What changed
`DECOMPOSE_PROMPT` in `query.py`: two new rules. Leave out "Neo Nomad"; name a product only for its own features,
and name the policy ("warranty cover for damage during installation") for policy topics.

## Expected effect
Higher retrieval_hit and pass on multi_topic and direct; no change expected on policy/unanswerable/adversarial.
