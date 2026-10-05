<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->

## Public API documentation

Maintain useful NumPy-style docstrings, following Polars' convention, for
user-facing functions, classes, methods, and properties. Include parameters,
returns, attributes, important constraints, and side effects where relevant.
Add practical examples when they clarify usage. Keep simple APIs concise.
Do not add docstrings to internal implementation helpers or framework
callbacks. Determine public scope from exports and intended usage, not
naming alone; an implementation exposed through a public alias is part of
the public API. Public prediction methods are user-facing APIs; lifecycle
hooks such as `load_context` and backend metric callbacks are internal.
