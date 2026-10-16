# brand-positioning

Strategy-phase agent of the **LinkGen** content-intelligence and publishing
system.

## Purpose

The specialist agent responsible for defining expert positioning: it consumes
the research/audience outputs (`ResearchRequest`, optional `brand_name`,
optional existing `BrandProfile`, optional `AudienceProfile[]`) and produces
contract-conforming `BrandProfile` + `Positioning` artifacts — statement,
differentiation, and justification grounded in the supplied research.

Design highlights (see `AGENTS.md` in the parent for the master rules):

- **Deterministic by construction** — brand ids derive from the input
  identity (brand name, else focus) and positioning ids from
  (run, request) with uuid5; time comes from the injected context clock.
- **LLM output is untrusted** — model JSON arrives router-validated and is
  sanitized field-by-field; empty statements/missions fall back to
  input-derived text, lists are bounded, and non-object fields fail closed.
- **Supplied brand wins** — when a `BrandProfile` is supplied in the input it
  is used verbatim and no model call is made.
- **Anti-positioning rules enforced in the prompt** — must not sound like a
  generic agency, motivational speaker, corporate social manager,
  salesperson, AI-generated content account, or desperate client-seeker;
  never invents credentials, results, statistics, or personal stories.
- **Mock-first ($0 development)** — the default provider is the offline
  deterministic `MockLLMProvider` via the `ModelRouter`.
- **No privileged tools** — no publishing/scheduling authority.

## Input / output contract

Input (validated against research/strategy contracts):

```json
{"request": {"request_id": "...", "run_id": "...", "focus": "..."},
 "brand_name": "Jane Doe",
 "brand": {...optional BrandProfile...},
 "audiences": [{...AudienceProfile...}]}
```

`request` is required; everything else is optional.

Output:

```json
{"brand_profile": {...BrandProfile...},
 "positioning": {...Positioning...}}
```

## Boundaries

- Output contracts: `BrandProfile`, `Positioning`.
- Forbidden tools: `publish`, `schedule`, `linkedin-post`.
- Depends only on `linkgen_contracts`, `linkgen_config`, `agent-framework`,
  and the `ModelRouter` interface from `providers/llm`.

## Testing

Run local tests with:

    python -m pytest

45 tests.

## Status

Phase 09 (strategy system) — implemented at 1.0.0.

## License

Proprietary — © Entelechy Labs. All rights reserved. No license is granted
to use, copy, modify, or distribute this software, except as agreed in
writing with the copyright holder.

## Publishing authority

No publishing/scheduling authority; publishing privilege belongs to the
publishing subsystem (publisher) only.
