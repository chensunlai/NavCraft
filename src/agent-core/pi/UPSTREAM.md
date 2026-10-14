# Pi upstream

Version: v1.2.0
Commit: 06a466935536f8c9bcacb2d5efc0fc021be1bdff
Source: https://github.com/earendil-works/pi/tree/v1.2.0/packages/coding-agent
License: MIT (LICENSE alongside this file)

The complete coding-agent package is retained. Other Pi packages are exact npm dependencies. Runtime imports resolve this workspace package; rebuild after changing its TypeScript sources.

Local changes:

- Standalone TypeScript build configuration, SDK-only build command, and exact dependency versions.
- `coding-agent/src/core/compaction/compaction.ts`: the native summary context builder includes NavCraft's navigation requirements. Initial summaries, updates to an existing summary, and split-turn prefix summaries all use this source-owned prompt. It preserves observation times, references and uncertainty without changing native cut points, summary formats or session persistence. Requirements apply when navigation information is present, including in ordinary Coding sessions of this local fork. Session-tree branch summaries use their own generator and are not changed.

The Harness transport wrapper still enforces the shared request budget and records summaries; it does not inject or replace the summary prompt. Public image/state restoration remains separate from summary generation.
