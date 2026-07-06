# utter — Subvocal's terminal frontend

This directory is a vendored, heavily modified fork of the [pi agent harness](https://github.com/earendil-works/pi)
by Mario Zechner, absorbed into Subvocal as its terminal UI and wired directly to
`@subvocal/encode`'s `AgentLoop` (dual-brain generation, KV-native conversation state,
ideogram-tagged tool calls) instead of pi's original provider abstraction.

It is no longer tracked as an external dependency — see the root [README](../README.md) and
[ARCHITECTURE.md](../ARCHITECTURE.md) for how it fits into the rest of Subvocal. The original
license is preserved in [LICENSE](LICENSE) per its terms.
