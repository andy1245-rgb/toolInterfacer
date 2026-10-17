# pi-extension — `useTool` for pi

TypeScript extension: registers the `useTool` tool the LLM calls, bridges to
the decision service over localhost HTTP, pushes the rolling session digest,
spawns/health-polls the service process.

P3 wires this; src/ currently carries typed stubs so the contract is visible.
