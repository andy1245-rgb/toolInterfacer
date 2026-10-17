import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI): void {
	// P3: registerTool("useTool", ...) per spec §3; bridge to the service;
	// expose /ti command for status. Stub now so the tool name + shape is fixed.
	void pi;
}
