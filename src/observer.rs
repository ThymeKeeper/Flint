// ---------------------------------------------------------------------------
// AgentObserver — streaming/status events decoupled from transport
// ---------------------------------------------------------------------------

/// Observer for live agent activity: text streaming and tool execution events.
/// Implemented by TUI to show real-time progress; Signal/REST passes None.
pub trait AgentObserver: Send + Sync {
    /// Called for each streamed text chunk from the LLM.
    fn on_text_chunk(&self, chunk: String);
    /// Called with the tool name just before each tool execution begins.
    fn on_tool_start(&self, name: &str);
    /// Called twice per tool: once with the formatted call line (before execution)
    /// and once with the formatted result line (after). Injected as stream chunks.
    fn on_tool_event(&self, text: String);
}
