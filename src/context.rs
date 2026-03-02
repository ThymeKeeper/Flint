use crate::config::ClaudeConfig;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub token_estimate: usize,
}

// ---------------------------------------------------------------------------
// ConversationContext
// ---------------------------------------------------------------------------

pub struct ConversationContext {
    messages: Vec<Message>,
    total_tokens: usize,
    config: ClaudeConfig,
}

impl ConversationContext {
    pub fn new(config: ClaudeConfig) -> Self {
        Self {
            messages: Vec::new(),
            total_tokens: 0,
            config,
        }
    }

    /// Push a new message to the conversation.
    pub fn push(&mut self, role: Role, content: String) {
        let token_estimate = estimate_tokens(&content);
        self.total_tokens += token_estimate;
        self.messages.push(Message {
            role,
            content,
            token_estimate,
        });
    }

    /// Get a reference to the messages list.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Current estimated token count.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Check if compaction is needed (tokens exceed threshold percentage of context limit).
    pub fn compaction_needed(&self) -> bool {
        let threshold =
            (self.config.context_limit as f64 * self.config.compaction_threshold) as usize;
        self.total_tokens > threshold
    }

    /// Remove and return the oldest half of messages for compaction.
    /// Keeps the newest half in place.
    pub fn take_oldest_half(&mut self) -> Vec<Message> {
        if self.messages.len() <= 1 {
            return Vec::new();
        }

        let split_point = self.messages.len() / 2;
        let oldest: Vec<Message> = self.messages.drain(..split_point).collect();

        // Recalculate total tokens for remaining messages
        self.total_tokens = self.messages.iter().map(|m| m.token_estimate).sum();

        oldest
    }

    /// Prepend a compaction summary as the first message in the conversation.
    /// This provides context about what was compacted.
    pub fn prepend_summary(&mut self, summary: String) {
        let token_estimate = estimate_tokens(&summary);
        self.total_tokens += token_estimate;
        self.messages.insert(
            0,
            Message {
                role: Role::User,
                content: summary,
                token_estimate,
            },
        );
    }

    /// Clear the entire conversation context.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.total_tokens = 0;
    }

    /// Number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the context is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Rough token estimation: 1 token per 4 characters, minimum 1 token.
pub fn estimate_tokens(s: &str) -> usize {
    (s.len() / 4).max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ClaudeConfig {
        ClaudeConfig {
            model: "test".to_string(),
            max_tokens: 4096,
            context_limit: 200000,
            compaction_threshold: 0.75,
        }
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 1); // min 1
        assert_eq!(estimate_tokens("hi"), 1); // 2/4 = 0, clamped to 1
        assert_eq!(estimate_tokens("hello world!!"), 3); // 13/4 = 3
        assert_eq!(estimate_tokens("a".repeat(400).as_str()), 100);
    }

    #[test]
    fn test_push_and_total_tokens() {
        let mut ctx = ConversationContext::new(test_config());
        assert_eq!(ctx.total_tokens(), 0);
        assert!(ctx.is_empty());

        ctx.push(Role::User, "Hello, how are you?".to_string());
        assert_eq!(ctx.len(), 1);
        assert!(ctx.total_tokens() > 0);

        let first_tokens = ctx.total_tokens();
        ctx.push(Role::Assistant, "I'm doing well!".to_string());
        assert_eq!(ctx.len(), 2);
        assert!(ctx.total_tokens() > first_tokens);
    }

    #[test]
    fn test_compaction_needed() {
        let config = ClaudeConfig {
            model: "test".to_string(),
            max_tokens: 100,
            context_limit: 100, // 75% = 75 tokens
            compaction_threshold: 0.75,
        };
        let mut ctx = ConversationContext::new(config);
        assert!(!ctx.compaction_needed());

        // Push enough text to exceed 75 tokens at 4 chars/token
        // Need > 75 tokens = > 300 chars
        ctx.push(Role::User, "x".repeat(400));
        assert!(ctx.compaction_needed());
    }

    #[test]
    fn test_compaction_not_needed() {
        let mut ctx = ConversationContext::new(test_config());
        ctx.push(Role::User, "Short message".to_string());
        // 200000 * 0.75 = 150000 tokens; "Short message" is ~3 tokens
        assert!(!ctx.compaction_needed());
    }

    #[test]
    fn test_take_oldest_half() {
        let mut ctx = ConversationContext::new(test_config());
        ctx.push(Role::User, "msg 1".to_string());
        ctx.push(Role::Assistant, "resp 1".to_string());
        ctx.push(Role::User, "msg 2".to_string());
        ctx.push(Role::Assistant, "resp 2".to_string());

        assert_eq!(ctx.len(), 4);
        let oldest = ctx.take_oldest_half();

        // 4 messages -> split at 2 -> oldest = 2 messages, remaining = 2 messages
        assert_eq!(oldest.len(), 2);
        assert_eq!(ctx.len(), 2);
        assert_eq!(oldest[0].content, "msg 1");
        assert_eq!(oldest[1].content, "resp 1");
        assert_eq!(ctx.messages()[0].content, "msg 2");
        assert_eq!(ctx.messages()[1].content, "resp 2");
    }

    #[test]
    fn test_take_oldest_half_single_message() {
        let mut ctx = ConversationContext::new(test_config());
        ctx.push(Role::User, "only message".to_string());

        let oldest = ctx.take_oldest_half();
        assert_eq!(oldest.len(), 0); // Can't split a single message
        assert_eq!(ctx.len(), 1);
    }

    #[test]
    fn test_take_oldest_half_empty() {
        let mut ctx = ConversationContext::new(test_config());
        let oldest = ctx.take_oldest_half();
        assert_eq!(oldest.len(), 0);
    }

    #[test]
    fn test_take_oldest_half_recalculates_tokens() {
        let mut ctx = ConversationContext::new(test_config());
        ctx.push(Role::User, "a".repeat(100)); // 25 tokens
        ctx.push(Role::Assistant, "b".repeat(200)); // 50 tokens
        ctx.push(Role::User, "c".repeat(100)); // 25 tokens
        ctx.push(Role::Assistant, "d".repeat(200)); // 50 tokens

        let total_before = ctx.total_tokens();
        assert_eq!(total_before, 150); // 25+50+25+50

        let oldest = ctx.take_oldest_half();
        let oldest_tokens: usize = oldest.iter().map(|m| m.token_estimate).sum();

        // Remaining tokens should equal total - oldest
        assert_eq!(ctx.total_tokens(), total_before - oldest_tokens);
    }

    #[test]
    fn test_prepend_summary() {
        let mut ctx = ConversationContext::new(test_config());
        ctx.push(Role::User, "current message".to_string());

        let tokens_before = ctx.total_tokens();
        ctx.prepend_summary("[Compacted: summary here]".to_string());

        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.messages()[0].content, "[Compacted: summary here]");
        assert_eq!(ctx.messages()[1].content, "current message");
        assert!(ctx.total_tokens() > tokens_before);
    }

    #[test]
    fn test_clear() {
        let mut ctx = ConversationContext::new(test_config());
        ctx.push(Role::User, "hello".to_string());
        ctx.push(Role::Assistant, "hi".to_string());
        assert!(!ctx.is_empty());

        ctx.clear();
        assert!(ctx.is_empty());
        assert_eq!(ctx.total_tokens(), 0);
        assert_eq!(ctx.len(), 0);
    }

    #[test]
    fn test_role_as_str() {
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }
}
