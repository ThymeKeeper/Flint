//! Terminal user interface — scrollable chat history + multi-line text input.
//!
//! The TUI runs on a dedicated blocking thread (via `tokio::task::spawn_blocking`).
//! Two `tokio::sync::mpsc` channels bridge the async main loop and the blocking
//! TUI thread:
//!   • `user_input_tx`    — TUI → main loop (user-submitted text)
//!   • `agent_update_rx`  — main loop → TUI (streaming chunks + final responses)
//!
//! Chat lines are pre-wrapped at exact character boundaries before being handed
//! to ratatui, so each output `Line` occupies exactly one terminal row.  This
//! makes scroll accounting and character-level mouse selection trivial — no
//! ratatui word-wrap logic is involved.

use std::io;
use std::time::Duration;

use arboard::Clipboard;

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode,
        KeyboardEnhancementFlags, KeyModifiers, MouseButton, MouseEventKind,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute, terminal,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Padding, Paragraph},
    Terminal,
};
use tokio::sync::mpsc;
use tui_textarea::TextArea;

use crate::tui_git_diff;

// ---------------------------------------------------------------------------
// Public channel types
// ---------------------------------------------------------------------------

/// Updates from the main async loop to the TUI thread.
pub enum AgentUpdate {
    /// A chunk of text from the streaming response.
    StreamChunk(String),
    /// The complete, final response text (replaces any streamed chunks).
    Complete(String),
    /// A tool is currently executing (name of the tool).
    StatusUpdate(String),
    /// A background job finished. Adds a System message + agent placeholder.
    JobNotification(String),
    /// Pre-populate the chat with persisted history on startup.
    /// Each entry is (display_role, content) e.g. ("You", "…") or ("Calcifer", "…").
    HistoryLoaded(Vec<(String, String)>),
    /// Append new turns to the end of the chat (e.g. from Signal exchanges).
    NewTurns(Vec<(String, String)>),
    // ── Sub-agent activity events ─────────────────────────────────────────
    /// A sub-agent was spawned — show its activity box.
    SubAgentStarted { id: u64, task: String },
    /// Streaming text from a sub-agent.
    SubAgentChunk { id: u64, chunk: String },
    /// Tool activity from a sub-agent.
    SubAgentToolEvent { id: u64, text: String },
    /// Sub-agent completed — remove its activity box.
    SubAgentCompleted { id: u64, result_summary: String },
    /// Add a streaming agent placeholder without a "System" message.
    /// Used when the agent will synthesise a result (e.g. sub-agent completion).
    AgentPlaceholder,
}

/// Channel endpoints owned by `TuiSignalClient`.
pub struct TuiChannels {
    /// `receive()` awaits on this to get user-submitted messages.
    pub user_input_rx: mpsc::Receiver<String>,
    /// `send()` and the streaming callback push updates through this.
    pub agent_update_tx: mpsc::Sender<AgentUpdate>,
}

/// Create all channels and return:
/// - `TuiChannels` — for the signal client (async side)
/// - `Sender<String>` — given to the TUI thread so it can deliver user input
/// - `Receiver<AgentUpdate>` — given to the TUI thread to receive agent output
pub fn create_channels() -> (TuiChannels, mpsc::Sender<String>, mpsc::Receiver<AgentUpdate>) {
    let (user_input_tx, user_input_rx) = mpsc::channel::<String>(16);
    let (agent_update_tx, agent_update_rx) = mpsc::channel::<AgentUpdate>(256);
    (
        TuiChannels { user_input_rx, agent_update_tx },
        user_input_tx,
        agent_update_rx,
    )
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct ChatMessage {
    role: String,
    text: String,
    streaming: bool,
    /// System messages start collapsed; toggled by click or keypress.
    collapsed: bool,
}

/// State for a live sub-agent activity box in the TUI.
struct SubAgentBox {
    id: u64,
    task: String,
    /// Rolling buffer of recent output, capped at MAX_SA_BOX_CHARS.
    output: String,
}

const MAX_SA_BOX_CHARS: usize = 2000;
const MAX_VISIBLE_SA_BOXES: usize = 4;

// ── Sage palette (from dotfiles/kitty/kitty.conf) ────────────────────────
const SAGE_GREEN: Color = Color::Rgb(0x8c, 0xb4, 0x96);   // color2  #8cb496
const SAGE_YELLOW: Color = Color::Rgb(0xc8, 0x96, 0x64);   // color3  #c89664
const SAGE_BLUE: Color = Color::Rgb(0x5f, 0x9e, 0xa0);     // color4  #5f9ea0
const SAGE_DIM: Color = Color::Rgb(0x65, 0x65, 0x65);       // color8  #656565
const SAGE_USER_BG: Color = Color::Rgb(0x28, 0x28, 0x28);   // slightly above bg #1e1e1e

// ---------------------------------------------------------------------------
// TUI entry point
// ---------------------------------------------------------------------------

/// Run the TUI. **Blocks** until the user quits (Ctrl+Q), then calls
/// `std::process::exit(0)`. Intended to be called via
/// `tokio::task::spawn_blocking`.
pub fn run_tui(
    agent_name: String,
    user_input_tx: mpsc::Sender<String>,
    mut agent_update_rx: mpsc::Receiver<AgentUpdate>,
) {
    // ── Terminal setup ────────────────────────────────────────────────────────
    terminal::enable_raw_mode().expect("failed to enable raw mode");
    let mut stdout = io::stdout();
    execute!(stdout, crossterm::terminal::EnterAlternateScreen).expect("enter alt screen");
    // Push kitty keyboard protocol so Shift+Enter is a distinct event in
    // supporting terminals (kitty, WezTerm, Ghostty…).  Unsupporting terminals
    // silently ignore the sequence.
    execute!(
        stdout,
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES),
        EnableMouseCapture,
    )
    .ok();

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend).expect("failed to create terminal");

    // ── App state ─────────────────────────────────────────────────────────────
    let mut textarea = make_textarea();
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut scroll_up: u16 = 0; // rows scrolled up from the bottom; 0 = pinned to bottom
    let mut auto_scroll = true;
    let mut h_scroll: usize = 0; // chars scrolled right on nowrap (table/code) lines
    let mut tool_status: Option<String> = None;
    let mut sa_boxes: Vec<SubAgentBox> = Vec::new();
    let mut input_scroll_top: u16 = 0; // tracks textarea viewport top row
    // System clipboard — may be unavailable in headless/Wayland-without-display environments.
    let mut clipboard = Clipboard::new().ok();

    // ── Mouse selection state ──────────────────────────────────────────────────
    // Selection endpoints are (rendered_row, col) pairs in absolute content
    // coordinates, where rendered_row counts pre-wrapped output rows from the
    // top of all content (not the visible viewport).
    let mut sel_start: Option<(u16, u16)> = None;
    let mut sel_end: Option<(u16, u16)> = None;
    let mut is_selecting = false;
    // Cached from the last draw call — used to map mouse coordinates to content rows/cols.
    let mut last_chat_area = Rect::default();
    let mut last_inner_width: u16 = 0;
    let mut last_scroll_top: u16 = 0;
    // Maps display row → message index (for click-to-toggle on system messages).
    let mut last_row_to_msg: Vec<usize> = Vec::new();

    // ── Main loop ─────────────────────────────────────────────────────────────
    loop {
        // Drain all pending agent updates.
        loop {
            match agent_update_rx.try_recv() {
                Ok(AgentUpdate::StatusUpdate(name)) => {
                    tool_status = Some(name);
                }
                Ok(AgentUpdate::StreamChunk(chunk)) => {
                    tool_status = None;
                    if let Some(last) = messages.last_mut() {
                        if last.streaming {
                            last.text.push_str(&chunk);
                        }
                    }
                    if auto_scroll {
                        scroll_up = 0;
                    }
                }
                Ok(AgentUpdate::Complete(text)) => {
                    tool_status = None;
                    if let Some(last) = messages.last_mut() {
                        if last.streaming {
                            // Only replace streaming content when final text is non-empty.
                            // Otherwise keep whatever was accumulated (e.g. tool event lines).
                            if !text.trim().is_empty() {
                                last.text = text;
                            }
                            last.streaming = false;
                        }
                    }
                    // Remove trailing agent message if there is nothing to display.
                    if messages.last().map_or(false, |m| !m.streaming && m.text.trim().is_empty()) {
                        messages.pop();
                    }
                    if auto_scroll {
                        scroll_up = 0;
                    }
                }
                Ok(AgentUpdate::AgentPlaceholder) => {
                    // Streaming placeholder only — no System message.
                    messages.push(ChatMessage {
                        role: agent_name.clone(),
                        text: String::new(),
                        streaming: true,
                        collapsed: false,
                    });
                    auto_scroll = true;
                    scroll_up = 0;
                }
                Ok(AgentUpdate::JobNotification(text)) => {
                    // Show the job completion as a System message, then add an
                    // agent streaming placeholder for the agent's response.
                    messages.push(ChatMessage {
                        role: "System".to_string(),
                        text,
                        streaming: false,
                        collapsed: true,
                    });
                    messages.push(ChatMessage {
                        role: agent_name.clone(),
                        text: String::new(),
                        streaming: true,
                        collapsed: false,
                    });
                    auto_scroll = true;
                    scroll_up = 0;
                }
                Ok(AgentUpdate::HistoryLoaded(turns)) => {
                    // Prepend persisted history so the user sees the full
                    // conversation on startup, including Signal exchanges.
                    let mut history_msgs: Vec<ChatMessage> = turns
                        .into_iter()
                        .map(|(role, content)| {
                            let is_sys = role == "System";
                            ChatMessage {
                                role,
                                text: content,
                                streaming: false,
                                collapsed: is_sys,
                            }
                        })
                        .collect();
                    history_msgs.append(&mut messages);
                    messages = history_msgs;
                    // Stay pinned to bottom so the most recent message is visible.
                    scroll_up = 0;
                }
                Ok(AgentUpdate::NewTurns(turns)) => {
                    // Append new turns (e.g. from Signal) to the end of the chat.
                    for (role, content) in turns {
                        let is_sys = role == "System";
                        messages.push(ChatMessage {
                            role,
                            text: content,
                            streaming: false,
                            collapsed: is_sys,
                        });
                    }
                    if auto_scroll {
                        scroll_up = 0;
                    }
                }
                // ── Sub-agent activity events ──────────────────────────────
                Ok(AgentUpdate::SubAgentStarted { id, task }) => {
                    sa_boxes.push(SubAgentBox {
                        id,
                        task,
                        output: String::new(),
                    });
                }
                Ok(AgentUpdate::SubAgentChunk { id, chunk }) => {
                    if let Some(b) = sa_boxes.iter_mut().find(|b| b.id == id) {
                        b.output.push_str(&chunk);
                        truncate_sa_output(&mut b.output);
                    }
                }
                Ok(AgentUpdate::SubAgentToolEvent { id, text }) => {
                    if let Some(b) = sa_boxes.iter_mut().find(|b| b.id == id) {
                        if !b.output.ends_with('\n') && !b.output.is_empty() {
                            b.output.push('\n');
                        }
                        b.output.push_str(&text);
                        truncate_sa_output(&mut b.output);
                    }
                }
                Ok(AgentUpdate::SubAgentCompleted { id, result_summary: _ }) => {
                    sa_boxes.retain(|b| b.id != id);
                }
                Err(_) => break,
            }
        }

        // ── Input events ─────────────────────────────────────────────────────
        // Process events *before* drawing so the frame reflects the latest
        // state (e.g. deleted newlines shrink the input area immediately).
        let waiting_for_events = messages.last().map(|m| m.streaming).unwrap_or(false);
        if event::poll(Duration::from_millis(16)).unwrap_or(false) {
            match event::read() {
                Ok(Event::Key(key)) => {
                    use crossterm::event::KeyEventKind;
                    if key.kind != KeyEventKind::Release {
                        match key.code {
                            KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                cleanup(&mut terminal);
                                std::process::exit(0);
                            }
                            KeyCode::PageUp => {
                                auto_scroll = false;
                                scroll_up = scroll_up.saturating_add(10);
                            }
                            KeyCode::PageDown => {
                                scroll_up = scroll_up.saturating_sub(10);
                                if scroll_up == 0 {
                                    auto_scroll = true;
                                    h_scroll = 0;
                                }
                            }
                            KeyCode::Left
                                if key.modifiers.contains(KeyModifiers::SHIFT)
                                    && (waiting_for_events || scroll_up > 0 || h_scroll > 0) =>
                            {
                                h_scroll = h_scroll.saturating_sub(8);
                            }
                            KeyCode::Right
                                if key.modifiers.contains(KeyModifiers::SHIFT)
                                    && (waiting_for_events || scroll_up > 0 || h_scroll > 0) =>
                            {
                                h_scroll += 8;
                            }
                            KeyCode::Enter
                                if !key.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                                    && !waiting_for_events =>
                            {
                                let text = textarea.lines().join("\n");
                                let text = text.trim().to_string();
                                if !text.is_empty() {
                                    messages.push(ChatMessage {
                                        role: "You".to_string(),
                                        text: text.clone(),
                                        streaming: false,
                                        collapsed: false,
                                    });
                                    messages.push(ChatMessage {
                                        role: agent_name.clone(),
                                        text: String::new(),
                                        streaming: true,
                                        collapsed: false,
                                    });
                                    user_input_tx.blocking_send(text).ok();
                                    textarea = make_textarea();
                                    input_scroll_top = 0;
                                    auto_scroll = true;
                                    scroll_up = 0;
                                    h_scroll = 0;
                                }
                            }
                            KeyCode::Char('c')
                                if key.modifiers.contains(KeyModifiers::CONTROL) && !waiting_for_events =>
                            {
                                textarea.copy();
                                let yanked = textarea.yank_text().to_string();
                                if !yanked.is_empty() {
                                    if let Some(ref mut cb) = clipboard {
                                        let _ = cb.set_text(yanked);
                                    }
                                }
                            }
                            KeyCode::Char('x')
                                if key.modifiers.contains(KeyModifiers::CONTROL) && !waiting_for_events =>
                            {
                                textarea.cut();
                                let yanked = textarea.yank_text().to_string();
                                if !yanked.is_empty() {
                                    if let Some(ref mut cb) = clipboard {
                                        let _ = cb.set_text(yanked);
                                    }
                                }
                            }
                            KeyCode::Char('v')
                                if key.modifiers.contains(KeyModifiers::CONTROL) && !waiting_for_events =>
                            {
                                if let Some(ref mut cb) = clipboard {
                                    if let Ok(text) = cb.get_text() {
                                        textarea.set_yank_text(text);
                                        textarea.paste();
                                    }
                                }
                            }
                            _ if !waiting_for_events => {
                                textarea.input(key);
                            }
                            _ => {}
                        }
                    }
                }
                Ok(Event::Mouse(mouse)) => match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        auto_scroll = false;
                        scroll_up = scroll_up.saturating_add(3);
                    }
                    MouseEventKind::ScrollDown => {
                        scroll_up = scroll_up.saturating_sub(3);
                        if scroll_up == 0 {
                            auto_scroll = true;
                        }
                    }
                    MouseEventKind::ScrollLeft => {
                        h_scroll = h_scroll.saturating_sub(3);
                    }
                    MouseEventKind::ScrollRight => {
                        h_scroll += 3;
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        if in_chat_content(mouse.column, mouse.row, last_chat_area) {
                            let row = content_row(mouse.row, last_chat_area, last_scroll_top);
                            let col = content_col(mouse.column, last_chat_area);
                            let row_idx = row as usize;
                            let mut toggled_sys = false;
                            if let Some(&msg_idx) = last_row_to_msg.get(row_idx) {
                                if messages[msg_idx].role == "System" {
                                    messages[msg_idx].collapsed = !messages[msg_idx].collapsed;
                                    sel_start = None;
                                    sel_end = None;
                                    is_selecting = false;
                                    toggled_sys = true;
                                }
                            }
                            if !toggled_sys {
                                sel_start = Some((row, col));
                                sel_end = Some((row, col));
                                is_selecting = true;
                            }
                        } else {
                            sel_start = None;
                            sel_end = None;
                            is_selecting = false;
                        }
                    }
                    MouseEventKind::Drag(MouseButton::Left) if is_selecting => {
                        if last_chat_area.height > 2 {
                            let top_content = last_chat_area.y + 1;
                            let bot_content = last_chat_area.y + last_chat_area.height - 2;
                            let left_content = last_chat_area.x + 1;
                            let right_content =
                                last_chat_area.x + last_chat_area.width.saturating_sub(2);
                            let clamped_row = mouse.row.clamp(top_content, bot_content);
                            let row = content_row(clamped_row, last_chat_area, last_scroll_top);
                            let col = content_col(mouse.column, last_chat_area);
                            sel_end = Some((row, col));
                            const SCROLLOFF: u16 = 2;
                            if mouse.row <= top_content + SCROLLOFF {
                                auto_scroll = false;
                                scroll_up = scroll_up.saturating_add(1);
                            } else if mouse.row >= bot_content.saturating_sub(SCROLLOFF) {
                                scroll_up = scroll_up.saturating_sub(1);
                                if scroll_up == 0 {
                                    auto_scroll = true;
                                }
                            }
                            const H_SCROLLOFF: u16 = 3;
                            if mouse.column <= left_content + H_SCROLLOFF {
                                h_scroll = h_scroll.saturating_sub(2);
                            } else if mouse.column >= right_content.saturating_sub(H_SCROLLOFF) {
                                h_scroll += 2;
                            }
                        }
                    }
                    MouseEventKind::Up(MouseButton::Left) if is_selecting => {
                        is_selecting = false;
                        if let (Some(s), Some(e)) = (sel_start, sel_end) {
                            let (s, e) = if s <= e { (s, e) } else { (e, s) };
                            if s != e {
                                let (tagged, _) = build_chat_lines(&messages);
                                let (lines, soft_wraps, full_lines, is_nowrap_row) =
                                    build_display_lines(&tagged, last_inner_width);
                                let _ = lines;
                                let text = extract_selected_text(
                                    &full_lines, &soft_wraps, &is_nowrap_row, s, e,
                                );
                                if !text.is_empty() {
                                    if let Some(ref mut cb) = clipboard {
                                        let _ = cb.set_text(text);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                },
                Ok(Event::Resize(_, _)) => {}
                _ => {}
            }
        }

        // Clamp the textarea's internal viewport so it doesn't leave blank
        // space at the bottom when lines are deleted.  textarea.scroll()
        // adjusts the viewport *and* moves the cursor, so we save/restore it.
        {
            let total_lines = textarea.lines().len() as u16;
            let (cur_row, cur_col) = textarea.cursor();
            // Scroll to the very top — this resets the viewport's prev_top to
            // 0 (via next_scroll_top seeing cursor < prev_top).  Then Jump
            // restores the cursor; the next render will recompute prev_top
            // correctly from 0, clamping to show no blank trailing rows.
            if total_lines > 0 {
                let big = -(total_lines as i16);
                textarea.scroll((big, 0i16));
                textarea.move_cursor(tui_textarea::CursorMove::Jump(cur_row as u16, cur_col as u16));
            }
        }

        // Recompute after event processing so the draw reflects current state.
        let waiting = messages.last().map(|m| m.streaming).unwrap_or(false);

        // Update textarea decoration to reflect waiting/tool state.
        let sa_count = sa_boxes.len();
        let input_title = match &tool_status {
            Some(name) if name == "spawn_subagent" => " sub-agent (locked) ".to_string(),
            Some(name) => format!(" {} ", name),
            None if waiting && sa_count > 0 => format!(" Waiting… ({sa_count} sub-agents) "),
            None if waiting => " Waiting… ".to_string(),
            None if sa_count > 0 => format!(" {sa_count} sub-agents "),
            None => String::new(),
        };
        let input_style = if waiting {
            Style::default().fg(SAGE_DIM)
        } else {
            Style::default()
        };
        textarea.set_block(Block::default().borders(Borders::TOP | Borders::BOTTOM).padding(Padding::new(3, 1, 0, 0)).title(input_title.clone()));
        textarea.set_style(input_style);

        // ── Draw ─────────────────────────────────────────────────────────────
        terminal
            .draw(|frame| {
                let area = frame.area();

                // Input box height: content lines + 2 border rows, clamped.
                let content_rows = textarea.lines().len() as u16;
                let input_height = (content_rows + 2).max(3).min(area.height / 3).max(3);

                // Sub-agent panel height: only visible when sub-agents are active.
                let sa_panel_height = if sa_boxes.is_empty() {
                    0u16
                } else {
                    // Each box gets ≥4 rows; cap total panel at 1/3 of screen.
                    let visible_boxes = sa_boxes.len().min(MAX_VISIBLE_SA_BOXES);
                    let per_box = 6u16;
                    let raw = (visible_boxes as u16 * per_box).max(per_box);
                    raw.min(area.height / 3).max(4)
                };

                let constraints = if sa_panel_height > 0 {
                    vec![
                        Constraint::Min(3),
                        Constraint::Length(sa_panel_height),
                        Constraint::Length(input_height),
                        Constraint::Length(1), // status bar
                    ]
                } else {
                    vec![
                        Constraint::Min(3),
                        Constraint::Length(input_height),
                        Constraint::Length(1), // status bar
                    ]
                };

                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(constraints)
                    .split(area);

                let (chat_area, sa_area, input_area, status_area) = if sa_panel_height > 0 {
                    (chunks[0], Some(chunks[1]), chunks[2], chunks[3])
                } else {
                    (chunks[0], None, chunks[1], chunks[2])
                };

                let inner_width = chat_area.width.saturating_sub(1); // right padding only
                let (tagged, tagged_msg_idx) = build_chat_lines(&messages);
                let (lines, _, _, _) = build_display_lines(&tagged, inner_width);

                // Build display-row → message-index mapping for click handling.
                // Count how many display rows each tagged line expanded to.
                {
                    let mut row_to_msg = Vec::with_capacity(lines.len());
                    let mut display_idx = 0;
                    for (tagged_idx, (line, nowrap)) in tagged.iter().enumerate() {
                        let msg_idx = tagged_msg_idx.get(tagged_idx).copied().unwrap_or(0);
                        if *nowrap || inner_width == 0 {
                            // 1:1 mapping
                            row_to_msg.push(msg_idx);
                            display_idx += 1;
                        } else {
                            let chars = line_to_chars(line);
                            let n_rows = if chars.is_empty() {
                                1
                            } else {
                                // Count rows the same way build_display_lines does.
                                let w = inner_width as usize;
                                let indent_len = chars.iter().take_while(|(c, _)| *c == ' ').count();
                                let mut s = 0;
                                let mut rows = 0;
                                let mut first = true;
                                while s < chars.len() {
                                    let avail = if first { w } else { w.saturating_sub(indent_len).max(1) };
                                    let hard_end = (s + avail).min(chars.len());
                                    let end = if hard_end < chars.len() {
                                        let mut brk = hard_end;
                                        while brk > s && chars[brk - 1].0 != ' ' { brk -= 1; }
                                        if brk == s { hard_end } else { brk }
                                    } else { hard_end };
                                    rows += 1;
                                    s = end;
                                    if s < chars.len() && chars[s].0 == ' ' { s += 1; }
                                    first = false;
                                }
                                rows
                            };
                            for _ in 0..n_rows {
                                row_to_msg.push(msg_idx);
                            }
                            display_idx += n_rows;
                        }
                    }
                    last_row_to_msg = row_to_msg;
                }

                // Pad lines that have a background color to fill the full width,
                // so the background extends across the entire row.
                let lines: Vec<Line<'static>> = lines.into_iter().map(|mut line| {
                    if line.style.bg.is_some() {
                        let content_width: usize = line.spans.iter()
                            .map(|s| s.content.chars().count())
                            .sum();
                        let pad = (inner_width as usize).saturating_sub(content_width);
                        if pad > 0 {
                            line.spans.push(Span::styled(" ".repeat(pad), line.style));
                        }
                    }
                    line
                }).collect();

                let total_rendered_rows = lines.len() as u16;
                let visible = chat_area.height.saturating_sub(2); // T+B borders
                let max_top = total_rendered_rows.saturating_sub(visible);
                let scroll_top = if auto_scroll {
                    max_top
                } else {
                    max_top.saturating_sub(scroll_up)
                };

                // Cache for mouse event handling (read after draw returns).
                last_chat_area = chat_area;
                last_inner_width = inner_width;
                last_scroll_top = scroll_top;

                // Apply character-level selection highlight before rendering.
                let lines = match (sel_start, sel_end) {
                    (Some(s), Some(e)) if s != e => {
                        let (s, e) = if s <= e { (s, e) } else { (e, s) };
                        apply_selection_highlight(lines, s, e)
                    }
                    _ => lines,
                };

                let chat_title = format!(" {} ", agent_name);
                let chat_block = Block::default()
                    .borders(Borders::TOP)
                    .padding(Padding::right(1))
                    .title(chat_title);
                // No .wrap() — lines are already pre-wrapped.
                let chat_para =
                    Paragraph::new(lines).block(chat_block).scroll((scroll_top, h_scroll as u16));
                frame.render_widget(chat_para, chat_area);

                // ── Sub-agent activity panel ──────────────────────────────────
                if let Some(sa_rect) = sa_area {
                    render_subagent_panel(frame, sa_rect, &sa_boxes);
                }

                frame.render_widget(&textarea, input_area);

                // Render persistent "❯" prompt aligned with the first line
                // of input. Hide it when the textarea has scrolled past line 0.
                // Since we reset the viewport to 0 before each render,
                // next_scroll_top(0, cursor, height) simplifies to this:
                let visible_rows = input_area.height.saturating_sub(2); // minus borders
                let (cursor_row, _) = textarea.cursor();
                let cursor_row = cursor_row as u16;
                input_scroll_top = if visible_rows > 0 && cursor_row >= visible_rows {
                    cursor_row + 1 - visible_rows
                } else {
                    0
                };
                if input_scroll_top == 0 && visible_rows > 0 {
                    let prompt_x = input_area.x + 1;
                    let prompt_y = input_area.y + 1; // after top border
                    frame.render_widget(
                        Paragraph::new(Span::styled("❯ ", Style::default().fg(SAGE_DIM))),
                        ratatui::layout::Rect::new(prompt_x, prompt_y, 2, 1),
                    );
                }

                // ── Status bar ────────────────────────────────────────────────
                let status_left = Span::styled(
                    format!(" {}", agent_name),
                    Style::default().fg(SAGE_DIM),
                );
                let status_right = match &tool_status {
                    Some(name) => Span::styled(
                        format!("{} ", name),
                        Style::default().fg(SAGE_YELLOW),
                    ),
                    None if waiting => Span::styled(
                        "thinking… ".to_string(),
                        Style::default().fg(SAGE_DIM),
                    ),
                    None => Span::raw(""),
                };
                let status_bar = Line::from(vec![
                    status_left,
                    Span::raw(" "),
                    status_right,
                ]);
                frame.render_widget(
                    Paragraph::new(status_bar),
                    status_area,
                );
            })
            .ok();

    }
}

// ---------------------------------------------------------------------------
// Sub-agent panel rendering
// ---------------------------------------------------------------------------

/// Render the sub-agent activity panel as a horizontal split of bordered boxes.
fn render_subagent_panel(
    frame: &mut ratatui::Frame,
    area: Rect,
    boxes: &[SubAgentBox],
) {
    if boxes.is_empty() {
        return;
    }

    let visible = &boxes[boxes.len().saturating_sub(MAX_VISIBLE_SA_BOXES)..];
    let n = visible.len();
    let overflow = boxes.len().saturating_sub(MAX_VISIBLE_SA_BOXES);

    // Split horizontally: equal width per box.
    let constraints: Vec<Constraint> = (0..n)
        .map(|_| Constraint::Ratio(1, n as u32))
        .collect();
    let box_areas = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    for (i, sa) in visible.iter().enumerate() {
        let title = if overflow > 0 && i == 0 {
            format!(" +{overflow} | ◈ {} ", trunc_str(&sa.task, 30))
        } else {
            format!(" ◈ {} ", trunc_str(&sa.task, 40))
        };

        let inner_height = box_areas[i].height.saturating_sub(2) as usize;
        let inner_width = box_areas[i].width.saturating_sub(2) as usize;

        // Take the last N lines that fit.
        let output_lines: Vec<Line<'static>> = sa.output
            .lines()
            .rev()
            .take(inner_height)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|l| {
                let display: String = l.chars().take(inner_width).collect();
                Line::from(Span::styled(display, Style::default().fg(SAGE_DIM)))
            })
            .collect();

        let block = Block::default()
            .borders(Borders::TOP | Borders::BOTTOM)
            .padding(Padding::horizontal(1))
            .border_style(Style::default().fg(SAGE_YELLOW))
            .title(Span::styled(title, Style::default().fg(SAGE_YELLOW)));

        let para = Paragraph::new(output_lines).block(block);
        frame.render_widget(para, box_areas[i]);
    }
}

/// Trim a sub-agent output buffer to MAX_SA_BOX_CHARS, respecting UTF-8 char
/// boundaries and snapping to the next newline so partial lines are dropped.
fn truncate_sa_output(buf: &mut String) {
    if buf.len() <= MAX_SA_BOX_CHARS {
        return;
    }
    let mut start = buf.len() - MAX_SA_BOX_CHARS;
    // Walk forward to the next char boundary.
    while start < buf.len() && !buf.is_char_boundary(start) {
        start += 1;
    }
    // Then find the next newline so we don't start mid-line.
    let trim = match buf[start..].find('\n') {
        Some(pos) => start + pos + 1,
        None => start,
    };
    *buf = buf[trim..].to_string();
}

fn trunc_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_textarea<'a>() -> TextArea<'a> {
    let mut ta = TextArea::default();
    ta.set_block(Block::default().borders(Borders::TOP | Borders::BOTTOM).padding(Padding::new(3, 1, 0, 0)));
    ta.set_cursor_line_style(Style::default()); // no background on cursor row
    ta
}

/// Parse inline markdown spans: `**bold**` and `` `code` ``.
/// Returns styled `Span`s with a 2-space leading indent on the first span.
/// Render inline markdown within a table cell, applying a base style.
fn render_inline_cell(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        // **bold**
        if chars[i] == '*' && i + 1 < n && chars[i + 1] == '*' {
            let mut j = i + 2;
            while j + 1 < n && !(chars[j] == '*' && chars[j + 1] == '*') {
                j += 1;
            }
            if j + 1 < n {
                if !current.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut current), base));
                }
                let word: String = chars[i + 2..j].iter().collect();
                spans.push(Span::styled(word, base.add_modifier(Modifier::BOLD)));
                i = j + 2;
                continue;
            }
        }
        // `inline code`
        if chars[i] == '`' {
            let mut j = i + 1;
            while j < n && chars[j] != '`' {
                j += 1;
            }
            if j < n {
                if !current.is_empty() {
                    spans.push(Span::styled(std::mem::take(&mut current), base));
                }
                let word: String = chars[i + 1..j].iter().collect();
                spans.push(Span::styled(word, base.fg(SAGE_BLUE)));
                i = j + 1;
                continue;
            }
        }
        current.push(chars[i]);
        i += 1;
    }

    if !current.is_empty() {
        spans.push(Span::styled(current, base));
    }
    spans
}

fn render_inline_spans(line: &str) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut current = String::from("  ");
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        // **bold**
        if chars[i] == '*' && i + 1 < n && chars[i + 1] == '*' {
            let mut j = i + 2;
            while j + 1 < n && !(chars[j] == '*' && chars[j + 1] == '*') {
                j += 1;
            }
            if j + 1 < n {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                let text: String = chars[i + 2..j].iter().collect();
                spans.push(Span::styled(text, Style::default().add_modifier(Modifier::BOLD)));
                i = j + 2;
                continue;
            }
        }
        // `inline code`
        if chars[i] == '`' {
            let mut j = i + 1;
            while j < n && chars[j] != '`' {
                j += 1;
            }
            if j < n {
                if !current.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut current)));
                }
                let text: String = chars[i + 1..j].iter().collect();
                spans.push(Span::styled(text, Style::default().fg(SAGE_BLUE)));
                i = j + 1;
                continue;
            }
        }
        current.push(chars[i]);
        i += 1;
    }

    if !current.is_empty() {
        spans.push(Span::raw(current));
    }
    if spans.is_empty() {
        spans.push(Span::raw("  ".to_string()));
    }
    spans
}

/// Render a complete accumulated table block as aligned, pipe-free lines.
/// Header rows (before the `|---|` separator) are bolded.
/// The separator row becomes a dim `─` rule sized to the table width.
fn flush_table(rows: &[String], result: &mut Vec<(Line<'static>, bool)>) {
    // Parse each row into (cells, is_separator)
    let parsed: Vec<(Vec<String>, bool)> = rows
        .iter()
        .map(|row| {
            let inner = row.trim().trim_start_matches('|').trim_end_matches('|');
            let cells: Vec<String> =
                inner.split('|').map(|c| c.trim().to_string()).collect();
            let is_sep = !cells.is_empty()
                && cells.iter().all(|c| {
                    !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':')
                });
            (cells, is_sep)
        })
        .collect();

    let max_cols = parsed
        .iter()
        .filter(|(_, is_sep)| !is_sep)
        .map(|(cells, _)| cells.len())
        .max()
        .unwrap_or(0);
    if max_cols == 0 {
        return;
    }

    // Compute column widths from content
    let mut col_widths: Vec<usize> = vec![0; max_cols];
    for (cells, is_sep) in &parsed {
        if *is_sep {
            continue;
        }
        for (j, cell) in cells.iter().enumerate() {
            if j < max_cols {
                col_widths[j] = col_widths[j].max(cell.len());
            }
        }
    }

    let sep_idx = parsed.iter().position(|(_, is_sep)| *is_sep);
    let rule_width: usize = col_widths.iter().sum::<usize>() + col_widths.len().saturating_sub(1) * 2;

    for (i, (cells, is_sep)) in parsed.iter().enumerate() {
        if *is_sep {
            result.push((
                Line::from(Span::styled(
                    format!("  {}", "─".repeat(rule_width)),
                    Style::default().fg(SAGE_DIM),
                )),
                true,
            ));
            continue;
        }

        let is_header = sep_idx.map(|s| i < s).unwrap_or(i == 0);
        let style = if is_header {
            Style::default().add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let mut spans: Vec<Span<'static>> = Vec::new();
        spans.push(Span::styled("  ", style));
        for (j, &width) in col_widths.iter().enumerate() {
            let cell = cells.get(j).map(|s| s.as_str()).unwrap_or("");
            if j > 0 {
                spans.push(Span::styled("  ", style));
            }
            // Strip markdown markers for width calculation (e.g. **bold** → bold)
            let display_len = cell
                .replace("**", "")
                .replace('`', "")
                .len();
            let cell_spans = render_inline_cell(cell, style);
            spans.extend(cell_spans);
            let pad = width.saturating_sub(display_len);
            if pad > 0 {
                spans.push(Span::styled(" ".repeat(pad), style));
            }
        }

        result.push((Line::from(spans), true));
    }
}

/// Convert a message's text to styled ratatui `Line`s with basic markdown rendering:
/// - Fenced code blocks (```) rendered in yellow
/// - Tables: columns aligned, pipes removed, header bolded, separator as dim rule
/// - ATX headers (`#`, `##`, `###`) rendered in color+bold
/// - Inline `**bold**` and `` `code` `` spans
fn render_markdown(text: &str) -> Vec<(Line<'static>, bool)> {
    let mut result: Vec<(Line<'static>, bool)> = Vec::new();
    let mut in_code_block = false;

    // ── Git diff detection ─────────────────────────────────────────────
    if tui_git_diff::is_git_diff(text) {
        return tui_git_diff::render_git_diff(text);
    }

    let mut table_buf: Vec<String> = Vec::new();

    for raw_line in text.lines() {
        // ── Code fence toggle ──────────────────────────────────────────────
        if raw_line.trim_start().starts_with("```") {
            if !table_buf.is_empty() {
                flush_table(&table_buf, &mut result);
                table_buf.clear();
            }
            in_code_block = !in_code_block;
            let lang = raw_line.trim().trim_start_matches('`').trim();
            let label = if lang.is_empty() {
                "  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄".to_string()
            } else {
                format!("  ┄ {lang} ")
            };
            // Fence markers are short — normal wrap
            result.push((
                Line::from(Span::styled(label, Style::default().fg(SAGE_DIM))),
                false,
            ));
            continue;
        }

        if in_code_block {
            // Code content — no wrap, horizontally scrollable
            result.push((
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(raw_line.to_string(), Style::default().fg(SAGE_YELLOW)),
                ]),
                true,
            ));
            continue;
        }

        // ── Table row accumulation ─────────────────────────────────────────
        if raw_line.trim_start().starts_with('|') {
            table_buf.push(raw_line.to_string());
            continue;
        } else if !table_buf.is_empty() {
            flush_table(&table_buf, &mut result);
            table_buf.clear();
        }

        // ── Tool call / result lines (injected by tool_event_cb) ──────────
        if let Some(rest) = raw_line.strip_prefix("⚙ ") {
            result.push((
                Line::from(Span::styled(
                    format!("  ⚙ {rest}"),
                    Style::default().fg(SAGE_DIM).add_modifier(Modifier::ITALIC),
                )),
                false,
            ));
            continue;
        }
        if let Some(rest) = raw_line.strip_prefix("  ↳ ") {
            result.push((
                Line::from(Span::styled(
                    format!("    ↳ {rest}"),
                    Style::default().fg(SAGE_DIM),
                )),
                false,
            ));
            continue;
        }

        // ── ATX headers ────────────────────────────────────────────────────
        if let Some(rest) = raw_line.strip_prefix("### ") {
            result.push((
                Line::from(Span::styled(
                    format!("  {rest}"),
                    Style::default().fg(SAGE_YELLOW).add_modifier(Modifier::BOLD),
                )),
                false,
            ));
        } else if let Some(rest) = raw_line.strip_prefix("## ") {
            result.push((
                Line::from(Span::styled(
                    format!("  {rest}"),
                    Style::default().fg(SAGE_BLUE).add_modifier(Modifier::BOLD),
                )),
                false,
            ));
        } else if let Some(rest) = raw_line.strip_prefix("# ") {
            result.push((
                Line::from(Span::styled(
                    format!("  {rest}"),
                    Style::default().fg(SAGE_GREEN).add_modifier(Modifier::BOLD),
                )),
                false,
            ));
        } else {
            result.push((Line::from(render_inline_spans(raw_line)), false));
        }
    }

    // Flush any table that ends at end-of-message
    if !table_buf.is_empty() {
        flush_table(&table_buf, &mut result);
    }

    result
}

/// Returns (tagged_lines, line_to_msg_index) where line_to_msg_index[i] is the
/// index into `messages` that produced tagged line i.
fn build_chat_lines(messages: &[ChatMessage]) -> (Vec<(Line<'static>, bool)>, Vec<usize>) {
    let mut lines: Vec<(Line<'static>, bool)> = Vec::new();
    let mut line_msg_idx: Vec<usize> = Vec::new();

    for (msg_idx, msg) in messages.iter().enumerate() {
        let is_user = msg.role == "You";
        let is_system = msg.role == "System";
        let user_bg = Style::default().bg(SAGE_USER_BG);
        let sys_style = Style::default().fg(SAGE_DIM);

        // Circle glyph color for the message-start indicator.
        let glyph_style = if is_user {
            Style::default().fg(SAGE_BLUE)
        } else if is_system {
            sys_style
        } else {
            Style::default().fg(SAGE_GREEN)
        };

        // Collapsed system messages: single line with toggle indicator.
        if is_system && msg.collapsed {
            let preview: String = msg.text.lines().next().unwrap_or("").chars().take(80).collect();
            let line = Line::from(vec![
                Span::styled("● ", glyph_style),
                Span::styled("▶ System: ", sys_style),
                Span::styled(preview, sys_style),
            ]);
            lines.push((line, false));
            line_msg_idx.push(msg_idx);
            // Blank separator.
            lines.push((Line::from(" "), false));
            line_msg_idx.push(msg_idx);
            continue;
        }

        // Role label — always wrappable.
        let role_style = if is_user {
            Style::default().fg(SAGE_BLUE).add_modifier(Modifier::BOLD)
        } else if is_system {
            sys_style
        } else {
            Style::default().fg(SAGE_GREEN).add_modifier(Modifier::BOLD)
        };
        let role_prefix = if is_system && !msg.collapsed { "▼ System:" } else { &format!("{}:", msg.role) };
        let mut role_line = Line::from(vec![
            Span::styled("● ", glyph_style),
            Span::styled(role_prefix.to_string(), role_style),
        ]);
        if is_user { role_line.style = user_bg; }
        lines.push((role_line, false));
        line_msg_idx.push(msg_idx);

        if msg.text.is_empty() && msg.streaming {
            // Thinking indicator.
            let mut thinking_line = Line::from(vec![
                Span::raw(" "),
                Span::raw("  "),
                Span::styled("▋", Style::default().fg(SAGE_YELLOW)),
            ]);
            if is_user { thinking_line.style = user_bg; }
            lines.push((thinking_line, false));
            line_msg_idx.push(msg_idx);
        } else {
            // Trim trailing newlines on completed messages.
            let text_content = if msg.streaming {
                msg.text.as_str()
            } else {
                msg.text.trim_end_matches('\n')
            };
            let mut md_lines = render_markdown(text_content);
            let n = md_lines.len();
            for (i, (line, _)) in md_lines.iter_mut().enumerate() {
                // Prepend 1-char left margin (occupies the glyph column).
                line.spans.insert(0, Span::raw(" "));
                if i + 1 == n && msg.streaming {
                    line.spans.push(Span::styled("▋", Style::default().fg(SAGE_YELLOW)));
                }
                if is_user {
                    line.style = user_bg;
                } else if is_system {
                    for span in &mut line.spans {
                        span.style = span.style.fg(SAGE_DIM);
                    }
                }
            }
            let md_count = md_lines.len();
            lines.extend(md_lines);
            for _ in 0..md_count {
                line_msg_idx.push(msg_idx);
            }
        }

        // Blank separator between messages.
        let mut sep = Line::from(" ");
        if is_user { sep.style = user_bg; }
        lines.push((sep, false));
        line_msg_idx.push(msg_idx);
    }

    (lines, line_msg_idx)
}

/// Flatten a `Line`'s spans into `(char, Style)` pairs.
/// The line's base `style` is merged in as the foundation so that properties
/// like background color set at the Line level (rather than on individual
/// spans) are preserved through the flatten → reconstruct roundtrip.
fn line_to_chars(line: &Line<'_>) -> Vec<(char, Style)> {
    let base = line.style;
    line.spans
        .iter()
        .flat_map(|span| {
            let style = base.patch(span.style);
            span.content.chars().map(move |c| (c, style))
        })
        .collect()
}

/// Convert a slice of `(char, Style)` pairs into `Span`s, merging consecutive
/// chars that share the same style.
fn chars_to_spans(chars: &[(char, Style)]) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    for &(ch, style) in chars {
        if let Some(last) = spans.last_mut() {
            if last.style == style {
                let mut s = last.content.to_string();
                s.push(ch);
                last.content = std::borrow::Cow::Owned(s);
                continue;
            }
        }
        spans.push(Span::styled(ch.to_string(), style));
    }
    spans
}

/// Convert tagged lines to final display lines.
/// - `nowrap=false`: word-wrap at `inner_width` (existing behaviour)
/// - `nowrap=true`:  clip to `[h_scroll, h_scroll+inner_width)` — one display
///   row, horizontally scrollable via `h_scroll`.
///
/// Returns four parallel vecs:
/// - `lines`: what gets rendered (nowrap lines clipped to viewport)
/// - `soft_wraps[i]`: `true` when the break after row `i` is a character-wrap
/// - `full_lines[i]`: unclipped original content (for text extraction)
/// - `is_nowrap_row[i]`: `true` when row `i` comes from a nowrap tagged line
fn build_display_lines(
    tagged: &[(Line<'static>, bool)],
    inner_width: u16,
) -> (Vec<Line<'static>>, Vec<bool>, Vec<Line<'static>>, Vec<bool>) {
    if inner_width == 0 {
        let lines: Vec<_> = tagged.iter().map(|(l, _)| l.clone()).collect();
        let n = lines.len();
        let full_lines = lines.clone();
        let is_nowrap_row = tagged.iter().map(|(_, nw)| *nw).collect();
        return (lines, vec![false; n], full_lines, is_nowrap_row);
    }
    let w = inner_width as usize;
    let mut lines = Vec::new();
    let mut soft_wraps = Vec::new();
    let mut full_lines = Vec::new();
    let mut is_nowrap_row = Vec::new();
    for (line, nowrap) in tagged {
        let base_style = line.style;
        let chars = line_to_chars(line);
        if chars.is_empty() {
            let mut empty = Line::from("");
            empty.style = base_style;
            lines.push(empty.clone());
            soft_wraps.push(false);
            full_lines.push(empty);
            is_nowrap_row.push(*nowrap);
            continue;
        }
        if *nowrap {
            // Emit full-width line — horizontal scrolling is handled
            // uniformly by the Paragraph's .scroll() offset.
            let mut full = Line::from(chars_to_spans(&chars));
            full.style = base_style;
            lines.push(full.clone());
            soft_wraps.push(false);
            full_lines.push(full);
            is_nowrap_row.push(true);
        } else {
            // Detect the leading indent of this line (spaces at the start)
            // so we can replicate it on wrapped continuation lines.
            let indent_len = chars.iter().take_while(|(c, _)| *c == ' ').count();
            let indent_style = chars.first().map(|(_, s)| *s).unwrap_or_default();

            let mut start = 0;
            let mut is_first_chunk = true;
            while start < chars.len() {
                let avail = if is_first_chunk { w } else { w.saturating_sub(indent_len).max(1) };
                let hard_end = (start + avail).min(chars.len());

                // Find word boundary: scan backwards from hard_end for a space.
                let end = if hard_end < chars.len() {
                    let mut brk = hard_end;
                    while brk > start && chars[brk - 1].0 != ' ' {
                        brk -= 1;
                    }
                    // If no space found in this chunk, fall back to character break.
                    if brk == start { hard_end } else { brk }
                } else {
                    hard_end
                };

                if is_first_chunk {
                    let mut chunk = Line::from(chars_to_spans(&chars[start..end]));
                    chunk.style = base_style;
                    lines.push(chunk.clone());
                    soft_wraps.push(end < chars.len());
                    full_lines.push(chunk);
                    is_nowrap_row.push(false);
                    is_first_chunk = false;
                } else {
                    let indent_chars: Vec<(char, Style)> =
                        std::iter::repeat((' ', indent_style)).take(indent_len).collect();
                    let mut combined = indent_chars;
                    combined.extend_from_slice(&chars[start..end]);
                    let mut chunk = Line::from(chars_to_spans(&combined));
                    chunk.style = base_style;
                    lines.push(chunk.clone());
                    soft_wraps.push(end < chars.len());
                    full_lines.push(chunk);
                    is_nowrap_row.push(false);
                }

                // Skip trailing space at the break point so the next line
                // doesn't start with a space.
                start = end;
                if start < chars.len() && chars[start].0 == ' ' {
                    start += 1;
                }
            }
        }
    }
    (lines, soft_wraps, full_lines, is_nowrap_row)
}

/// Returns true when (col, row) is inside the inner content of `chat`
/// (i.e. within the borders).
fn in_chat_content(col: u16, row: u16, chat: Rect) -> bool {
    col >= chat.x + 1
        && col < chat.x + chat.width.saturating_sub(1)
        && row >= chat.y + 1
        && row < chat.y + chat.height.saturating_sub(1)
}

/// Converts a screen row inside the chat pane to an absolute rendered-row
/// index (0 = very top of the rendered content, regardless of scroll).
fn content_row(screen_row: u16, chat: Rect, scroll_top: u16) -> u16 {
    screen_row
        .saturating_sub(chat.y + 1) // relative to content top
        .saturating_add(scroll_top) // offset by how far we've scrolled
}

/// Converts a screen column to a content column (0 = leftmost content char).
fn content_col(screen_col: u16, chat: Rect) -> u16 {
    screen_col.saturating_sub(chat.x + 1)
}

/// Apply a character-level selection highlight to `lines`.
/// `sel_start` and `sel_end` are `(rendered_row, col)` pairs, both inclusive,
/// with `sel_start <= sel_end`.
fn apply_selection_highlight(
    mut lines: Vec<Line<'static>>,
    sel_start: (u16, u16),
    sel_end: (u16, u16),
) -> Vec<Line<'static>> {
    let (start_row, start_col) = sel_start;
    let (end_row, end_col) = sel_end;

    for (i, line) in lines.iter_mut().enumerate() {
        let row = i as u16;
        if row < start_row || row > end_row {
            continue;
        }

        let chars = line_to_chars(line);
        let highlighted: Vec<(char, Style)> = chars
            .into_iter()
            .enumerate()
            .map(|(ci, (ch, style))| {
                let ci = ci as u16;
                let in_sel = if start_row == end_row {
                    ci >= start_col && ci <= end_col
                } else if row == start_row {
                    ci >= start_col
                } else if row == end_row {
                    ci <= end_col
                } else {
                    true // middle row — fully selected
                };
                (ch, if in_sel { style.bg(SAGE_DIM) } else { style })
            })
            .collect();

        *line = Line::from(chars_to_spans(&highlighted));
    }
    lines
}

/// Collect the plain text of the selected character range from display rows.
///
/// - `full_lines`: unclipped content for each display row (nowrap rows have
///   their full width, not the viewport-clipped slice).
/// - `soft_wraps[i]`: `true` → break after row `i` is a character-wrap; skip `\n`.
/// - `is_nowrap_row[i]`: `true` → row comes from a table/code nowrap line.
///   For nowrap rows we include the entire line content so that content
///   scrolled off to the right is not silently dropped.
fn extract_selected_text(
    full_lines: &[Line<'static>],
    soft_wraps: &[bool],
    is_nowrap_row: &[bool],
    sel_start: (u16, u16),
    sel_end: (u16, u16),
) -> String {
    let (start_row, start_col) = sel_start;
    let (end_row, end_col) = sel_end;
    let mut result = String::new();

    for (i, line) in full_lines.iter().enumerate() {
        let row = i as u16;
        if row < start_row || row > end_row {
            continue;
        }

        let chars: Vec<char> = line.spans.iter().flat_map(|s| s.content.chars()).collect();

        if chars.is_empty() {
            if row < end_row && !soft_wraps.get(i).copied().unwrap_or(false) {
                result.push('\n');
            }
            continue;
        }

        let (from, to) = if is_nowrap_row.get(i).copied().unwrap_or(false) {
            // Nowrap (table/code) rows: include full original content so
            // text that extends past the visible viewport is not lost.
            (0, chars.len().saturating_sub(1))
        } else {
            let from = if row == start_row { start_col as usize } else { 0 };
            let to = if row == end_row {
                (end_col as usize).min(chars.len().saturating_sub(1))
            } else {
                chars.len().saturating_sub(1)
            };
            (from, to)
        };

        if from < chars.len() && from <= to {
            result.extend(chars[from..=to].iter());
        }
        if row < end_row && !soft_wraps.get(i).copied().unwrap_or(false) {
            result.push('\n');
        }
    }

    result
}

fn cleanup(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    execute!(
        terminal.backend_mut(),
        PopKeyboardEnhancementFlags,
        DisableMouseCapture,
        crossterm::terminal::LeaveAlternateScreen,
    )
    .ok();
    terminal::disable_raw_mode().ok();
}
