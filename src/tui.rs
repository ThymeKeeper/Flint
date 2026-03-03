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
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use tokio::sync::mpsc;
use tui_textarea::TextArea;

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
}

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
                            last.text = text;
                            last.streaming = false;
                        }
                    }
                    if auto_scroll {
                        scroll_up = 0;
                    }
                }
                Ok(AgentUpdate::JobNotification(text)) => {
                    // Show the job completion as a System message, then add an
                    // agent streaming placeholder for the agent's response.
                    messages.push(ChatMessage {
                        role: "System".to_string(),
                        text,
                        streaming: false,
                    });
                    messages.push(ChatMessage {
                        role: agent_name.clone(),
                        text: String::new(),
                        streaming: true,
                    });
                    auto_scroll = true;
                    scroll_up = 0;
                }
                Err(_) => break,
            }
        }

        // True when we are awaiting an agent response.
        let waiting = messages.last().map(|m| m.streaming).unwrap_or(false);

        // Update textarea decoration to reflect waiting/tool state.
        let input_title = match &tool_status {
            Some(name) if name == "spawn_subagent" => " sub-agent (locked) ".to_string(),
            Some(name) => format!(" {} ", name),
            None if waiting => " Waiting… ".to_string(),
            None => " You ".to_string(),
        };
        let input_style = if waiting {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        textarea.set_block(Block::default().borders(Borders::ALL).title(input_title.clone()));
        textarea.set_style(input_style);

        // ── Draw ─────────────────────────────────────────────────────────────
        terminal
            .draw(|frame| {
                let area = frame.area();

                // Input box height: content lines + 2 border rows, clamped.
                let content_rows = textarea.lines().len() as u16;
                let input_height = (content_rows + 2).max(3).min(area.height / 3).max(3);

                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(3), Constraint::Length(input_height)])
                    .split(area);

                let chat_area = chunks[0];
                let input_area = chunks[1];

                let inner_width = chat_area.width.saturating_sub(2); // L+R borders
                let tagged = build_chat_lines(&messages);
                let (lines, _, _, _) = build_display_lines(&tagged, inner_width, h_scroll);

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

                let chat_title = if h_scroll > 0 {
                    format!(" {}  +{h_scroll}→ ", agent_name)
                } else {
                    format!(" {} ", agent_name)
                };
                let chat_block = Block::default()
                    .borders(Borders::ALL)
                    .title(chat_title);
                // No .wrap() — lines are already pre-wrapped.
                let chat_para =
                    Paragraph::new(lines).block(chat_block).scroll((scroll_top, 0));
                frame.render_widget(chat_para, chat_area);
                frame.render_widget(&textarea, input_area);
            })
            .ok();

        // ── Input events (~60 fps) ────────────────────────────────────────────
        if !event::poll(Duration::from_millis(16)).unwrap_or(false) {
            continue;
        }

        match event::read() {
            Ok(Event::Key(key)) => {
                use crossterm::event::KeyEventKind;
                // Only process press and repeat; ignore release events.
                if key.kind == KeyEventKind::Release {
                    continue;
                }

                match key.code {
                    // ── Ctrl+Q — quit ─────────────────────────────────────────
                    KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        cleanup(&mut terminal);
                        std::process::exit(0);
                    }

                    // ── Page Up / Down — vertical scroll ──────────────────────
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

                    // ── Shift+Left / Right — horizontal scroll (tables/code) ───
                    // Active when waiting, reviewing history, or already scrolled.
                    // Falls through to the textarea otherwise (text selection).
                    KeyCode::Left
                        if key.modifiers.contains(KeyModifiers::SHIFT)
                            && (waiting || scroll_up > 0 || h_scroll > 0) =>
                    {
                        h_scroll = h_scroll.saturating_sub(8);
                    }
                    KeyCode::Right
                        if key.modifiers.contains(KeyModifiers::SHIFT)
                            && (waiting || scroll_up > 0 || h_scroll > 0) =>
                    {
                        h_scroll += 8;
                    }

                    // ── Enter (no Shift / Alt) — submit message ───────────────
                    KeyCode::Enter
                        if !key
                            .modifiers
                            .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                            && !waiting =>
                    {
                        let text = textarea.lines().join("\n");
                        let text = text.trim().to_string();
                        if !text.is_empty() {
                            // Add user message to history.
                            messages.push(ChatMessage {
                                role: "You".to_string(),
                                text: text.clone(),
                                streaming: false,
                            });
                            // Placeholder for the agent response.
                            messages.push(ChatMessage {
                                role: agent_name.clone(),
                                text: String::new(),
                                streaming: true,
                            });
                            user_input_tx.blocking_send(text).ok();
                            textarea = make_textarea();
                            auto_scroll = true;
                            scroll_up = 0;
                            h_scroll = 0;
                        }
                    }

                    // ── Ctrl+C — copy selection to system clipboard ───────────
                    KeyCode::Char('c')
                        if key.modifiers.contains(KeyModifiers::CONTROL) && !waiting =>
                    {
                        textarea.copy();
                        let yanked = textarea.yank_text().to_string();
                        if !yanked.is_empty() {
                            if let Some(ref mut cb) = clipboard {
                                let _ = cb.set_text(yanked);
                            }
                        }
                    }

                    // ── Ctrl+X — cut selection to system clipboard ────────────
                    KeyCode::Char('x')
                        if key.modifiers.contains(KeyModifiers::CONTROL) && !waiting =>
                    {
                        textarea.cut();
                        let yanked = textarea.yank_text().to_string();
                        if !yanked.is_empty() {
                            if let Some(ref mut cb) = clipboard {
                                let _ = cb.set_text(yanked);
                            }
                        }
                    }

                    // ── Ctrl+V — paste from system clipboard ──────────────────
                    KeyCode::Char('v')
                        if key.modifiers.contains(KeyModifiers::CONTROL) && !waiting =>
                    {
                        if let Some(ref mut cb) = clipboard {
                            if let Ok(text) = cb.get_text() {
                                textarea.set_yank_text(text);
                                textarea.paste();
                            }
                        }
                    }

                    // ── All other keys → textarea ─────────────────────────────
                    // Includes Shift+Enter (newline) and regular typing.
                    _ if !waiting => {
                        textarea.input(key);
                    }

                    _ => {}
                }
            }

            Ok(Event::Mouse(mouse)) => match mouse.kind {
                // ── Scroll wheel / trackpad ───────────────────────────────────
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

                // ── Left-button drag — character-level text selection ─────────
                MouseEventKind::Down(MouseButton::Left) => {
                    if in_chat_content(mouse.column, mouse.row, last_chat_area) {
                        let row = content_row(mouse.row, last_chat_area, last_scroll_top);
                        let col = content_col(mouse.column, last_chat_area);
                        sel_start = Some((row, col));
                        sel_end = Some((row, col));
                        is_selecting = true;
                    } else {
                        // Click outside chat area clears any active selection.
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
                        // Clamp to the visible content rows and update selection end.
                        let clamped_row = mouse.row.clamp(top_content, bot_content);
                        let row = content_row(clamped_row, last_chat_area, last_scroll_top);
                        let col = content_col(mouse.column, last_chat_area);
                        sel_end = Some((row, col));

                        // Vertical scrolloff: autoscroll when dragging near top / bottom.
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

                        // Horizontal scrolloff: scroll nowrap content when dragging
                        // near the left / right edge of the chat pane.
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
                    // Copy selected text to the system clipboard on release.
                    if let (Some(s), Some(e)) = (sel_start, sel_end) {
                        let (s, e) = if s <= e { (s, e) } else { (e, s) };
                        if s != e {
                            let tagged = build_chat_lines(&messages);
                            let (lines, soft_wraps, full_lines, is_nowrap_row) =
                                build_display_lines(&tagged, last_inner_width, h_scroll);
                            let _ = lines; // display lines not needed here
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

            Ok(Event::Resize(_, _)) => {} // redrawn on next iteration
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_textarea<'a>() -> TextArea<'a> {
    let mut ta = TextArea::default();
    ta.set_block(Block::default().borders(Borders::ALL).title(" You "));
    ta.set_cursor_line_style(Style::default()); // no background on cursor row
    ta
}

/// Parse inline markdown spans: `**bold**` and `` `code` ``.
/// Returns styled `Span`s with a 2-space leading indent on the first span.
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
                spans.push(Span::styled(text, Style::default().fg(Color::Cyan)));
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
                    Style::default().fg(Color::DarkGray),
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

        let mut row_str = String::from("  ");
        for (j, &width) in col_widths.iter().enumerate() {
            let cell = cells.get(j).map(|s| s.as_str()).unwrap_or("");
            if j > 0 {
                row_str.push_str("  ");
            }
            row_str.push_str(&format!("{cell:<width$}"));
        }

        result.push((
            Line::from(Span::styled(row_str.trim_end().to_string(), style)),
            true,
        ));
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
                Line::from(Span::styled(label, Style::default().fg(Color::DarkGray))),
                false,
            ));
            continue;
        }

        if in_code_block {
            // Code content — no wrap, horizontally scrollable
            result.push((
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(raw_line.to_string(), Style::default().fg(Color::Yellow)),
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
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                )),
                false,
            ));
            continue;
        }
        if let Some(rest) = raw_line.strip_prefix("  ↳ ") {
            result.push((
                Line::from(Span::styled(
                    format!("    ↳ {rest}"),
                    Style::default().fg(Color::DarkGray),
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
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                )),
                false,
            ));
        } else if let Some(rest) = raw_line.strip_prefix("## ") {
            result.push((
                Line::from(Span::styled(
                    format!("  {rest}"),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                )),
                false,
            ));
        } else if let Some(rest) = raw_line.strip_prefix("# ") {
            result.push((
                Line::from(Span::styled(
                    format!("  {rest}"),
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
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

fn build_chat_lines(messages: &[ChatMessage]) -> Vec<(Line<'static>, bool)> {
    let mut lines: Vec<(Line<'static>, bool)> = Vec::new();

    for msg in messages {
        let is_user = msg.role == "You";
        let is_system = msg.role == "System";
        let user_bg = Style::default().bg(Color::Rgb(40, 40, 40));

        // Role label — always wrappable.
        let role_style = if is_user {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else if is_system {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
        };
        let mut role_line = Line::from(Span::styled(format!("{}:", msg.role), role_style));
        if is_user { role_line.style = user_bg; }
        lines.push((role_line, false));

        if msg.text.is_empty() && msg.streaming {
            // Thinking indicator.
            let mut thinking_line = Line::from(vec![
                Span::raw("  "),
                Span::styled("▋", Style::default().fg(Color::Yellow)),
            ]);
            if is_user { thinking_line.style = user_bg; }
            lines.push((thinking_line, false));
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
                if i + 1 == n && msg.streaming {
                    line.spans.push(Span::styled("▋", Style::default().fg(Color::Yellow)));
                }
                if is_user { line.style = user_bg; }
            }
            lines.extend(md_lines);
        }

        // Blank separator between messages.
        let mut sep = Line::from("");
        if is_user { sep.style = user_bg; }
        lines.push((sep, false));
    }

    lines
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
    h_scroll: usize,
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
        let chars = line_to_chars(line);
        if chars.is_empty() {
            lines.push(Line::from(""));
            soft_wraps.push(false);
            full_lines.push(Line::from(""));
            is_nowrap_row.push(*nowrap);
            continue;
        }
        if *nowrap {
            let start = h_scroll.min(chars.len());
            let end = (start + w).min(chars.len());
            lines.push(if start < end {
                Line::from(chars_to_spans(&chars[start..end]))
            } else {
                Line::from("")
            });
            soft_wraps.push(false);
            full_lines.push(Line::from(chars_to_spans(&chars)));
            is_nowrap_row.push(true);
        } else {
            let mut start = 0;
            while start < chars.len() {
                let end = (start + w).min(chars.len());
                let chunk = Line::from(chars_to_spans(&chars[start..end]));
                lines.push(chunk.clone());
                soft_wraps.push(end < chars.len());
                full_lines.push(chunk);
                is_nowrap_row.push(false);
                start = end;
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
                (ch, if in_sel { style.bg(Color::DarkGray) } else { style })
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
