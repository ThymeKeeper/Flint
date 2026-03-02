//! Terminal user interface — scrollable chat history + multi-line text input.
//!
//! The TUI runs on a dedicated blocking thread (via `tokio::task::spawn_blocking`).
//! Two `tokio::sync::mpsc` channels bridge the async main loop and the blocking
//! TUI thread:
//!   • `user_input_tx`    — TUI → main loop (user-submitted text)
//!   • `agent_update_rx`  — main loop → TUI (streaming chunks + final responses)

use std::io;
use std::time::Duration;

use arboard::Clipboard;

use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode,
        KeyboardEnhancementFlags, KeyModifiers, MouseEventKind,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute, terminal,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
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
    let mut scroll_up: u16 = 0; // lines scrolled up from the bottom; 0 = pinned to bottom
    let mut auto_scroll = true;
    // System clipboard — may be unavailable in headless/Wayland-without-display environments.
    let mut clipboard = Clipboard::new().ok();

    // ── Main loop ─────────────────────────────────────────────────────────────
    loop {
        // Drain all pending agent updates.
        loop {
            match agent_update_rx.try_recv() {
                Ok(AgentUpdate::StreamChunk(chunk)) => {
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
                Err(_) => break,
            }
        }

        // True when we are awaiting an agent response.
        let waiting = messages.last().map(|m| m.streaming).unwrap_or(false);

        // Update textarea decoration to reflect waiting state.
        let input_title = if waiting { " Waiting… " } else { " You " };
        let input_style = if waiting {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        textarea.set_block(Block::default().borders(Borders::ALL).title(input_title));
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

                // Build chat lines.
                let lines = build_chat_lines(&messages);
                let total_lines = lines.len() as u16;
                let visible = chat_area.height.saturating_sub(2); // subtract borders

                // Scroll offset from the top of the text.
                let max_top = total_lines.saturating_sub(visible);
                let scroll_top = if auto_scroll {
                    max_top
                } else {
                    max_top.saturating_sub(scroll_up)
                };

                let chat_block = Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} ", agent_name));
                let chat_para = Paragraph::new(ratatui::text::Text::from(lines))
                    .block(chat_block)
                    .wrap(Wrap { trim: false })
                    .scroll((scroll_top, 0));
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

                    // ── Page Up / Down — scroll chat history ──────────────────
                    KeyCode::PageUp => {
                        auto_scroll = false;
                        scroll_up = scroll_up.saturating_add(10);
                    }
                    KeyCode::PageDown => {
                        scroll_up = scroll_up.saturating_sub(10);
                        if scroll_up == 0 {
                            auto_scroll = true;
                        }
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

fn build_chat_lines(messages: &[ChatMessage]) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for msg in messages {
        // Role label.
        let role_style = if msg.role == "You" {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
        };
        lines.push(Line::from(Span::styled(
            format!("{}:", msg.role),
            role_style,
        )));

        if msg.text.is_empty() && msg.streaming {
            // Thinking indicator.
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("▋", Style::default().fg(Color::Yellow)),
            ]));
        } else {
            let text_lines: Vec<String> =
                msg.text.lines().map(|l| l.to_owned()).collect();
            let n = text_lines.len();
            for (i, line) in text_lines.into_iter().enumerate() {
                let is_last = i + 1 == n;
                if is_last && msg.streaming {
                    lines.push(Line::from(vec![
                        Span::raw(format!("  {line}")),
                        Span::styled("▋", Style::default().fg(Color::Yellow)),
                    ]));
                } else {
                    lines.push(Line::from(format!("  {line}")));
                }
            }
        }

        // Blank separator between messages.
        lines.push(Line::from(""));
    }

    lines
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
