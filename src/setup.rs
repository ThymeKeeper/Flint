//! First-run interactive setup wizard — cozy RPG edition.
//!
//! Creates `~/.clawd/config.toml` and `~/.clawd/soul.yaml` (or a user-chosen
//! directory) by prompting for the minimum required information.
//!
//! **Caret / backspace fix**: decorative text is always printed via `println!`
//! *before* the `Input` / `Select` call, putting the cursor on a clean new
//! line.  The prompt string passed to dialoguer contains only plain ASCII, so
//! its internal cursor-column arithmetic is never confused by wide Unicode or
//! ANSI escape sequences.

use anyhow::{Context, Result};
use dialoguer::{Confirm, Input, Select};
use std::fmt;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

// ── ANSI helpers (warm palette) ───────────────────────────────────────────────

fn bold(s: &str) -> String  { format!("\x1b[1m{s}\x1b[0m") }
fn dim(s: &str) -> String   { format!("\x1b[2m{s}\x1b[0m") }
fn amber(s: &str) -> String { format!("\x1b[33m{s}\x1b[0m") }
fn sage(s: &str) -> String  { format!("\x1b[32m{s}\x1b[0m") }

// ── Custom dialoguer theme ─────────────────────────────────────────────────────
//
// The built-in themes all append ": " after the prompt string, producing
// "> : value" in the terminal.  CozyTheme omits that colon entirely.

struct CozyTheme;

impl dialoguer::theme::Theme for CozyTheme {
    // Input prompt — just the prompt text, no colon or default display.
    // (We already print the default ourselves above the Input call.)
    fn format_input_prompt(
        &self,
        f: &mut dyn fmt::Write,
        prompt: &str,
        _default: Option<&str>,
    ) -> fmt::Result {
        write!(f, "{prompt}")
    }

    // What the line looks like after the user confirms.
    fn format_input_prompt_selection(
        &self,
        f: &mut dyn fmt::Write,
        prompt: &str,
        sel: &str,
    ) -> fmt::Result {
        write!(f, "{prompt}{}", dim(sel))
    }

    // Confirm prompt — append [Y/n] hint inline.
    fn format_confirm_prompt(
        &self,
        f: &mut dyn fmt::Write,
        prompt: &str,
        default: Option<bool>,
    ) -> fmt::Result {
        let hint = match default {
            Some(true) => " [Y/n]",
            Some(false) => " [y/N]",
            None => " [y/n]",
        };
        write!(f, "{prompt}{hint}")
    }

    fn format_confirm_prompt_selection(
        &self,
        f: &mut dyn fmt::Write,
        prompt: &str,
        sel: Option<bool>,
    ) -> fmt::Result {
        let s = match sel {
            Some(true) => "yes",
            Some(false) => "no",
            None => "—",
        };
        write!(f, "{prompt}{}", dim(s))
    }

    // Select prompt.
    fn format_select_prompt(&self, f: &mut dyn fmt::Write, prompt: &str) -> fmt::Result {
        write!(f, "{prompt}")
    }

    fn format_select_prompt_selection(
        &self,
        f: &mut dyn fmt::Write,
        prompt: &str,
        sel: &str,
    ) -> fmt::Result {
        write!(f, "{prompt}{}", dim(sel))
    }

    fn format_select_prompt_item(
        &self,
        f: &mut dyn fmt::Write,
        text: &str,
        active: bool,
    ) -> fmt::Result {
        if active {
            write!(f, "  {} {}", amber(">"), text)
        } else {
            write!(f, "    {}", dim(text))
        }
    }
}

// ── Typewriter ────────────────────────────────────────────────────────────────

fn typewrite(text: &str, ms_per_char: u64) {
    for ch in text.chars() {
        print!("{ch}");
        std::io::stdout().flush().ok();
        thread::sleep(Duration::from_millis(ms_per_char));
    }
    println!();
}

fn pause(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

// ── Layout helpers ────────────────────────────────────────────────────────────

/// A full-width amber rule line.
fn rule() {
    println!("{}", amber("  ──────────────────────────────────────────────────"));
}

/// Section header — open `┌─` bracket with no right border to close.
/// (A right border would require knowing the exact visible width of the
/// title string after ANSI stripping, which is more trouble than it's worth.)
fn section(title: &str) {
    let bar_len = 48usize.saturating_sub(title.len());
    println!();
    println!("{}", amber(&format!("  ┌─ {title} {}", "─".repeat(bar_len))));
    println!();
}

// ── Prompt helpers ────────────────────────────────────────────────────────────
//
// Print all decorative context text ourselves, then hand off to dialoguer
// with a plain ASCII `"  > "` prompt so cursor math stays trivially correct.

fn prompt_input(title: &str, body: &str, default: &str) -> Result<String> {
    section(title);
    for line in body.lines() {
        println!("  {}", dim(line));
    }
    println!("  {}", dim(&format!("  (default: {default})")));
    println!();

    let val: String = Input::with_theme(&CozyTheme)
        .with_prompt("  > ")
        .default(default.to_string())
        .interact_text()
        .context(format!("Failed to read: {title}"))?;
    Ok(val)
}

fn prompt_secret(title: &str, body: &str) -> Result<String> {
    section(title);
    for line in body.lines() {
        println!("  {}", dim(line));
    }
    println!();

    let val: String = Input::with_theme(&CozyTheme)
        .with_prompt("  > ")
        .allow_empty(true)
        .interact_text()
        .context(format!("Failed to read: {title}"))?;
    Ok(val)
}

fn prompt_select(title: &str, body: &str, items: &[&str]) -> Result<usize> {
    section(title);
    for line in body.lines() {
        println!("  {}", dim(line));
    }
    println!();

    let idx = Select::with_theme(&CozyTheme)
        .with_prompt("  > ")
        .items(items)
        .default(0)
        .interact()
        .context(format!("Failed to select: {title}"))?;
    Ok(idx)
}

fn prompt_confirm(title: &str, body: &str, default: bool) -> Result<bool> {
    section(title);
    for line in body.lines() {
        println!("  {}", dim(line));
    }
    println!();

    let val = Confirm::with_theme(&CozyTheme)
        .with_prompt("  > ")
        .default(default)
        .interact()
        .context(format!("Failed to confirm: {title}"))?;
    Ok(val)
}

// ── Default paths ─────────────────────────────────────────────────────────────

/// Default data directory: `~/.clawd/`
pub fn default_data_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".clawd"))
}

/// Default config path: `~/.clawd/config.toml`
pub fn default_config_path() -> Option<PathBuf> {
    default_data_dir().map(|d| d.join("config.toml"))
}

// ── Startup splash (normal launch) ────────────────────────────────────────────

pub fn print_splash() {
    print!("\x1b[2J\x1b[H");
    std::io::stdout().flush().ok();

    println!();
    rule();
    println!();
    println!("        /\\_/\\");
    println!("       ( · · )   {}", bold("c l a w d"));
    println!("        \\ v /");
    println!("         `~'");
    println!();
    rule();
    println!();
}

// ── Title screen ──────────────────────────────────────────────────────────────

fn print_title() {
    // Clear the terminal before we draw anything.
    print!("\x1b[2J\x1b[H");
    std::io::stdout().flush().ok();

    println!();
    rule();
    println!();
    println!("       {}   {}   {}", amber("♪"), dim("· · ·"), amber("♪"));
    println!();
    typewrite(&dim("       *blink*"), 18);
    pause(500);
    typewrite(&dim("       *blink*"), 18);
    pause(700);
    typewrite(&dim("       ...oh, hello."), 18);
    pause(400);
    println!();
    typewrite(&dim("       I wasn't here a moment ago."), 18);
    typewrite(&dim("       No name, no home, no key — not yet."), 18);
    pause(150);
    typewrite(&dim("       Let's take care of that!"), 18);
    pause(1000);
    println!();
    rule();
    println!();
}

// ── The wizard ────────────────────────────────────────────────────────────────

/// Run the interactive setup wizard. Returns the path to the written config.
pub fn run_setup_wizard() -> Result<PathBuf> {
    print_title();
    pause(250);
    println!("  {}", dim("(press enter to accept the defaults)"));
    pause(100);

    // ── Data directory ───────────────────────────────────────────────────────
    let default_dir = default_data_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "./clawd-data".to_string());

    let data_dir_str = prompt_input(
        "where do I live?",
        "I'll need somewhere to keep things — memories, config, notes.\n\
         Where should I make my home?",
        &default_dir,
    )?;
    let data_dir = PathBuf::from(&data_dir_str);
    std::fs::create_dir_all(&data_dir)
        .with_context(|| format!("Failed to create directory: {}", data_dir.display()))?;

    // ── Agent name ───────────────────────────────────────────────────────────
    let agent_name = prompt_input(
        "what am I called?",
        "I don't have a name yet. What would you like to call me?",
        "Clawd",
    )?;

    // ── Persona ──────────────────────────────────────────────────────────────
    let persona = prompt_input(
        "what kind of companion am I?",
        "A sentence or two is fine. This shapes how I think and speak.\n\
         You can always update it later.",
        "A helpful personal AI assistant with memory.",
    )?;

    // ── Anthropic API key ────────────────────────────────────────────────────
    let anthropic_key: Option<String> = if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        println!();
        println!(
            "  {} {}",
            sage("✓"),
            dim("I can see ANTHROPIC_API_KEY in the environment. Good.")
        );
        None
    } else {
        section("I need a key");
        typewrite(
            &dim("  I can't reach the outside world without an Anthropic API key."),
            14,
        );
        typewrite(
            &dim("  Get one at: console.anthropic.com/settings/keys"),
            14,
        );
        println!();
        println!("  {}", dim("  You can skip this for now and add it to config.toml later."));
        println!();

        let open_browser = prompt_confirm(
            "open that page?",
            "Should I open it in your browser?",
            false,
        )?;
        if open_browser {
            let _ = std::process::Command::new("xdg-open")
                .arg("https://console.anthropic.com/settings/keys")
                .spawn();
            println!();
            println!("  {}", dim("Browser opened."));
        }

        println!();
        println!("  {}", dim("  (the key will be visible as you type)"));
        println!();

        let key = prompt_secret(
            "the key",
            "Paste it below, or leave blank to add it later.",
        )?;
        let trimmed = key.trim().to_string();
        if trimmed.is_empty() {
            println!();
            println!(
                "  {} {}",
                amber("~"),
                dim("No key yet. Add anthropic_api_key to config.toml before we chat."),
            );
            None
        } else {
            Some(trimmed)
        }
    };

    // ── Claude model ─────────────────────────────────────────────────────────
    let model_items = [
        "Sonnet 4.6    balanced, capable      (recommended)",
        "Opus 4.6      thorough, deliberate   (higher cost)",
        "Haiku 4.5     fast, lightweight      (lowest cost)",
    ];
    let model_values = [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
    ];

    let model_idx = prompt_select(
        "how should I think?",
        "Which model should I use to reason and speak?",
        &model_items,
    )?;
    let model = model_values[model_idx];

    // ── Write soul.yaml ──────────────────────────────────────────────────────
    let soul_path = data_dir.join("soul.yaml");
    write_soul_yaml(&soul_path, &agent_name, &persona)?;

    // ── Write config.toml ────────────────────────────────────────────────────
    let db_path = data_dir.join("memories.duckdb");
    let config_path = data_dir.join("config.toml");
    write_config_toml(&config_path, &soul_path, &db_path, anthropic_key.as_deref(), model)?;

    // ── Completion ───────────────────────────────────────────────────────────
    println!();
    pause(300);
    rule();
    println!();
    pause(150);
    typewrite(
        &format!("       Hello. I'm {}.  {}", bold(&agent_name), amber("♪")),
        16,
    );
    println!();
    println!("       {}", dim(&format!("Run {}  to start talking.", bold("clawd"))));
    println!();
    println!("       {}  {}", dim("config"), dim(&config_path.display().to_string()));
    println!("       {}    {}", dim("soul"), dim(&soul_path.display().to_string()));
    println!("       {}  {}", dim("memory"), dim(&db_path.display().to_string()));
    println!();
    rule();
    println!();

    Ok(config_path)
}

// ── File writers ──────────────────────────────────────────────────────────────

fn write_soul_yaml(path: &Path, name: &str, persona: &str) -> Result<()> {
    let content = format!(
        r#"name: "{name}"
persona: |
  {persona}
values:
  - honesty
  - curiosity
  - care
communication_style: "Concise and warm."
proactive_interests:
  - "tech trends"
  - "user patterns"
  - "follow-ups on past conversations"
heartbeat_prompt: |
  Review the recent memories provided below. Consider:
  1. Are there patterns or themes worth noting?
  2. Is there anything the user mentioned that deserves follow-up?

  Return a JSON object (no markdown fences):
  {{"reflection": "...", "proactive_message": null}}

  Only send a proactive message if genuinely helpful. Prefer null.
"#
    );
    std::fs::write(path, content)
        .with_context(|| format!("Failed to write soul.yaml to {}", path.display()))
}

fn write_config_toml(
    path: &Path,
    soul_path: &Path,
    db_path: &Path,
    anthropic_key: Option<&str>,
    model: &str,
) -> Result<()> {
    let soul_str = soul_path.to_string_lossy();
    let db_str = db_path.to_string_lossy();

    let mut out = format!(
        r#"soul_path = "{soul_str}"
db_path   = "{db_str}"
primary_contact = "user"
"#
    );

    if let Some(key) = anthropic_key {
        out.push_str(&format!("anthropic_api_key = \"{key}\"\n"));
    }

    out.push_str(&format!(
        r#"
[claude]
model = "{model}"
max_tokens = 4096
context_limit = 200000
compaction_threshold = 0.75
"#
    ));

    out.push_str(
        r#"
[memory]
max_memories = 10000
top_k_retrieval = 10
importance_decay_days = 30.0
min_importance_to_keep = 0.1
ttl_days_episodic = 90.0

[heartbeat]
interval_secs = 3600
"#,
    );

    std::fs::write(path, &out)
        .with_context(|| format!("Failed to write config.toml to {}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .context("Failed to set config.toml permissions")?;
    }

    Ok(())
}
