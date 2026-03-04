//! Manages the signal-cli subprocess: download, initial device link, and HTTP daemon.
//!
//! signal-cli is a Java-free native binary that exposes the Signal protocol.
//! clawd downloads it on first use and manages the daemon lifecycle.
//!
//! Supported platforms: Linux x86_64 only (the only platform with a native binary).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Child;
use tracing::{info, warn};

/// TCP port the managed signal-cli daemon listens on for JSON-RPC.
pub const TCP_PORT: u16 = 7583;
const SIGNAL_CLI_VERSION: &str = "0.14.0";
const DOWNLOAD_URL: &str = concat!(
    "https://github.com/AsamK/signal-cli/releases/download/v",
    "0.14.0",
    "/signal-cli-0.14.0-Linux-native.tar.gz"
);

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Path to the signal-cli binary inside the clawd data directory.
pub fn bin_path(data_dir: &Path) -> PathBuf {
    data_dir.join("bin").join("signal-cli")
}

/// Path where signal-cli stores its account data.
pub fn signal_data_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("signal-data")
}

/// The TCP address the managed daemon listens on for JSON-RPC.
pub fn daemon_tcp_addr() -> String {
    format!("127.0.0.1:{TCP_PORT}")
}

// ---------------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------------

/// Download and install signal-cli if not already present. Returns the bin path.
///
/// Only works on Linux x86_64 — other platforms bail with a helpful message.
pub fn ensure_signal_cli(data_dir: &Path) -> Result<PathBuf> {
    let bin = bin_path(data_dir);
    if bin.exists() {
        info!("signal-cli already present at {}", bin.display());
        return Ok(bin);
    }

    // Runtime platform check (compile-time cfg is unreliable for cross-compiled binaries).
    if std::env::consts::OS != "linux" || std::env::consts::ARCH != "x86_64" {
        anyhow::bail!(
            "signal-cli auto-download is only supported on Linux x86_64. \
             Install signal-cli {} manually and set [signal] base_url in config.toml.",
            SIGNAL_CLI_VERSION
        );
    }

    let bin_dir = bin.parent().unwrap();
    std::fs::create_dir_all(bin_dir)
        .with_context(|| format!("Failed to create {}", bin_dir.display()))?;

    download_and_extract(&bin)?;

    // Make executable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755))
            .context("Failed to set signal-cli permissions")?;
    }

    info!("signal-cli installed at {}", bin.display());
    Ok(bin)
}

fn download_and_extract(dest: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    use tar::Archive;

    println!("  Downloading signal-cli v{SIGNAL_CLI_VERSION} (~92 MB)…");
    println!("  (this only happens once)");
    println!();

    let response = ureq::get(DOWNLOAD_URL)
        .call()
        .with_context(|| format!("Failed to download signal-cli from {DOWNLOAD_URL}"))?;

    let gz = GzDecoder::new(response.into_reader());
    let mut archive = Archive::new(gz);

    for entry in archive.entries().context("Failed to read signal-cli tarball")? {
        let mut entry = entry.context("Failed to read tarball entry")?;
        let path = entry.path().context("Failed to get entry path")?.into_owned();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name == "signal-cli" {
            entry.unpack(dest).context("Failed to extract signal-cli")?;
            println!("  Extracted to {}", dest.display());
            return Ok(());
        }
    }

    anyhow::bail!("signal-cli binary not found in tarball — the release format may have changed")
}

// ---------------------------------------------------------------------------
// Device link
// ---------------------------------------------------------------------------

/// Run `signal-cli link` interactively (prints QR code in terminal, user scans
/// with their phone), then return the linked phone number.
///
/// The QR code and URI are written directly to the terminal by signal-cli.
pub fn run_link_flow(data_dir: &Path) -> Result<String> {
    let bin = bin_path(data_dir);
    let sig_data = signal_data_dir(data_dir);

    std::fs::create_dir_all(&sig_data)
        .with_context(|| format!("Failed to create {}", sig_data.display()))?;

    println!("  Starting link — scan the QR code with your Signal app:");
    println!("  {}", dim_str("  Devices → Link New Device"));
    println!();

    let status = std::process::Command::new(&bin)
        .args([
            "--config",
            sig_data.to_str().unwrap(),
            "link",
            "-n",
            "clawd",
        ])
        .status()
        .context("Failed to run signal-cli link")?;

    if !status.success() {
        anyhow::bail!(
            "signal-cli link exited with code {}",
            status.code().unwrap_or(-1)
        );
    }

    let phone = read_account_number(&bin, &sig_data)?;
    println!();
    println!("  Linked as: {phone}");
    Ok(phone)
}

fn read_account_number(bin: &Path, sig_data: &Path) -> Result<String> {
    let output = std::process::Command::new(bin)
        .args(["--config", sig_data.to_str().unwrap(), "listAccounts"])
        .output()
        .context("Failed to run signal-cli listAccounts")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // signal-cli output format varies by version:
    //   v0.11-:  "+15551234567 - Phone (linked)"          first token is the number
    //   v0.13+:  "Number: +15551234567"                   number is a value token
    //   JSON:    [{"number":"+15551234567",...}]           number is in a string value
    // Scan every whitespace-separated token from both stdout and stderr for
    // anything that looks like an E.164 phone number.
    let combined = format!("{stdout}{stderr}");
    for token in combined.split_whitespace() {
        // Strip surrounding punctuation (quotes, commas, colons, brackets).
        let token = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '+');
        if token.starts_with('+')
            && token.len() >= 8
            && token[1..].chars().all(|c| c.is_ascii_digit())
        {
            return Ok(token.to_string());
        }
    }

    anyhow::bail!(
        "Could not parse a phone number from signal-cli listAccounts.\n\
         stdout: {stdout}\nstderr: {stderr}"
    )
}

fn dim_str(s: &str) -> String {
    format!("\x1b[2m{s}\x1b[0m")
}

// ---------------------------------------------------------------------------
// Daemon
// ---------------------------------------------------------------------------

/// A running signal-cli TCP daemon. The process is killed when this is dropped.
pub struct SignalDaemon {
    process: Child,
    data_dir: PathBuf,
    phone_number: String,
}

impl Drop for SignalDaemon {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

impl SignalDaemon {
    /// Returns true if the process is still running.
    pub fn is_running(&mut self) -> bool {
        matches!(self.process.try_wait(), Ok(None))
    }

    /// Kill the current process (if alive) and spawn a fresh one.
    pub async fn restart(&mut self) -> Result<()> {
        let _ = self.process.kill();
        let _ = self.process.wait();
        self.process = spawn_process(&self.data_dir, &self.phone_number)?;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        Ok(())
    }
}

fn spawn_process(data_dir: &Path, phone_number: &str) -> Result<Child> {
    let bin = bin_path(data_dir);
    let sig_data = signal_data_dir(data_dir);
    std::process::Command::new(&bin)
        .args([
            "--config",
            sig_data.to_str().unwrap(),
            "-a",
            phone_number,
            "daemon",
            "--tcp",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("Failed to spawn signal-cli daemon")
}

/// Start signal-cli as a TCP JSON-RPC daemon for the given account.
/// Waits briefly for the daemon to bind its port before returning.
pub async fn start_daemon(data_dir: &Path, phone_number: &str) -> Result<SignalDaemon> {
    info!("Starting signal-cli daemon for {phone_number} on TCP port {TCP_PORT}");

    let process = spawn_process(data_dir, phone_number)?;

    // Give the daemon time to bind the TCP port.
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    Ok(SignalDaemon {
        process,
        data_dir: data_dir.to_path_buf(),
        phone_number: phone_number.to_string(),
    })
}

/// Watchdog task: checks every 10 seconds and restarts the daemon if it has exited.
///
/// After `MAX_RESTARTS` consecutive crashes, calls `notify` with a user-visible
/// message and stops trying. A healthy check cycle resets the counter.
///
/// Takes ownership of the daemon; run with `tokio::spawn`.
pub async fn run_watchdog(
    mut daemon: SignalDaemon,
    notify: impl Fn(String) + Send + 'static,
) {
    const MAX_RESTARTS: u32 = 3;
    let mut restarts: u32 = 0;

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;

        if daemon.is_running() {
            restarts = 0;
            continue;
        }

        restarts += 1;
        if restarts > MAX_RESTARTS {
            let msg = format!(
                "Signal: signal-cli has crashed {restarts} times in a row. \
                 Restart clawd to reconnect."
            );
            warn!("{msg}");
            notify(msg);
            return;
        }

        warn!(
            "signal-cli daemon exited; restarting (attempt {restarts}/{MAX_RESTARTS})…"
        );
        match daemon.restart().await {
            Ok(()) => info!("signal-cli daemon restarted successfully"),
            Err(e) => warn!("signal-cli restart failed: {e:#}"),
        }
    }
}
