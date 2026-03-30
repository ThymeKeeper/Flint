use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

// Git diff colors using the sage palette
const SAGE_GREEN: Color = Color::Rgb(0x8c, 0xb4, 0x96);   // for additions (+)
const SAGE_RED: Color = Color::Rgb(0xa0, 0x5f, 0x5f);     // for deletions (-)
const SAGE_BLUE: Color = Color::Rgb(0x5f, 0x9e, 0xa0);    // for headers (@@ and diff/index)
const SAGE_YELLOW: Color = Color::Rgb(0xc8, 0x96, 0x64);   // for file paths
const SAGE_DIM: Color = Color::Rgb(0x65, 0x65, 0x65);      // for context lines

/// Detect if the text contains git diff content
pub fn is_git_diff(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return false;
    }
    
    // Look for git diff indicators
    for line in &lines {
        if line.starts_with("diff --git") ||
           line.starts_with("index ") ||
           line.starts_with("@@") ||
           (line.starts_with("--- ") && line.contains("/")) ||
           (line.starts_with("+++ ") && line.contains("/")) {
            return true;
        }
    }
    
    // Also check for a high density of +/- lines which indicates a diff
    let diff_lines = lines.iter()
        .filter(|line| line.starts_with('+') || line.starts_with('-'))
        .count();
    
    // If more than 20% of lines are diff lines and we have at least 3, likely a diff
    if lines.len() >= 5 && diff_lines >= 3 && (diff_lines * 5) >= lines.len() {
        return true;
    }
    
    false
}

/// Render git diff content with appropriate syntax highlighting
pub fn render_git_diff(text: &str) -> Vec<(Line<'static>, bool)> {
    let mut result: Vec<(Line<'static>, bool)> = Vec::new();
    
    for raw_line in text.lines() {
        let line = if raw_line.starts_with("diff --git") {
            // Git diff header - blue
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_BLUE).add_modifier(Modifier::BOLD)),
            ])
        } else if raw_line.starts_with("index ") {
            // Index line - blue  
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_BLUE)),
            ])
        } else if raw_line.starts_with("--- ") {
            // Old file marker - yellow
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_YELLOW)),
            ])
        } else if raw_line.starts_with("+++ ") {
            // New file marker - yellow
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_YELLOW)),
            ])
        } else if raw_line.starts_with("@@") {
            // Hunk header - blue with bold
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_BLUE).add_modifier(Modifier::BOLD)),
            ])
        } else if raw_line.starts_with('+') && !raw_line.starts_with("+++") {
            // Addition line - green
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_GREEN)),
            ])
        } else if raw_line.starts_with('-') && !raw_line.starts_with("---") {
            // Deletion line - red
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_RED)),
            ])
        } else {
            // Context line or other - dim
            Line::from(vec![
                Span::raw("  "),
                Span::styled(raw_line.to_string(), Style::default().fg(SAGE_DIM)),
            ])
        };
        
        // All git diff lines should not wrap to preserve formatting
        result.push((line, true)); // true = nowrap
    }
    
    result
}