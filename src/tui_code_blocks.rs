use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

// Sage palette colors for syntax highlighting
const SAGE_GREEN: Color = Color::Rgb(0x8c, 0xb4, 0x96);    // strings, comments
const SAGE_YELLOW: Color = Color::Rgb(0xc8, 0x96, 0x64);   // keywords, numbers
const SAGE_BLUE: Color = Color::Rgb(0x5f, 0x9e, 0xa0);     // types, functions
const SAGE_RED: Color = Color::Rgb(0xa0, 0x5f, 0x5f);      // errors, special
const SAGE_DIM: Color = Color::Rgb(0x65, 0x65, 0x65);      // comments, delimiters
const SAGE_WHITE: Color = Color::Rgb(0xe0, 0xe0, 0xe0);    // default text
const SAGE_CODE_BG: Color = Color::Rgb(0x2a, 0x2a, 0x2a);  // background for code blocks

/// Detect if a line is a code block fence
pub fn is_code_fence(line: &str) -> bool {
    line.trim_start().starts_with("```")
}

/// Extract language from code fence line
pub fn extract_language(fence_line: &str) -> Option<String> {
    let lang = fence_line.trim().trim_start_matches('`').trim();
    if lang.is_empty() {
        None
    } else {
        Some(lang.to_lowercase())
    }
}

/// Apply syntax highlighting to a code line based on language
pub fn highlight_code_line(line: &str, language: Option<&str>) -> Vec<Span<'static>> {
    match language {
        Some("rust") => highlight_rust(line),
        Some("python") | Some("py") => highlight_python(line),
        Some("javascript") | Some("js") => highlight_javascript(line),
        Some("typescript") | Some("ts") => highlight_typescript(line),
        Some("json") => highlight_json(line),
        Some("yaml") | Some("yml") => highlight_yaml(line),
        Some("toml") => highlight_toml(line),
        Some("bash") | Some("sh") | Some("shell") => highlight_shell(line),
        Some("sql") => highlight_sql(line),
        Some("c") => highlight_c(line),
        Some("cpp") | Some("c++") => highlight_cpp(line),
        Some("go") => highlight_go(line),
        Some("java") => highlight_java(line),
        Some("html") => highlight_html(line),
        Some("css") => highlight_css(line),
        Some("xml") => highlight_xml(line),
        Some("markdown") | Some("md") => highlight_markdown(line),
        Some("diff") => highlight_diff(line),
        _ => vec![Span::styled(line.to_string(), Style::default().fg(SAGE_WHITE))],
    }
}

/// Render a code block with proper formatting and syntax highlighting
pub fn render_code_block(lines: &[String], language: Option<&str>) -> Vec<(Line<'static>, bool)> {
    let mut result = Vec::new();
    
    // Opening fence with language label
    let fence_label = match language {
        Some(lang) => format!("  ┌─ {} ", lang),
        None => "  ┌─ code ".to_string(),
    };
    result.push((
        Line::from(Span::styled(fence_label, Style::default().fg(SAGE_DIM))),
        false, // fence can wrap
    ));
    
    // Code content with background and syntax highlighting
    for line in lines {
        let mut spans = vec![Span::styled("  │ ", Style::default().fg(SAGE_DIM))];
        spans.extend(highlight_code_line(line, language));
        
        let code_line = Line::from(spans);
        result.push((code_line, true)); // true = nowrap, preserves formatting
    }
    
    // Closing fence
    result.push((
        Line::from(Span::styled("  └─".to_string(), Style::default().fg(SAGE_DIM))),
        false,
    ));
    
    result
}

/// Rust syntax highlighting
fn highlight_rust(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "fn", "let", "mut", "const", "static", "struct", "enum", "impl", "trait",
        "pub", "use", "mod", "crate", "super", "self", "Self", "match", "if", "else",
        "while", "for", "loop", "break", "continue", "return", "async", "await",
        "move", "ref", "dyn", "where", "unsafe", "extern", "type", "as", "in",
    ];
    
    let types = [
        "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize",
        "f32", "f64", "bool", "char", "str", "String", "Vec", "Option", "Result", "Box",
        "Rc", "Arc", "RefCell", "Cell", "Mutex", "RwLock",
    ];
    
    highlight_generic(line, &keywords, &types, "//")
}

/// Python syntax highlighting
fn highlight_python(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "def", "class", "if", "elif", "else", "for", "while", "try", "except",
        "finally", "with", "as", "import", "from", "return", "yield", "lambda",
        "and", "or", "not", "in", "is", "None", "True", "False", "async", "await",
        "pass", "break", "continue", "global", "nonlocal", "del", "raise", "assert",
    ];
    
    let types = [
        "int", "float", "str", "bool", "list", "dict", "tuple", "set", "frozenset",
        "bytes", "bytearray", "object", "type", "callable",
    ];
    
    highlight_generic(line, &keywords, &types, "#")
}

/// JavaScript syntax highlighting
fn highlight_javascript(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "function", "const", "let", "var", "if", "else", "for", "while", "do",
        "switch", "case", "default", "break", "continue", "return", "try", "catch",
        "finally", "throw", "new", "delete", "typeof", "instanceof", "in", "of",
        "async", "await", "class", "extends", "super", "static", "this", "null",
        "undefined", "true", "false", "import", "export", "from", "as", "default",
    ];
    
    let types = [
        "Array", "Object", "String", "Number", "Boolean", "Symbol", "BigInt",
        "Function", "Promise", "RegExp", "Date", "Error", "Map", "Set", "WeakMap", "WeakSet",
    ];
    
    highlight_generic(line, &keywords, &types, "//")
}

/// TypeScript syntax highlighting
fn highlight_typescript(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "function", "const", "let", "var", "if", "else", "for", "while", "do",
        "switch", "case", "default", "break", "continue", "return", "try", "catch",
        "finally", "throw", "new", "delete", "typeof", "instanceof", "in", "of",
        "async", "await", "class", "extends", "super", "static", "this", "null",
        "undefined", "true", "false", "import", "export", "from", "as", "default",
        "interface", "type", "enum", "namespace", "module", "declare", "abstract",
        "readonly", "private", "protected", "public", "implements",
    ];
    
    let types = [
        "string", "number", "boolean", "object", "symbol", "bigint", "any", "unknown",
        "void", "never", "Array", "Object", "String", "Number", "Boolean", "Symbol",
        "BigInt", "Function", "Promise", "RegExp", "Date", "Error", "Map", "Set",
    ];
    
    highlight_generic(line, &keywords, &types, "//")
}

/// JSON syntax highlighting
fn highlight_json(line: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut escaped = false;
    
    for ch in line.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        
        match ch {
            '"' if !escaped => {
                if in_string {
                    current.push(ch);
                    spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_GREEN)));
                    current.clear();
                    in_string = false;
                } else {
                    if !current.is_empty() {
                        spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_WHITE)));
                        current.clear();
                    }
                    current.push(ch);
                    in_string = true;
                }
            }
            '\\' if in_string => {
                current.push(ch);
                escaped = true;
            }
            '{' | '}' | '[' | ']' | ':' | ',' if !in_string => {
                if !current.trim().is_empty() {
                    let trimmed = current.trim();
                    if trimmed == "true" || trimmed == "false" || trimmed == "null" {
                        spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_YELLOW)));
                    } else if trimmed.parse::<f64>().is_ok() {
                        spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_BLUE)));
                    } else {
                        spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_WHITE)));
                    }
                    current.clear();
                }
                spans.push(Span::styled(ch.to_string(), Style::default().fg(SAGE_DIM)));
            }
            _ => current.push(ch),
        }
    }
    
    if !current.is_empty() {
        let style = if in_string {
            Style::default().fg(SAGE_GREEN)
        } else {
            Style::default().fg(SAGE_WHITE)
        };
        spans.push(Span::styled(current, style));
    }
    
    spans
}

/// Shell script syntax highlighting
fn highlight_shell(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "if", "then", "else", "elif", "fi", "case", "esac", "for", "while", "until",
        "do", "done", "function", "select", "in", "break", "continue", "return",
        "exit", "export", "local", "readonly", "declare", "unset", "shift",
        "echo", "printf", "read", "test", "exec", "eval", "source", "alias", "cd",
        "pwd", "ls", "grep", "sed", "awk", "cut", "sort", "uniq", "wc", "find",
        "xargs", "chmod", "chown", "cp", "mv", "rm", "mkdir", "rmdir", "tar", "gzip",
    ];
    
    highlight_generic(line, &keywords, &[], "#")
}

/// SQL syntax highlighting
fn highlight_sql(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "select", "from", "where", "join", "inner", "left", "right", "outer", "full",
        "on", "as", "group", "by", "having", "order", "limit", "offset", "union",
        "intersect", "except", "insert", "into", "values", "update", "set", "delete",
        "create", "table", "database", "index", "view", "drop", "alter", "add",
        "column", "constraint", "primary", "key", "foreign", "references", "unique",
        "not", "null", "default", "check", "and", "or", "like", "in", "between",
        "exists", "case", "when", "then", "else", "end", "is", "distinct", "all",
        "any", "some", "count", "sum", "avg", "min", "max", "coalesce", "nullif",
    ];
    
    let types = [
        "varchar", "char", "text", "int", "integer", "bigint", "smallint", "tinyint",
        "decimal", "numeric", "float", "real", "double", "boolean", "bool", "bit",
        "date", "time", "datetime", "timestamp", "year", "binary", "varbinary",
        "blob", "json", "xml", "uuid", "serial", "auto_increment",
    ];
    
    highlight_generic(line, &keywords, &types, "--")
}

/// Generic syntax highlighting function
fn highlight_generic(line: &str, keywords: &[&str], types: &[&str], comment_start: &str) -> Vec<Span<'static>> {
    if let Some(comment_pos) = line.find(comment_start) {
        let (code_part, comment_part) = line.split_at(comment_pos);
        let mut spans = highlight_code_part(code_part, keywords, types);
        spans.push(Span::styled(comment_part.to_string(), Style::default().fg(SAGE_DIM)));
        spans
    } else {
        highlight_code_part(line, keywords, types)
    }
}

/// Highlight the code part (non-comment) of a line
fn highlight_code_part(code: &str, keywords: &[&str], types: &[&str]) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut string_char = '"';
    let mut escaped = false;
    
    for ch in code.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        
        if in_string {
            current.push(ch);
            if ch == '\\' {
                escaped = true;
            } else if ch == string_char {
                spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_GREEN)));
                current.clear();
                in_string = false;
            }
        } else if ch == '"' || ch == '\'' {
            finish_current_token(&mut spans, &mut current, keywords, types);
            current.push(ch);
            in_string = true;
            string_char = ch;
        } else if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            finish_current_token(&mut spans, &mut current, keywords, types);
            if !ch.is_whitespace() {
                spans.push(Span::styled(ch.to_string(), Style::default().fg(SAGE_DIM)));
            } else {
                spans.push(Span::raw(ch.to_string()));
            }
        }
    }
    
    if !current.is_empty() {
        if in_string {
            spans.push(Span::styled(current, Style::default().fg(SAGE_GREEN)));
        } else {
            finish_current_token(&mut spans, &mut current, keywords, types);
        }
    }
    
    spans
}

/// Finish processing a token and add it to spans
fn finish_current_token(spans: &mut Vec<Span<'static>>, current: &mut String, keywords: &[&str], types: &[&str]) {
    if current.is_empty() {
        return;
    }
    
    let token = current.trim();
    let style = if keywords.iter().any(|&kw| kw == token) {
        Style::default().fg(SAGE_YELLOW).add_modifier(Modifier::BOLD)
    } else if types.iter().any(|&ty| ty == token) {
        Style::default().fg(SAGE_BLUE)
    } else if token.parse::<f64>().is_ok() {
        Style::default().fg(SAGE_BLUE)
    } else {
        Style::default().fg(SAGE_WHITE)
    };
    
    spans.push(Span::styled(current.clone(), style));
    current.clear();
}

/// Simple implementations for other languages
fn highlight_yaml(line: &str) -> Vec<Span<'static>> {
    if line.trim_start().starts_with('#') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_DIM))]
    } else if line.contains(':') {
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        let mut spans = vec![Span::styled(parts[0].to_string(), Style::default().fg(SAGE_BLUE))];
        if parts.len() > 1 {
            spans.push(Span::styled(":".to_string(), Style::default().fg(SAGE_DIM)));
            spans.push(Span::styled(parts[1].to_string(), Style::default().fg(SAGE_WHITE)));
        }
        spans
    } else {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_WHITE))]
    }
}

fn highlight_toml(line: &str) -> Vec<Span<'static>> {
    if line.trim_start().starts_with('#') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_DIM))]
    } else if line.trim_start().starts_with('[') && line.trim_end().ends_with(']') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_YELLOW).add_modifier(Modifier::BOLD))]
    } else if line.contains('=') {
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        let mut spans = vec![Span::styled(parts[0].to_string(), Style::default().fg(SAGE_BLUE))];
        if parts.len() > 1 {
            spans.push(Span::styled("=".to_string(), Style::default().fg(SAGE_DIM)));
            spans.push(Span::styled(parts[1].to_string(), Style::default().fg(SAGE_GREEN)));
        }
        spans
    } else {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_WHITE))]
    }
}

fn highlight_c(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "inline", "int", "long", "register", "restrict", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
        "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "_Imaginary",
    ];
    
    let types = [
        "char", "short", "int", "long", "float", "double", "void", "size_t", "ssize_t",
        "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
    ];
    
    highlight_generic(line, &keywords, &types, "//")
}

fn highlight_cpp(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
        "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
        "compl", "concept", "const", "constexpr", "const_cast", "continue",
        "decltype", "default", "delete", "do", "double", "dynamic_cast", "else",
        "enum", "explicit", "export", "extern", "false", "float", "for", "friend",
        "goto", "if", "inline", "int", "long", "mutable", "namespace", "new",
        "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
        "private", "protected", "public", "register", "reinterpret_cast",
        "requires", "return", "short", "signed", "sizeof", "static", "static_assert",
        "static_cast", "struct", "switch", "template", "this", "thread_local",
        "throw", "true", "try", "typedef", "typeid", "typename", "union",
        "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while",
    ];
    
    let types = [
        "std::string", "std::vector", "std::map", "std::set", "std::list",
        "std::deque", "std::array", "std::unique_ptr", "std::shared_ptr",
        "std::weak_ptr", "std::function", "std::thread", "std::mutex",
    ];
    
    highlight_generic(line, &keywords, &types, "//")
}

fn highlight_go(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "break", "case", "chan", "const", "continue", "default", "defer", "else",
        "fallthrough", "for", "func", "go", "goto", "if", "import", "interface",
        "map", "package", "range", "return", "select", "struct", "switch", "type",
        "var", "bool", "byte", "complex64", "complex128", "error", "float32",
        "float64", "int", "int8", "int16", "int32", "int64", "rune", "string",
        "uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "true", "false",
        "iota", "nil", "append", "cap", "close", "complex", "copy", "delete",
        "imag", "len", "make", "new", "panic", "print", "println", "real", "recover",
    ];
    
    highlight_generic(line, &keywords, &[], "//")
}

fn highlight_java(line: &str) -> Vec<Span<'static>> {
    let keywords = [
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
        "class", "const", "continue", "default", "do", "double", "else", "enum",
        "extends", "final", "finally", "float", "for", "goto", "if", "implements",
        "import", "instanceof", "int", "interface", "long", "native", "new",
        "package", "private", "protected", "public", "return", "short", "static",
        "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
        "transient", "try", "void", "volatile", "while", "true", "false", "null",
    ];
    
    let types = [
        "String", "Object", "Integer", "Double", "Float", "Boolean", "Character",
        "Byte", "Short", "Long", "BigInteger", "BigDecimal", "ArrayList", "HashMap",
        "HashSet", "LinkedList", "TreeMap", "TreeSet", "Collections", "Arrays",
    ];
    
    highlight_generic(line, &keywords, &types, "//")
}

fn highlight_html(line: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut current = String::new();
    let mut in_tag = false;
    let mut in_string = false;
    let mut string_char = '"';
    
    for ch in line.chars() {
        match ch {
            '<' if !in_string => {
                if !current.is_empty() {
                    spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_WHITE)));
                    current.clear();
                }
                current.push(ch);
                in_tag = true;
            }
            '>' if in_tag && !in_string => {
                current.push(ch);
                spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_BLUE)));
                current.clear();
                in_tag = false;
            }
            '"' | '\'' if in_tag => {
                if in_string && ch == string_char {
                    current.push(ch);
                    spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_GREEN)));
                    current.clear();
                    in_string = false;
                } else if !in_string {
                    if !current.is_empty() {
                        spans.push(Span::styled(current.clone(), Style::default().fg(SAGE_YELLOW)));
                        current.clear();
                    }
                    current.push(ch);
                    in_string = true;
                    string_char = ch;
                } else {
                    current.push(ch);
                }
            }
            _ => current.push(ch),
        }
    }
    
    if !current.is_empty() {
        let style = if in_string {
            Style::default().fg(SAGE_GREEN)
        } else if in_tag {
            Style::default().fg(SAGE_BLUE)
        } else {
            Style::default().fg(SAGE_WHITE)
        };
        spans.push(Span::styled(current, style));
    }
    
    spans
}

fn highlight_css(line: &str) -> Vec<Span<'static>> {
    if line.trim_start().starts_with("/*") || line.trim().starts_with('*') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_DIM))]
    } else if line.contains(':') && line.contains(';') {
        // Property line
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        let mut spans = vec![Span::styled(parts[0].to_string(), Style::default().fg(SAGE_BLUE))];
        if parts.len() > 1 {
            spans.push(Span::styled(":".to_string(), Style::default().fg(SAGE_DIM)));
            let value_part = parts[1].replace(';', "");
            spans.push(Span::styled(value_part, Style::default().fg(SAGE_GREEN)));
            spans.push(Span::styled(";".to_string(), Style::default().fg(SAGE_DIM)));
        }
        spans
    } else if line.contains('{') || line.contains('}') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_YELLOW))]
    } else {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_WHITE))]
    }
}

fn highlight_xml(line: &str) -> Vec<Span<'static>> {
    highlight_html(line) // XML and HTML have similar highlighting patterns
}

fn highlight_diff(line: &str) -> Vec<Span<'static>> {
    if line.starts_with("diff --git") {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_BLUE).add_modifier(Modifier::BOLD))]
    } else if line.starts_with("index ") {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_BLUE))]
    } else if line.starts_with("--- ") || line.starts_with("+++ ") {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_YELLOW))]
    } else if line.starts_with("@@") {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_BLUE).add_modifier(Modifier::BOLD))]
    } else if line.starts_with('+') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_GREEN))]
    } else if line.starts_with('-') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_RED))]
    } else {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_DIM))]
    }
}

fn highlight_markdown(line: &str) -> Vec<Span<'static>> {
    if line.starts_with('#') {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_BLUE).add_modifier(Modifier::BOLD))]
    } else if line.starts_with("```") {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_DIM))]
    } else if line.trim_start().starts_with("- ") || line.trim_start().starts_with("* ") {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_YELLOW))]
    } else {
        vec![Span::styled(line.to_string(), Style::default().fg(SAGE_WHITE))]
    }
}