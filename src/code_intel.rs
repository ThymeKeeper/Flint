//! Tree-sitter based code intelligence for Rust, Python, and SQL.
//!
//! Provides symbol extraction, go-to-definition, find-references, and syntax
//! diagnostics — all compiled into the binary with zero external dependencies.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor, Tree};

// ---------------------------------------------------------------------------
// Language detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    Rust,
    Python,
    Sql,
}

impl Lang {
    /// Detect language from file extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "rs" => Some(Lang::Rust),
            "py" | "pyi" => Some(Lang::Python),
            "sql" => Some(Lang::Sql),
            _ => None,
        }
    }

    fn ts_language(self) -> Language {
        match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::Sql => tree_sitter_sequel::LANGUAGE.into(),
        }
    }

    /// Tree-sitter S-expression query that captures symbol definitions.
    /// Each capture is named `@name` (the identifier) and `@def` (the whole node).
    fn symbols_query(self) -> &'static str {
        match self {
            Lang::Rust => r#"
                (function_item name: (identifier) @name) @def
                (struct_item name: (type_identifier) @name) @def
                (enum_item name: (type_identifier) @name) @def
                (trait_item name: (type_identifier) @name) @def
                (impl_item type: (type_identifier) @name) @def
                (type_item name: (type_identifier) @name) @def
                (const_item name: (identifier) @name) @def
                (static_item name: (identifier) @name) @def
                (mod_item name: (identifier) @name) @def
            "#,
            Lang::Python => r#"
                (function_definition name: (identifier) @name) @def
                (class_definition name: (identifier) @name) @def
            "#,
            Lang::Sql => r#"
                (create_table_statement name: (identifier) @name) @def
                (create_view_statement name: (identifier) @name) @def
                (create_function_statement name: (identifier) @name) @def
            "#,
        }
    }

    fn kind_label(self, node_kind: &str) -> &'static str {
        match self {
            Lang::Rust => match node_kind {
                "function_item" => "fn",
                "struct_item" => "struct",
                "enum_item" => "enum",
                "trait_item" => "trait",
                "impl_item" => "impl",
                "type_item" => "type",
                "const_item" => "const",
                "static_item" => "static",
                "mod_item" => "mod",
                _ => "symbol",
            },
            Lang::Python => match node_kind {
                "function_definition" => "def",
                "class_definition" => "class",
                _ => "symbol",
            },
            Lang::Sql => match node_kind {
                "create_table_statement" => "table",
                "create_view_statement" => "view",
                "create_function_statement" => "function",
                _ => "symbol",
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Symbol / Location types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: String,
    pub file: PathBuf,
    pub line: usize,   // 1-based
    pub column: usize,  // 0-based
    pub end_line: usize,
    /// One-line preview of the definition.
    pub preview: String,
}

#[derive(Debug, Clone)]
pub struct Reference {
    pub file: PathBuf,
    pub line: usize,
    pub column: usize,
    pub preview: String,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub file: PathBuf,
    pub line: usize,
    pub column: usize,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_source(lang: Lang, source: &[u8]) -> Option<Tree> {
    let mut parser = Parser::new();
    parser.set_language(&lang.ts_language()).ok()?;
    parser.parse(source, None)
}

/// Extract syntax errors from a parse tree.
fn collect_errors(tree: &Tree, source: &[u8], file: &Path) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    let mut cursor = tree.walk();
    walk_errors(&mut cursor, source, file, &mut diags);
    diags
}

fn walk_errors(
    cursor: &mut tree_sitter::TreeCursor,
    source: &[u8],
    file: &Path,
    diags: &mut Vec<Diagnostic>,
) {
    loop {
        let node = cursor.node();
        if node.is_error() || node.is_missing() {
            let start = node.start_position();
            let snippet: String = source
                .get(node.byte_range())
                .map(|b| String::from_utf8_lossy(b).chars().take(40).collect())
                .unwrap_or_default();
            let message = if node.is_missing() {
                format!("missing {}", node.kind())
            } else {
                format!("syntax error near `{snippet}`")
            };
            diags.push(Diagnostic {
                file: file.to_path_buf(),
                line: start.row + 1,
                column: start.column,
                message,
            });
        }
        if cursor.goto_first_child() {
            walk_errors(cursor, source, file, diags);
            cursor.goto_parent();
        }
        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

/// Extract symbol definitions from parsed source.
fn extract_symbols(lang: Lang, source: &[u8], tree: &Tree, file: &Path) -> Vec<Symbol> {
    let query = match Query::new(&lang.ts_language(), lang.symbols_query()) {
        Ok(q) => q,
        Err(_) => return Vec::new(),
    };
    let name_idx = match query.capture_index_for_name("name") {
        Some(i) => i,
        None => return Vec::new(),
    };
    let def_idx = query.capture_index_for_name("def");

    let mut cursor = QueryCursor::new();
    let mut symbols = Vec::new();

    let mut matches = cursor.matches(&query, tree.root_node(), source);
    while let Some(m) = matches.next() {
        let name_cap = match m.captures.iter().find(|c| c.index == name_idx) {
            Some(c) => c,
            None => continue,
        };
        let name = name_cap.node.utf8_text(source).unwrap_or("").to_string();
        let def_node = def_idx
            .and_then(|i| m.captures.iter().find(|c| c.index == i))
            .map(|c| c.node)
            .unwrap_or(name_cap.node);

        let start = def_node.start_position();
        let end = def_node.end_position();

        // Build a one-line preview from the source.
        let preview = source_line(source, start.row);

        symbols.push(Symbol {
            name,
            kind: lang.kind_label(def_node.kind()).to_string(),
            file: file.to_path_buf(),
            line: start.row + 1,
            column: start.column,
            end_line: end.row + 1,
            preview,
        });
    }
    symbols
}

/// Get a single source line by 0-based row index.
fn source_line(source: &[u8], row: usize) -> String {
    let text = String::from_utf8_lossy(source);
    text.lines()
        .nth(row)
        .unwrap_or("")
        .trim()
        .chars()
        .take(120)
        .collect()
}

/// Find all occurrences of `identifier` nodes matching `name` in the tree.
fn find_identifier_refs(
    lang: Lang,
    source: &[u8],
    tree: &Tree,
    file: &Path,
    name: &str,
) -> Vec<Reference> {
    let ident_kinds = match lang {
        Lang::Rust => &["identifier", "type_identifier", "field_identifier"][..],
        Lang::Python => &["identifier"][..],
        Lang::Sql => &["identifier"][..],
    };
    let mut refs = Vec::new();
    let mut cursor = tree.walk();
    walk_refs(&mut cursor, source, file, name, ident_kinds, &mut refs);
    refs
}

fn walk_refs(
    cursor: &mut tree_sitter::TreeCursor,
    source: &[u8],
    file: &Path,
    name: &str,
    ident_kinds: &[&str],
    refs: &mut Vec<Reference>,
) {
    loop {
        let node = cursor.node();
        if ident_kinds.contains(&node.kind()) {
            if let Ok(text) = node.utf8_text(source) {
                if text == name {
                    let start = node.start_position();
                    refs.push(Reference {
                        file: file.to_path_buf(),
                        line: start.row + 1,
                        column: start.column,
                        preview: source_line(source, start.row),
                    });
                }
            }
        }
        if cursor.goto_first_child() {
            walk_refs(cursor, source, file, name, ident_kinds, refs);
            cursor.goto_parent();
        }
        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Workspace index
// ---------------------------------------------------------------------------

/// Cached workspace symbol index.  Lazily built on first use, then
/// invalidated when files change.
pub struct CodeIndex {
    /// symbol-name → list of definitions.
    symbols: Mutex<HashMap<PathBuf, Vec<Symbol>>>,
}

impl CodeIndex {
    pub fn new() -> Self {
        Self {
            symbols: Mutex::new(HashMap::new()),
        }
    }

    /// Parse a single file and return its symbols (caches internally).
    fn index_file(&self, path: &Path) -> Vec<Symbol> {
        let lang = match Lang::from_path(path) {
            Some(l) => l,
            None => return Vec::new(),
        };
        let source = match std::fs::read(path) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let tree = match parse_source(lang, &source) {
            Some(t) => t,
            None => return Vec::new(),
        };
        let syms = extract_symbols(lang, &source, &tree, path);
        let mut cache = self.symbols.lock().unwrap();
        cache.insert(path.to_path_buf(), syms.clone());
        syms
    }

    /// Walk a directory and index all supported files.
    fn index_directory(&self, dir: &Path) -> Vec<Symbol> {
        let mut all = Vec::new();
        let walker = match std::fs::read_dir(dir) {
            Ok(w) => w,
            Err(_) => return all,
        };
        for entry in walker.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip hidden dirs and common non-source dirs.
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.')
                        || name == "target"
                        || name == "node_modules"
                        || name == "__pycache__"
                        || name == ".git"
                    {
                        continue;
                    }
                }
                all.extend(self.index_directory(&path));
            } else if Lang::from_path(&path).is_some() {
                all.extend(self.index_file(&path));
            }
        }
        all
    }

    // ----- Public tool implementations -----

    /// List symbols in a file or directory, optionally filtered by query.
    pub fn symbols(&self, path: &Path, query: Option<&str>) -> Vec<Symbol> {
        let syms = if path.is_dir() {
            self.index_directory(path)
        } else {
            self.index_file(path)
        };
        match query {
            Some(q) => {
                let q_lower = q.to_lowercase();
                syms.into_iter()
                    .filter(|s| s.name.to_lowercase().contains(&q_lower))
                    .collect()
            }
            None => syms,
        }
    }

    /// Find the definition of the symbol at the given position.
    /// Looks up the identifier under the cursor, then searches the workspace.
    pub fn goto_definition(
        &self,
        path: &Path,
        line: usize,   // 1-based
        column: usize,  // 0-based
    ) -> Option<Symbol> {
        let lang = Lang::from_path(path)?;
        let source = std::fs::read(path).ok()?;
        let tree = parse_source(lang, &source)?;

        // Find the identifier node at (line, column).
        let point = tree_sitter::Point::new(line.saturating_sub(1), column);
        let node = tree
            .root_node()
            .descendant_for_point_range(point, point)?;
        let name = node.utf8_text(&source).ok()?;

        // Search same file first.
        let file_syms = extract_symbols(lang, &source, &tree, path);
        if let Some(sym) = file_syms.iter().find(|s| s.name == name) {
            return Some(sym.clone());
        }

        // Search workspace (parent directory as project root heuristic).
        let project_root = find_project_root(path);
        let all_syms = self.index_directory(&project_root);
        all_syms.into_iter().find(|s| s.name == name)
    }

    /// Find all references to a symbol across a directory.
    pub fn find_references(
        &self,
        symbol: &str,
        directory: &Path,
    ) -> Vec<Reference> {
        let mut all_refs = Vec::new();
        self.walk_for_refs(directory, symbol, &mut all_refs);
        all_refs
    }

    fn walk_for_refs(&self, dir: &Path, symbol: &str, refs: &mut Vec<Reference>) {
        let walker = match std::fs::read_dir(dir) {
            Ok(w) => w,
            Err(_) => return,
        };
        for entry in walker.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.')
                        || name == "target"
                        || name == "node_modules"
                        || name == "__pycache__"
                    {
                        continue;
                    }
                }
                self.walk_for_refs(&path, symbol, refs);
            } else if let Some(lang) = Lang::from_path(&path) {
                if let Ok(source) = std::fs::read(&path) {
                    if let Some(tree) = parse_source(lang, &source) {
                        refs.extend(find_identifier_refs(
                            lang, &source, &tree, &path, symbol,
                        ));
                    }
                }
            }
        }
    }

    /// Parse a file and return syntax diagnostics.
    /// Uses enhanced language-specific parsers where available, with
    /// tree-sitter as a fallback.
    pub fn diagnostics(&self, path: &Path) -> Vec<Diagnostic> {
        let lang = match Lang::from_path(path) {
            Some(l) => l,
            None => return vec![Diagnostic {
                file: path.to_path_buf(),
                line: 0,
                column: 0,
                message: "unsupported file type".to_string(),
            }],
        };
        match lang {
            Lang::Rust => rust_diagnostics_enhanced(path),
            Lang::Python => python_diagnostics_enhanced(path),
            Lang::Sql => sql_diagnostics_enhanced(path),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk up from a file path to find a project root (directory containing
/// Cargo.toml, pyproject.toml, setup.py, .git, etc.).
fn find_project_root(path: &Path) -> PathBuf {
    let mut dir = if path.is_file() {
        path.parent().unwrap_or(path)
    } else {
        path
    };
    loop {
        for marker in &["Cargo.toml", "pyproject.toml", "setup.py", ".git"] {
            if dir.join(marker).exists() {
                return dir.to_path_buf();
            }
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent,
            _ => return dir.to_path_buf(),
        }
    }
}

// ---------------------------------------------------------------------------
// Enhanced diagnostics
// ---------------------------------------------------------------------------

/// Convert a byte offset into (line, column), both 1-based, by scanning the
/// source text.  Returns (1, 1) if the offset is out of range.
fn offset_to_line_col(src: &str, byte_offset: usize) -> (usize, usize) {
    let safe_offset = byte_offset.min(src.len());
    let before = &src[..safe_offset];
    let line = before.bytes().filter(|&b| b == b'\n').count() + 1;
    let col = before
        .rfind('\n')
        .map(|nl| safe_offset - nl - 1 + 1)
        .unwrap_or(safe_offset + 1);
    (line, col)
}

/// Inner implementation for Rust enhanced diagnostics using `ra_ap_syntax`.
/// Returns `Err` on any I/O or parser-setup failure so the caller can fall back.
fn try_rust_diagnostics(path: &Path) -> Result<Vec<Diagnostic>, Box<dyn std::error::Error>> {
    use ra_ap_syntax::{Edition, SourceFile};

    let src = std::fs::read_to_string(path)?;
    let parsed = SourceFile::parse(&src, Edition::CURRENT);
    let diags = parsed
        .errors()
        .iter()
        .map(|e| {
            // e.range() → TextRange; start offset as u32
            let start: u32 = e.range().start().into();
            let (line, col) = offset_to_line_col(&src, start as usize);
            Diagnostic {
                file: path.to_path_buf(),
                line,
                column: col.saturating_sub(1), // convert to 0-based column
                message: e.to_string(),
            }
        })
        .collect();
    Ok(diags)
}

/// Rust diagnostics using `ra_ap_syntax` (richer than tree-sitter).
/// Falls back to tree-sitter diagnostics on any error.
pub fn rust_diagnostics_enhanced(path: &Path) -> Vec<Diagnostic> {
    match try_rust_diagnostics(path) {
        Ok(diags) => diags,
        Err(_) => {
            let source = std::fs::read(path).unwrap_or_default();
            match parse_source(Lang::Rust, &source) {
                Some(tree) => collect_errors(&tree, &source, path),
                None => vec![],
            }
        }
    }
}

/// Inner implementation for Python enhanced diagnostics using
/// `rustpython-parser`.
fn try_python_diagnostics(path: &Path) -> Result<Vec<Diagnostic>, Box<dyn std::error::Error>> {
    use rustpython_parser::{parse, Mode};

    let src = std::fs::read_to_string(path)?;
    let path_str = path.to_string_lossy();
    match parse(&src, Mode::Module, &path_str) {
        Ok(_) => Ok(vec![]),
        Err(e) => {
            // e.offset is a TextSize (byte offset into the source).
            let byte_off = u32::from(e.offset) as usize;
            let (line, col) = offset_to_line_col(&src, byte_off);
            Ok(vec![Diagnostic {
                file: path.to_path_buf(),
                line,
                column: col.saturating_sub(1),
                message: e.error.to_string(),
            }])
        }
    }
}

/// Python diagnostics using `rustpython-parser`.
/// Falls back to tree-sitter on any error.
pub fn python_diagnostics_enhanced(path: &Path) -> Vec<Diagnostic> {
    match try_python_diagnostics(path) {
        Ok(diags) => diags,
        Err(_) => {
            let source = std::fs::read(path).unwrap_or_default();
            match parse_source(Lang::Python, &source) {
                Some(tree) => collect_errors(&tree, &source, path),
                None => vec![],
            }
        }
    }
}

/// Inner implementation for SQL diagnostics using `sqlparser` (parse errors)
/// and `sqruff-lib` (lint violations).
fn try_sql_diagnostics(path: &Path) -> Result<Vec<Diagnostic>, Box<dyn std::error::Error>> {
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser as SqlParser;
    use sqruff_lib::core::config::FluffConfig;
    use sqruff_lib::core::linter::core::Linter;

    let src = std::fs::read_to_string(path)?;
    let mut diags: Vec<Diagnostic> = Vec::new();

    // --- sqlparser: catch hard parse errors ---
    let dialect = GenericDialect {};
    if let Err(e) = SqlParser::new(&dialect)
        .try_with_sql(&src)
        .and_then(|mut p| p.parse_statements())
    {
        // sqlparser doesn't give a position, so we report line 1.
        diags.push(Diagnostic {
            file: path.to_path_buf(),
            line: 1,
            column: 0,
            message: format!("SQL parse error: {e}"),
        });
        // Still run sqruff below for any additional context.
    }

    // --- sqruff-lib: richer lint violations ---
    let config = FluffConfig::default();
    match Linter::new(config, None, None, true) {
        Ok(linter) => {
            let fname = path.to_string_lossy().into_owned();
            match linter.lint_string(&src, Some(fname), false) {
                Ok(linted) => {
                    for v in linted.violations() {
                        diags.push(Diagnostic {
                            file: path.to_path_buf(),
                            line: v.line_no,
                            column: v.line_pos.saturating_sub(1),
                            message: v.description.clone(),
                        });
                    }
                }
                Err(_) => { /* sqruff lint failed; sqlparser results still usable */ }
            }
        }
        Err(_) => { /* sqruff config failed; sqlparser results still usable */ }
    }

    Ok(diags)
}

/// SQL diagnostics using `sqlparser` (parse errors) + `sqruff-lib` (lint).
/// Falls back to tree-sitter on any error.
pub fn sql_diagnostics_enhanced(path: &Path) -> Vec<Diagnostic> {
    match try_sql_diagnostics(path) {
        Ok(diags) => diags,
        Err(_) => {
            let source = std::fs::read(path).unwrap_or_default();
            match parse_source(Lang::Sql, &source) {
                Some(tree) => collect_errors(&tree, &source, path),
                None => vec![],
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lang_from_path() {
        assert_eq!(Lang::from_path(Path::new("foo.rs")), Some(Lang::Rust));
        assert_eq!(Lang::from_path(Path::new("bar.py")), Some(Lang::Python));
        assert_eq!(Lang::from_path(Path::new("q.sql")), Some(Lang::Sql));
        assert_eq!(Lang::from_path(Path::new("data.json")), None);
    }

    #[test]
    fn test_parse_rust() {
        let src = b"fn hello() { println!(\"hi\"); }";
        let tree = parse_source(Lang::Rust, src).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn test_parse_python() {
        let src = b"def greet(name):\n    print(f'hello {name}')";
        let tree = parse_source(Lang::Python, src).unwrap();
        assert!(!tree.root_node().has_error());
    }

    #[test]
    fn test_extract_rust_symbols() {
        let src = b"pub struct Foo {}\nfn bar() {}\nenum Baz { A, B }";
        let tree = parse_source(Lang::Rust, src).unwrap();
        let syms = extract_symbols(Lang::Rust, src, &tree, Path::new("test.rs"));
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"bar"));
        assert!(names.contains(&"Baz"));
    }

    #[test]
    fn test_extract_python_symbols() {
        let src = b"class MyClass:\n    pass\n\ndef my_func():\n    pass";
        let tree = parse_source(Lang::Python, src).unwrap();
        let syms = extract_symbols(Lang::Python, src, &tree, Path::new("test.py"));
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"MyClass"));
        assert!(names.contains(&"my_func"));
    }

    #[test]
    fn test_syntax_errors() {
        let src = b"fn broken( { }";
        let tree = parse_source(Lang::Rust, src).unwrap();
        let diags = collect_errors(&tree, src, Path::new("bad.rs"));
        assert!(!diags.is_empty());
    }

    #[test]
    fn test_find_refs() {
        let src = b"fn foo() {}\nfn bar() { foo(); }";
        let tree = parse_source(Lang::Rust, src).unwrap();
        let refs = find_identifier_refs(
            Lang::Rust,
            src,
            &tree,
            Path::new("test.rs"),
            "foo",
        );
        // Should find "foo" in the definition and in the call.
        assert!(refs.len() >= 2);
    }

    #[test]
    fn test_offset_to_line_col() {
        let src = "fn main() {\n    let x = 1;\n}";
        // Offset 0 → line 1, col 1
        assert_eq!(offset_to_line_col(src, 0), (1, 1));
        // First char of line 2 (after the '\n' at offset 11)
        assert_eq!(offset_to_line_col(src, 12), (2, 1));
    }

    #[test]
    fn test_rust_enhanced_valid() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "fn hello() {{ }}").unwrap();
        let diags = rust_diagnostics_enhanced(f.path());
        assert!(diags.is_empty(), "valid Rust should have no diagnostics: {:?}", diags);
    }

    #[test]
    fn test_rust_enhanced_invalid() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "fn broken( {{ }}").unwrap();
        let diags = rust_diagnostics_enhanced(f.path());
        assert!(!diags.is_empty(), "broken Rust should produce diagnostics");
    }

    #[test]
    fn test_python_enhanced_valid() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.as_file_mut()
            .write_all(b"def foo():\n    pass\n")
            .unwrap();
        // Rename to .py so Lang::from_path works (not needed here since we
        // call the function directly).
        let diags = python_diagnostics_enhanced(f.path());
        assert!(diags.is_empty(), "valid Python should have no diagnostics: {:?}", diags);
    }

    #[test]
    fn test_python_enhanced_invalid() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.as_file_mut().write_all(b"def foo(\n").unwrap();
        let diags = python_diagnostics_enhanced(f.path());
        assert!(!diags.is_empty(), "broken Python should produce diagnostics");
    }

    #[test]
    fn test_sql_enhanced_valid() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "SELECT 1;").unwrap();
        // Valid SQL — sqlparser should not error; sqruff may or may not warn.
        let diags = sql_diagnostics_enhanced(f.path());
        // Just check it runs without panic.
        let _ = diags;
    }
}
