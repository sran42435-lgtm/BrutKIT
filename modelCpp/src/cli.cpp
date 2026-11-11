#include "cli.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

char lowerChar(char c) {
    return static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))
    );
}

std::string toLower(std::string s) {
    for (char& ch : s) {
        ch = lowerChar(ch);
    }
    return s;
}

bool isSpaceChar(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

bool isAlphaChar(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) != 0;
}

std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && isSpaceChar(s[start])) {
        ++start;
    }

    std::size_t end = s.size();
    while (end > start && isSpaceChar(s[end - 1])) {
        --end;
    }

    return s.substr(start, end - start);
}

FlagSpec makeFlag(
    std::string name,
    FlagType type,
    std::string help,
    std::vector<std::string> aliases = {},
    std::string shortName = {},
    bool required = false
) {
    FlagSpec f;
    f.name = toLower(std::move(name));

    for (std::string& alias : aliases) {
        alias = toLower(alias);
    }

    f.aliases = std::move(aliases);
    f.shortName = toLower(std::move(shortName));
    f.type = type;
    f.required = required;
    f.help = std::move(help);

    return f;
}

CommandSpec makeCommand(
    std::string name,
    std::string help,
    std::vector<FlagSpec> flags,
    bool allowPositional = false,
    int minPositional = 0,
    int maxPositional = 0
) {
    CommandSpec c;
    c.name = toLower(std::move(name));
    c.help = std::move(help);
    c.flags = std::move(flags);
    c.allowPositional = allowPositional;
    c.minPositional = minPositional;
    c.maxPositional = maxPositional;
    return c;
}

CommandSpec makeGlobalSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("help", FlagType::Bool, "Show help", {"?"}, "h"),
        makeFlag("version", FlagType::Bool, "Show version", {}, "V"),

        makeFlag("model-name", FlagType::String, "Create or select model name", {"name"}, "m"),
        makeFlag("config", FlagType::String, "Path to config file", {}, "c"),
        makeFlag("profile", FlagType::String, "Preset profile: tiny, small, nol_s, default", {}, "p"),
        makeFlag("tokenizer", FlagType::String, "Tokenizer path", {}, "t"),

        makeFlag("data", FlagType::String, "Training data path", {"input"}, "d"),
        makeFlag("val-data", FlagType::String, "Validation data path"),

        makeFlag("resume", FlagType::String, "Resume local model by name", {}, "r"),
        makeFlag("now", FlagType::Bool, "Start training immediately after resume"),
        makeFlag("training", FlagType::Bool, "Enter training mode"),

        makeFlag("export", FlagType::String, "Export target model name", {}, "x"),
        makeFlag("from", FlagType::String, "Source model for export"),

        makeFlag("seed", FlagType::Int, "Random seed", {}, "s"),
        makeFlag("yes", FlagType::Bool, "Assume yes for confirmations", {}, "y"),
        makeFlag("overwrite", FlagType::Bool, "Allow overwrite existing model"),
        makeFlag("dry-run", FlagType::Bool, "Do not write files"),
        makeFlag("verbose", FlagType::Bool, "Verbose logs"),
        makeFlag("quiet", FlagType::Bool, "Quiet logs"),
        makeFlag("log-path", FlagType::String, "Log file path"),

        makeFlag("train-tokenizer", FlagType::Bool, "Train tokenizer from data"),
        makeFlag("vocab-size", FlagType::Int, "Tokenizer vocab size"),
        makeFlag("out", FlagType::String, "Generic output path", {}, "o"),

        makeFlag("set", FlagType::KeyVal, "Override config key=value, example: --set model.d_model=256")
    };

    return makeCommand("global", "Global proyek launcher", flags, false, 0, 0);
}

CommandSpec makeTrainingSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("epochs", FlagType::Int, "Number of epochs", {}, "e"),
        makeFlag("steps", FlagType::Int, "Maximum global training steps", {}, "t"),
        makeFlag("batch-size", FlagType::Int, "Training batch size", {}, "b"),
        makeFlag("seq-len", FlagType::Int, "Sequence length", {"sequence-length"}, "q"),

        makeFlag("lr", FlagType::Double, "Learning rate", {"learning-rate"}, "l"),
        makeFlag("min-lr", FlagType::Double, "Minimum learning rate"),
        makeFlag("weight-decay", FlagType::Double, "AdamW weight decay", {}, "w"),
        makeFlag("warmup-ratio", FlagType::Double, "Linear warmup ratio"),
        makeFlag("grad-clip", FlagType::Double, "Gradient clipping norm"),

        makeFlag("save-every-steps", FlagType::Int, "Save checkpoint every N steps"),
        makeFlag("save-every-epochs", FlagType::Int, "Save checkpoint every N epochs"),

        makeFlag("stop-loss-below", FlagType::Double, "Stop training if total loss below this value"),
        makeFlag("stop-acc-above", FlagType::Double, "Stop training if accuracy above this value"),
        makeFlag("early-patience", FlagType::Int, "Early stopping patience in evaluations"),

        makeFlag("data", FlagType::String, "Training data path", {}, "d"),
        makeFlag("val-data", FlagType::String, "Validation data path"),
        makeFlag("seed", FlagType::Int, "Random seed", {}, "s"),

        makeFlag("dry-run", FlagType::Bool, "Plan training without writing checkpoints"),
        makeFlag("persist-config", FlagType::Bool, "Persist overridden training config")
    };

    return makeCommand("training", "Train current model", flags, false, 0, 0);
}

CommandSpec makeResumeSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("epochs", FlagType::Int, "Extend or set epoch target", {}, "e"),
        makeFlag("steps", FlagType::Int, "Extend or set step target", {}, "t"),
        makeFlag("batch-size", FlagType::Int, "Training batch size", {}, "b"),
        makeFlag("seq-len", FlagType::Int, "Sequence length", {"sequence-length"}, "q"),

        makeFlag("lr", FlagType::Double, "Learning rate", {"learning-rate"}, "l"),
        makeFlag("min-lr", FlagType::Double, "Minimum learning rate"),
        makeFlag("grad-clip", FlagType::Double, "Gradient clipping norm"),

        makeFlag("save-every-steps", FlagType::Int, "Save checkpoint every N steps"),
        makeFlag("data", FlagType::String, "Training data path", {}, "d"),
        makeFlag("val-data", FlagType::String, "Validation data path"),
        makeFlag("seed", FlagType::Int, "Random seed", {}, "s"),

        makeFlag("dry-run", FlagType::Bool, "Show resume plan without training"),
        makeFlag("persist-config", FlagType::Bool, "Persist overridden training config")
    };

    return makeCommand("resume", "Resume unfinished training", flags, false, 0, 0);
}

CommandSpec makeSwitchSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("model", FlagType::String, "Local model name to switch into", {"name"}, "m", true)
    };

    return makeCommand("switch", "Switch lobby to another local model", flags, false, 0, 0);
}

CommandSpec makeExitSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("save", FlagType::Bool, "Force save checkpoint before exit"),
        makeFlag("no-save", FlagType::Bool, "Do not save checkpoint before exit")
    };

    return makeCommand("exit", "Exit current model session", flags, false, 0, 0);
}

CommandSpec makePromptSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("max-new-tokens", FlagType::Int, "Maximum generated tokens", {"max-tokens"}, "n"),
        makeFlag("temperature", FlagType::Double, "Sampling temperature", {"temp"}, "t"),
        makeFlag("top-k", FlagType::Int, "Top-k sampling", {}, "k"),
        makeFlag("top-p", FlagType::Double, "Nucleus sampling", {"nucleus"}, "p"),
        makeFlag("min-p", FlagType::Double, "Min-p sampling"),
        makeFlag("repetition-penalty", FlagType::Double, "Repetition penalty"),
        makeFlag("seed", FlagType::Int, "Sampling seed", {}, "s"),

        makeFlag("stream", FlagType::Bool, "Stream generated tokens"),
        makeFlag("echo", FlagType::Bool, "Echo prompt before response"),
        makeFlag("stop", FlagType::String, "Stop string"),
        makeFlag("show-probs", FlagType::Bool, "Show top token probabilities"),
        makeFlag("disable-hyper", FlagType::Bool, "Disable hyperbolic branches for this prompt"),
        makeFlag("topic-lock", FlagType::String, "Lock routing to a topic/expert id")
    };

    return makeCommand(
        "prompt",
        "Prompt current model",
        flags,
        true,
        1,
        -1
    );
}

CommandSpec makeExportSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("as", FlagType::String, "Target exported model name", {"name"}, "a", true),
        makeFlag("from", FlagType::String, "Source model name"),
        makeFlag("dtype", FlagType::String, "Export dtype: f32 or bf16"),
        makeFlag("include-tokenizer", FlagType::Bool, "Include tokenizer in export"),
        makeFlag("include-optimizer", FlagType::Bool, "Include optimizer state in export"),
        makeFlag("strip-ema", FlagType::Bool, "Strip EMA weights if present"),
        makeFlag("card-text", FlagType::String, "Custom model card text"),
        makeFlag("overwrite", FlagType::Bool, "Overwrite existing export target")
    };

    return makeCommand("export", "Export current model bundle", flags, false, 0, 0);
}

CommandSpec makeStatusSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("json", FlagType::Bool, "Print status as JSON")
    };

    return makeCommand("status", "Show current model status", flags, false, 0, 0);
}

CommandSpec makeConfigSpec() {
    std::vector<FlagSpec> flags = {
        makeFlag("show", FlagType::Bool, "Show resolved config"),
        makeFlag("save", FlagType::Bool, "Save resolved config"),
        makeFlag("set", FlagType::KeyVal, "Override config key=value")
    };

    return makeCommand("config", "Inspect or override current config", flags, false, 0, 0);
}

CommandSpec makeHelpSpec() {
    std::vector<FlagSpec> flags = {};

    return makeCommand(
        "help",
        "Show help for a command",
        flags,
        true,
        0,
        1
    );
}

CommandSpec makeModelsSpec() {
    std::vector<FlagSpec> flags = {};

    return makeCommand("models", "List local models", flags, false, 0, 0);
}

const std::vector<CommandSpec>& lobbySpecs() {
    static const std::vector<CommandSpec> specs = [] {
        std::vector<CommandSpec> v;
        v.push_back(makeTrainingSpec());
        v.push_back(makeResumeSpec());
        v.push_back(makeSwitchSpec());
        v.push_back(makeExitSpec());
        v.push_back(makePromptSpec());
        v.push_back(makeExportSpec());
        v.push_back(makeStatusSpec());
        v.push_back(makeConfigSpec());
        v.push_back(makeHelpSpec());
        v.push_back(makeModelsSpec());
        return v;
    }();

    return specs;
}

const FlagSpec* findLongSpec(
    const CommandSpec& spec,
    const std::string& name
) {
    std::string n = toLower(name);

    for (const FlagSpec& f : spec.flags) {
        if (f.name == n) {
            return &f;
        }

        for (const std::string& alias : f.aliases) {
            if (alias == n) {
                return &f;
            }
        }
    }

    return nullptr;
}

const FlagSpec* findShortSpec(
    const CommandSpec& spec,
    char ch
) {
    std::string s(1, lowerChar(ch));

    for (const FlagSpec& f : spec.flags) {
        if (!f.shortName.empty() && f.shortName == s) {
            return &f;
        }
    }

    return nullptr;
}

void addFlag(
    ParsedCommand& cmd,
    const FlagSpec& spec,
    const std::string& value,
    const CliToken& tok
) {
    CliFlag f;
    f.name = spec.name;
    f.values.push_back(value);
    f.hasValue = true;
    f.line = tok.line;
    f.column = tok.column;
    cmd.flags.push_back(f);
}

void parseLongFlagToken(
    const std::vector<CliToken>& tokens,
    std::size_t& i,
    const CommandSpec& spec,
    ParsedCommand& cmd
) {
    const CliToken& tok = tokens[i];
    const std::string& text = tok.text;

    std::size_t eq = text.find('=');
    bool hasInline = (eq != std::string::npos);

    std::string name;
    std::string inlineValue;

    if (hasInline) {
        name = text.substr(2, eq - 2);
        inlineValue = text.substr(eq + 1);
    } else {
        name = text.substr(2);
    }

    const FlagSpec* fs = findLongSpec(spec, name);

    if (!fs) {
        cmd.errors.push_back(
            "unknown flag --" + name +
            " near column " + std::to_string(tok.column)
        );

        CliFlag unknown;
        unknown.name = name;
        unknown.hasValue = hasInline;
        unknown.line = tok.line;
        unknown.column = tok.column;

        if (hasInline) {
            unknown.values.push_back(inlineValue);
        }

        cmd.flags.push_back(unknown);
        ++i;
        return;
    }

    if (fs->type == FlagType::Bool) {
        addFlag(cmd, *fs, hasInline ? inlineValue : std::string("true"), tok);
        ++i;
        return;
    }

    std::string value;

    if (hasInline) {
        value = inlineValue;
    } else {
        if (
            i + 1 < tokens.size() &&
            (
                tokens[i + 1].type == CliTokenType::Value ||
                tokens[i + 1].type == CliTokenType::QuotedValue
            )
        ) {
            value = tokens[i + 1].text;
            ++i;
        } else {
            cmd.errors.push_back(
                "flag --" + name +
                " expects a value near column " +
                std::to_string(tok.column)
            );
            value.clear();
        }
    }

    addFlag(cmd, *fs, value, tok);
    ++i;
}

void parseShortFlagToken(
    const std::vector<CliToken>& tokens,
    std::size_t& i,
    const CommandSpec& spec,
    ParsedCommand& cmd
) {
    const CliToken& tok = tokens[i];
    const std::string& text = tok.text;

    std::size_t eq = text.find('=');
    bool hasInline = (eq != std::string::npos);

    std::string cluster;
    std::string inlineValue;

    if (hasInline) {
        cluster = text.substr(1, eq - 1);
        inlineValue = text.substr(eq + 1);
    } else {
        cluster = text.substr(1);
    }

    if (cluster.empty()) {
        cmd.errors.push_back(
            "unknown short flag - near column " +
            std::to_string(tok.column)
        );
        ++i;
        return;
    }

    if (hasInline && cluster.size() != 1) {
        cmd.errors.push_back(
            "short flag with inline value must be one letter near column " +
            std::to_string(tok.column)
        );
        ++i;
        return;
    }

    for (std::size_t p = 0; p < cluster.size(); ++p) {
        char ch = cluster[p];
        const FlagSpec* fs = findShortSpec(spec, ch);

        if (!fs) {
            cmd.errors.push_back(
                "unknown short flag -" + std::string(1, ch) +
                " near column " + std::to_string(tok.column)
            );
            continue;
        }

        if (fs->type == FlagType::Bool) {
            addFlag(
                cmd,
                *fs,
                hasInline ? inlineValue : std::string("true"),
                tok
            );

            if (hasInline) {
                break;
            }

            continue;
        }

        std::string value;

        if (hasInline) {
            value = inlineValue;
        } else if (p + 1 < cluster.size()) {
            value = cluster.substr(p + 1);
            p = cluster.size();
        } else {
            if (
                i + 1 < tokens.size() &&
                (
                    tokens[i + 1].type == CliTokenType::Value ||
                    tokens[i + 1].type == CliTokenType::QuotedValue
                )
            ) {
                value = tokens[i + 1].text;
                ++i;
            } else {
                cmd.errors.push_back(
                    "flag -" + std::string(1, ch) +
                    " expects a value near column " +
                    std::to_string(tok.column)
                );
                value.clear();
            }
        }

        addFlag(cmd, *fs, value, tok);
        break;
    }

    ++i;
}

void parseTokensIntoCommand(
    const std::vector<CliToken>& tokens,
    std::size_t start,
    const CommandSpec& spec,
    ParsedCommand& cmd
) {
    bool noMoreFlags = false;
    std::size_t i = start;

    while (i < tokens.size()) {
        const CliToken& tok = tokens[i];

        if (!noMoreFlags && tok.type == CliTokenType::Separator) {
            noMoreFlags = true;
            ++i;
            continue;
        }

        if (!noMoreFlags && tok.type == CliTokenType::LongFlag) {
            parseLongFlagToken(tokens, i, spec, cmd);
            continue;
        }

        if (!noMoreFlags && tok.type == CliTokenType::ShortFlag) {
            parseShortFlagToken(tokens, i, spec, cmd);
            continue;
        }

        cmd.positional.push_back(tok.text);
        ++i;
    }
}

const char* flagTypeToString(FlagType type) {
    switch (type) {
        case FlagType::Bool:
            return "bool";
        case FlagType::Int:
            return "int";
        case FlagType::Double:
            return "double";
        case FlagType::String:
            return "string";
        case FlagType::KeyVal:
            return "key=value";
    }

    return "unknown";
}

} // namespace

const CommandSpec& globalCommandSpec() {
    static const CommandSpec spec = makeGlobalSpec();
    return spec;
}

bool getLobbyCommandSpec(const std::string& name, CommandSpec& out) {
    std::string n = toLower(trim(name));

    for (const CommandSpec& spec : lobbySpecs()) {
        if (spec.name == n) {
            out = spec;
            return true;
        }
    }

    return false;
}

std::vector<std::string> knownLobbyCommands() {
    std::vector<std::string> names;

    for (const CommandSpec& spec : lobbySpecs()) {
        names.push_back(spec.name);
    }

    return names;
}

std::vector<CliToken> tokenizeCliLine(
    const std::string& line,
    std::vector<std::string>& errors
) {
    std::vector<CliToken> tokens;
    std::size_t i = 0;
    int col = 1;
    bool firstToken = true;

    while (i < line.size()) {
        unsigned char uch = static_cast<unsigned char>(line[i]);

        if (isSpaceChar(static_cast<char>(uch))) {
            ++i;
            ++col;
            continue;
        }

        if (firstToken && line[i] == '/') {
            std::size_t start = i;
            int startCol = col;

            while (
                i < line.size() &&
                !isSpaceChar(line[i]) &&
                line[i] != '"' &&
                line[i] != '\''
            ) {
                ++i;
                ++col;
            }

            tokens.push_back({
                CliTokenType::Command,
                line.substr(start, i - start),
                1,
                startCol
            });

            firstToken = false;
            continue;
        }

        if (
            line[i] == '-' &&
            i + 1 < line.size() &&
            line[i + 1] == '-'
        ) {
            if (i + 2 >= line.size() || isSpaceChar(line[i + 2])) {
                tokens.push_back({
                    CliTokenType::Separator,
                    "--",
                    1,
                    col
                });

                i += 2;
                col += 2;
                firstToken = false;
                continue;
            }

            std::size_t start = i;
            int startCol = col;

            i += 2;
            col += 2;

            while (
                i < line.size() &&
                !isSpaceChar(line[i]) &&
                line[i] != '='
            ) {
                ++i;
                ++col;
            }

            std::string name = line.substr(start + 2, i - (start + 2));
            std::string text = "--" + name;

            if (i < line.size() && line[i] == '=') {
                ++i;
                ++col;

                std::size_t valueStart = i;

                while (i < line.size() && !isSpaceChar(line[i])) {
                    ++i;
                    ++col;
                }

                text += "=" + line.substr(valueStart, i - valueStart);
            }

            tokens.push_back({
                CliTokenType::LongFlag,
                text,
                1,
                startCol
            });

            firstToken = false;
            continue;
        }

        if (
            line[i] == '-' &&
            i + 1 < line.size() &&
            isAlphaChar(line[i + 1])
        ) {
            std::size_t start = i;
            int startCol = col;

            ++i;
            ++col;

            while (
                i < line.size() &&
                !isSpaceChar(line[i]) &&
                line[i] != '='
            ) {
                ++i;
                ++col;
            }

            std::string cluster = line.substr(start + 1, i - (start + 1));
            std::string text = "-" + cluster;

            if (i < line.size() && line[i] == '=') {
                ++i;
                ++col;

                std::size_t valueStart = i;

                while (i < line.size() && !isSpaceChar(line[i])) {
                    ++i;
                    ++col;
                }

                text += "=" + line.substr(valueStart, i - valueStart);
            }

            tokens.push_back({
                CliTokenType::ShortFlag,
                text,
                1,
                startCol
            });

            firstToken = false;
            continue;
        }

        if (line[i] == '"' || line[i] == '\'') {
            char quote = line[i];
            int startCol = col;

            ++i;
            ++col;

            std::string value;
            bool closed = false;

            while (i < line.size()) {
                char c = line[i];

                if (c == '\\' && i + 1 < line.size()) {
                    char e = line[i + 1];

                    ++i;
                    ++col;

                    switch (e) {
                        case 'n':
                            value.push_back('\n');
                            break;
                        case 't':
                            value.push_back('\t');
                            break;
                        case 'r':
                            value.push_back('\r');
                            break;
                        case '\\':
                            value.push_back('\\');
                            break;
                        case '"':
                            value.push_back('"');
                            break;
                        case '\'':
                            value.push_back('\'');
                            break;
                        default:
                            value.push_back('\\');
                            value.push_back(e);
                            break;
                    }

                    ++i;
                    ++col;
                    continue;
                }

                if (c == quote) {
                    ++i;
                    ++col;
                    closed = true;
                    break;
                }

                value.push_back(c);
                ++i;
                ++col;
            }

            if (!closed) {
                errors.push_back(
                    "unmatched quote at column " + std::to_string(startCol)
                );
            }

            tokens.push_back({
                CliTokenType::QuotedValue,
                value,
                1,
                startCol
            });

            firstToken = false;
            continue;
        }

        {
            std::size_t start = i;
            int startCol = col;

            while (i < line.size() && !isSpaceChar(line[i])) {
                ++i;
                ++col;
            }

            tokens.push_back({
                CliTokenType::Value,
                line.substr(start, i - start),
                1,
                startCol
            });

            firstToken = false;
        }
    }

    return tokens;
}

std::vector<CliToken> tokenizeArgv(
    const std::vector<std::string>& args,
    std::vector<std::string>& errors
) {
    (void)errors;

    std::vector<CliToken> tokens;
    tokens.reserve(args.size());

    for (std::size_t idx = 0; idx < args.size(); ++idx) {
        const std::string& arg = args[idx];
        int col = static_cast<int>(idx + 1);

        if (arg == "--") {
            tokens.push_back({
                CliTokenType::Separator,
                arg,
                1,
                col
            });
        } else if (
            arg.size() >= 2 &&
            arg[0] == '-' &&
            arg[1] == '-'
        ) {
            tokens.push_back({
                CliTokenType::LongFlag,
                arg,
                1,
                col
            });
        } else if (
            arg.size() >= 2 &&
            arg[0] == '-' &&
            isAlphaChar(arg[1])
        ) {
            tokens.push_back({
                CliTokenType::ShortFlag,
                arg,
                1,
                col
            });
        } else {
            tokens.push_back({
                CliTokenType::Value,
                arg,
                1,
                col
            });
        }
    }

    return tokens;
}

ParsedCommand parseLobbyLine(const std::string& line) {
    ParsedCommand cmd;
    cmd.name.clear();

    std::vector<std::string> tokenErrors;
    std::vector<CliToken> tokens = tokenizeCliLine(line, tokenErrors);

    cmd.errors = tokenErrors;

    if (!tokenErrors.empty()) {
        return cmd;
    }

    if (tokens.empty()) {
        cmd.errors.push_back("empty command");
        return cmd;
    }

    if (tokens[0].type != CliTokenType::Command) {
        cmd.errors.push_back("lobby command must start with '/'");
        return cmd;
    }

    std::string rawName = tokens[0].text;

    if (!rawName.empty() && rawName[0] == '/') {
        rawName = rawName.substr(1);
    }

    cmd.name = toLower(rawName);

    CommandSpec spec;

    if (!getLobbyCommandSpec(cmd.name, spec)) {
        cmd.errors.push_back("unknown lobby command: /" + cmd.name);
        return cmd;
    }

    parseTokensIntoCommand(tokens, 1, spec, cmd);
    validateParsedCommand(cmd, spec);

    return cmd;
}

ParsedCommand parseGlobalArgs(const std::vector<std::string>& args) {
    ParsedCommand cmd;
    cmd.name = "global";

    std::vector<std::string> tokenErrors;
    std::vector<CliToken> tokens = tokenizeArgv(args, tokenErrors);

    cmd.errors = tokenErrors;

    if (!tokenErrors.empty()) {
        return cmd;
    }

    const CommandSpec& spec = globalCommandSpec();

    parseTokensIntoCommand(tokens, 0, spec, cmd);
    validateParsedCommand(cmd, spec);

    return cmd;
}

ParsedCommand parseGlobalLine(const std::string& line) {
    ParsedCommand cmd;
    cmd.name = "global";

    std::vector<std::string> tokenErrors;
    std::vector<CliToken> tokens = tokenizeCliLine(line, tokenErrors);

    cmd.errors = tokenErrors;

    if (!tokenErrors.empty()) {
        return cmd;
    }

    std::size_t start = 0;

    if (
        !tokens.empty() &&
        tokens[0].type == CliTokenType::Value &&
        (
            tokens[0].text == "proyek" ||
            tokens[0].text == "./proyek" ||
            tokens[0].text == ".\\proyek"
        )
    ) {
        start = 1;
    }

    const CommandSpec& spec = globalCommandSpec();

    parseTokensIntoCommand(tokens, start, spec, cmd);
    validateParsedCommand(cmd, spec);

    return cmd;
}

bool parseBoolValue(
    const std::string& value,
    bool& out,
    std::string& err
) {
    std::string v = toLower(trim(value));

    if (v == "true" || v == "1" || v == "yes" || v == "on") {
        out = true;
        err.clear();
        return true;
    }

    if (v == "false" || v == "0" || v == "no" || v == "off") {
        out = false;
        err.clear();
        return true;
    }

    err = "invalid boolean value: '" + value + "'";
    return false;
}

bool parseIntValue(
    const std::string& value,
    long long& out,
    std::string& err
) {
    std::string v = trim(value);

    if (v.empty()) {
        err = "empty integer value";
        return false;
    }

    try {
        std::size_t pos = 0;
        out = std::stoll(v, &pos);

        if (pos != v.size()) {
            err = "trailing characters in integer value: '" + value + "'";
            return false;
        }

        err.clear();
        return true;
    } catch (...) {
        err = "invalid integer value: '" + value + "'";
        return false;
    }
}

bool parseDoubleValue(
    const std::string& value,
    double& out,
    std::string& err
) {
    std::string v = trim(value);

    if (v.empty()) {
        err = "empty double value";
        return false;
    }

    try {
        std::size_t pos = 0;
        out = std::stod(v, &pos);

        if (pos != v.size()) {
            err = "trailing characters in double value: '" + value + "'";
            return false;
        }

        if (!std::isfinite(out)) {
            err = "non-finite double value: '" + value + "'";
            return false;
        }

        err.clear();
        return true;
    } catch (...) {
        err = "invalid double value: '" + value + "'";
        return false;
    }
}

void validateParsedCommand(
    ParsedCommand& cmd,
    const CommandSpec& spec
) {
    for (const FlagSpec& fs : spec.flags) {
        if (fs.required && !hasFlag(cmd, fs.name)) {
            cmd.errors.push_back("missing required flag --" + fs.name);
        }
    }

    if (!spec.allowPositional) {
        if (!cmd.positional.empty()) {
            cmd.errors.push_back("command does not accept positional arguments");
        }
    } else {
        int posCount = static_cast<int>(cmd.positional.size());

        if (posCount < spec.minPositional) {
            cmd.errors.push_back(
                "command requires at least " +
                std::to_string(spec.minPositional) +
                " positional argument(s)"
            );
        }

        if (spec.maxPositional >= 0 && posCount > spec.maxPositional) {
            cmd.errors.push_back(
                "command accepts at most " +
                std::to_string(spec.maxPositional) +
                " positional argument(s)"
            );
        }
    }

    for (const CliFlag& flag : cmd.flags) {
        const FlagSpec* fs = findLongSpec(spec, flag.name);

        if (!fs) {
            continue;
        }

        for (const std::string& value : flag.values) {
            std::string terr;

            switch (fs->type) {
                case FlagType::Bool: {
                    bool b = false;

                    if (!parseBoolValue(value, b, terr)) {
                        cmd.errors.push_back(
                            "flag --" + flag.name + ": " + terr +
                            " near column " + std::to_string(flag.column)
                        );
                    }

                    break;
                }

                case FlagType::Int: {
                    long long x = 0;

                    if (!parseIntValue(value, x, terr)) {
                        cmd.errors.push_back(
                            "flag --" + flag.name + ": " + terr +
                            " near column " + std::to_string(flag.column)
                        );
                    }

                    break;
                }

                case FlagType::Double: {
                    double x = 0.0;

                    if (!parseDoubleValue(value, x, terr)) {
                        cmd.errors.push_back(
                            "flag --" + flag.name + ": " + terr +
                            " near column " + std::to_string(flag.column)
                        );
                    }

                    break;
                }

                case FlagType::String:
                    break;

                case FlagType::KeyVal:
                    if (value.find('=') == std::string::npos) {
                        cmd.errors.push_back(
                            "flag --" + flag.name +
                            " expects KEY=VALUE near column " +
                            std::to_string(flag.column)
                        );
                    }
                    break;
            }
        }
    }
}

bool hasFlag(
    const ParsedCommand& cmd,
    const std::string& name
) {
    std::string n = toLower(name);

    for (const CliFlag& f : cmd.flags) {
        if (f.name == n) {
            return true;
        }
    }

    return false;
}

bool getFlagValue(
    const ParsedCommand& cmd,
    const std::string& name,
    std::string& value
) {
    return getLastFlagValue(cmd, name, value);
}

bool getLastFlagValue(
    const ParsedCommand& cmd,
    const std::string& name,
    std::string& value
) {
    std::string n = toLower(name);
    bool found = false;

    for (const CliFlag& f : cmd.flags) {
        if (f.name == n && !f.values.empty()) {
            value = f.values.back();
            found = true;
        }
    }

    return found;
}

std::vector<std::string> getAllFlagValues(
    const ParsedCommand& cmd,
    const std::string& name
) {
    std::vector<std::string> out;
    std::string n = toLower(name);

    for (const CliFlag& f : cmd.flags) {
        if (f.name == n) {
            for (const std::string& v : f.values) {
                out.push_back(v);
            }
        }
    }

    return out;
}

bool getBoolFlag(
    const ParsedCommand& cmd,
    const std::string& name,
    bool& out
) {
    std::string v;

    if (!getLastFlagValue(cmd, name, v)) {
        return false;
    }

    std::string err;
    return parseBoolValue(v, out, err);
}

bool getIntFlag(
    const ParsedCommand& cmd,
    const std::string& name,
    long long& out
) {
    std::string v;

    if (!getLastFlagValue(cmd, name, v)) {
        return false;
    }

    std::string err;
    return parseIntValue(v, out, err);
}

bool getDoubleFlag(
    const ParsedCommand& cmd,
    const std::string& name,
    double& out
) {
    std::string v;

    if (!getLastFlagValue(cmd, name, v)) {
        return false;
    }

    std::string err;
    return parseDoubleValue(v, out, err);
}

std::string joinPositionals(
    const ParsedCommand& cmd,
    const std::string& separator
) {
    std::ostringstream os;

    for (std::size_t i = 0; i < cmd.positional.size(); ++i) {
        if (i != 0) {
            os << separator;
        }

        os << cmd.positional[i];
    }

    return os.str();
}

std::string commandUsage(const CommandSpec& spec) {
    std::ostringstream os;

    if (spec.name == "global") {
        os << "proyek";
    } else {
        os << "/" << spec.name;
    }

    os << " - " << spec.help << "\n";

    if (!spec.flags.empty()) {
        os << "Flags:\n";

        for (const FlagSpec& f : spec.flags) {
            os << "  --" << f.name;

            if (!f.shortName.empty()) {
                os << ", -" << f.shortName;
            }

            if (f.type == FlagType::KeyVal) {
                os << " KEY=VALUE";
            } else if (f.type != FlagType::Bool) {
                os << " VALUE";
            }

            if (f.required) {
                os << " (required)";
            }

            os << " : " << f.help;
            os << " [type=" << flagTypeToString(f.type) << "]";

            if (!f.aliases.empty()) {
                os << " aliases:";

                for (const std::string& a : f.aliases) {
                    os << " --" << a;
                }
            }

            os << "\n";
        }
    }

    if (spec.allowPositional) {
        os << "Positional:\n";
        os << "  min=" << spec.minPositional;

        if (spec.maxPositional >= 0) {
            os << ", max=" << spec.maxPositional;
        } else {
            os << ", max=unlimited";
        }

        os << "\n";
    }

    return os.str();
}

#ifdef CLI_SELFTEST

int main() {
    int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "CHECK failed: " #cond                           \
                      << " at line " << __LINE__ << "\n";                 \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

    ParsedCommand cmd;

    // 1. Basic lobby command.
    cmd = parseLobbyLine("/training --epochs 3");
    CHECK(cmd.ok());
    CHECK(cmd.name == "training");
    CHECK(hasFlag(cmd, "epochs"));

    std::string v;
    CHECK(getLastFlagValue(cmd, "epochs", v));
    CHECK(v == "3");

    long long epochs = 0;
    CHECK(getIntFlag(cmd, "epochs", epochs));
    CHECK(epochs == 3);

    // 2. Quoted prompt.
    cmd = parseLobbyLine(R"(/prompt "Halo dunia" --top-k 40)");
    CHECK(cmd.ok());
    CHECK(cmd.name == "prompt");
    CHECK(cmd.positional.size() == 1);
    CHECK(cmd.positional[0] == "Halo dunia");

    long long topK = 0;
    CHECK(getIntFlag(cmd, "top-k", topK));
    CHECK(topK == 40);

    // 3. Escaped quote inside double-quoted string.
    cmd = parseLobbyLine(R"(/prompt "He said \"hi\"" --max-new-tokens 5)");
    CHECK(cmd.ok());
    CHECK(cmd.positional.size() == 1);
    CHECK(cmd.positional[0] == "He said \"hi\"");

    long long maxNewTokens = 0;
    CHECK(getIntFlag(cmd, "max-new-tokens", maxNewTokens));
    CHECK(maxNewTokens == 5);

    // 4. Single-quoted string and alias flag.
    cmd = parseLobbyLine(R"(/prompt 'halo dunia' --temp 0.8)");
    CHECK(cmd.ok());
    CHECK(cmd.positional.size() == 1);
    CHECK(cmd.positional[0] == "halo dunia");

    double temp = 0.0;
    CHECK(getDoubleFlag(cmd, "temperature", temp));
    CHECK(std::fabs(temp - 0.8) < 1e-6);

    // 5. Inline flags and scientific notation.
    cmd = parseLobbyLine("/training --epochs=5 --lr=3e-4");
    CHECK(cmd.ok());

    long long epochsInline = 0;
    CHECK(getIntFlag(cmd, "epochs", epochsInline));
    CHECK(epochsInline == 5);

    double lr = 0.0;
    CHECK(getDoubleFlag(cmd, "lr", lr));
    CHECK(std::fabs(lr - 0.0003) < 1e-6);

    // 6. Negative numeric value must not be treated as flag.
    cmd = parseLobbyLine("/training --grad-clip -1.0");
    CHECK(cmd.ok());

    double gradClip = 0.0;
    CHECK(getDoubleFlag(cmd, "grad-clip", gradClip));
    CHECK(std::fabs(gradClip - (-1.0)) < 1e-6);

    // 7. Boolean flags.
    cmd = parseLobbyLine("/training --dry-run");
    CHECK(cmd.ok());

    bool dryRun = false;
    CHECK(getBoolFlag(cmd, "dry-run", dryRun));
    CHECK(dryRun == true);

    cmd = parseLobbyLine("/training --dry-run=false");
    CHECK(cmd.ok());

    dryRun = true;
    CHECK(getBoolFlag(cmd, "dry-run", dryRun));
    CHECK(dryRun == false);

    // 8. Unknown flag error.
    cmd = parseLobbyLine("/training --epochss 3");
    CHECK(!cmd.ok());

    bool foundUnknown = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("unknown flag") != std::string::npos) {
            foundUnknown = true;
            break;
        }
    }
    CHECK(foundUnknown);

    // 9. Missing value error.
    cmd = parseLobbyLine("/training --epochs");
    CHECK(!cmd.ok());

    bool foundMissingValue = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("expects a value") != std::string::npos) {
            foundMissingValue = true;
            break;
        }
    }
    CHECK(foundMissingValue);

    // 10. Short flag with separated value.
    cmd = parseLobbyLine("/training -e 3");
    CHECK(cmd.ok());

    long long shortEpochs = 0;
    CHECK(getIntFlag(cmd, "epochs", shortEpochs));
    CHECK(shortEpochs == 3);

    // 11. Short flag with attached value.
    cmd = parseLobbyLine("/training -e3");
    CHECK(cmd.ok());

    shortEpochs = 0;
    CHECK(getIntFlag(cmd, "epochs", shortEpochs));
    CHECK(shortEpochs == 3);

    // 12. Short boolean cluster.
    cmd = parseGlobalArgs({"-vy"});
    CHECK(cmd.ok());
    CHECK(hasFlag(cmd, "version"));
    CHECK(hasFlag(cmd, "yes"));

    bool yes = false;
    CHECK(getBoolFlag(cmd, "yes", yes));
    CHECK(yes == true);

    // 13. Separator -- makes following tokens positional.
    cmd = parseLobbyLine(R"(/prompt -- --not-a-flag)");
    CHECK(cmd.ok());
    CHECK(cmd.positional.size() == 1);
    CHECK(cmd.positional[0] == "--not-a-flag");

    // 14. Unmatched quote error.
    cmd = parseLobbyLine("/prompt \"halo");
    CHECK(!cmd.ok());

    bool foundUnmatched = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("unmatched quote") != std::string::npos) {
            foundUnmatched = true;
            break;
        }
    }
    CHECK(foundUnmatched);

    // 15. Required flag validation.
    cmd = parseLobbyLine("/switch");
    CHECK(!cmd.ok());

    bool foundRequired = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("missing required flag") != std::string::npos) {
            foundRequired = true;
            break;
        }
    }
    CHECK(foundRequired);

    cmd = parseLobbyLine("/switch --model melati");
    CHECK(cmd.ok());

    std::string modelName;
    CHECK(getLastFlagValue(cmd, "model", modelName));
    CHECK(modelName == "melati");

    // 16. Positional count validation.
    cmd = parseLobbyLine("/status extra");
    CHECK(!cmd.ok());

    bool foundPositionalError = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("positional") != std::string::npos) {
            foundPositionalError = true;
            break;
        }
    }
    CHECK(foundPositionalError);

    // 17. KeyVal flag.
    cmd = parseLobbyLine("/config --set model.d_model=256");
    CHECK(cmd.ok());

    std::vector<std::string> sets = getAllFlagValues(cmd, "set");
    CHECK(sets.size() == 1);
    CHECK(sets[0] == "model.d_model=256");

    cmd = parseLobbyLine("/config --set model.d_model=256 --set train.epochs=2");
    CHECK(cmd.ok());

    sets = getAllFlagValues(cmd, "set");
    CHECK(sets.size() == 2);
    CHECK(sets[0] == "model.d_model=256");
    CHECK(sets[1] == "train.epochs=2");

    // 18. Invalid integer.
    cmd = parseLobbyLine("/training --epochs abc");
    CHECK(!cmd.ok());

    bool foundIntError = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("invalid integer") != std::string::npos) {
            foundIntError = true;
            break;
        }
    }
    CHECK(foundIntError);

    // 19. Invalid boolean.
    cmd = parseGlobalArgs({"--yes=maybe"});
    CHECK(!cmd.ok());

    bool foundBoolError = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("invalid boolean") != std::string::npos) {
            foundBoolError = true;
            break;
        }
    }
    CHECK(foundBoolError);

    // 20. Global args.
    cmd = parseGlobalArgs({
        "--model-name",
        "mawar",
        "--yes",
        "--seed",
        "42"
    });

    CHECK(cmd.ok());
    CHECK(cmd.name == "global");

    std::string globalModelName;
    CHECK(getLastFlagValue(cmd, "model-name", globalModelName));
    CHECK(globalModelName == "mawar");

    bool globalYes = false;
    CHECK(getBoolFlag(cmd, "yes", globalYes));
    CHECK(globalYes == true);

    long long seed = 0;
    CHECK(getIntFlag(cmd, "seed", seed));
    CHECK(seed == 42);

    // 21. Global unknown flag.
    cmd = parseGlobalArgs({"--nope"});
    CHECK(!cmd.ok());

    foundUnknown = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("unknown flag") != std::string::npos) {
            foundUnknown = true;
            break;
        }
    }
    CHECK(foundUnknown);

    // 22. Global rejects positional by default.
    cmd = parseGlobalArgs({"foo"});
    CHECK(!cmd.ok());

    foundPositionalError = false;
    for (const std::string& e : cmd.errors) {
        if (e.find("positional") != std::string::npos) {
            foundPositionalError = true;
            break;
        }
    }
    CHECK(foundPositionalError);

    // 23. parseGlobalLine skips leading program name.
    cmd = parseGlobalLine(R"(./proyek --model-name "mawar" --yes)");
    CHECK(cmd.ok());

    globalModelName.clear();
    CHECK(getLastFlagValue(cmd, "model-name", globalModelName));
    CHECK(globalModelName == "mawar");

    globalYes = false;
    CHECK(getBoolFlag(cmd, "yes", globalYes));
    CHECK(globalYes == true);

    // 24. joinPositionals.
    cmd = parseLobbyLine("/prompt halo dunia apa kabar");
    CHECK(cmd.ok());
    CHECK(cmd.positional.size() == 4);
    CHECK(joinPositionals(cmd, " ") == "halo dunia apa kabar");

    // 25. Usage text.
    std::string globalUsage = commandUsage(globalCommandSpec());
    CHECK(globalUsage.find("--model-name") != std::string::npos);
    CHECK(globalUsage.find("--set") != std::string::npos);

    CommandSpec trainingSpec;
    CHECK(getLobbyCommandSpec("training", trainingSpec));

    std::string trainingUsage = commandUsage(trainingSpec);
    CHECK(trainingUsage.find("--epochs") != std::string::npos);
    CHECK(trainingUsage.find("--lr") != std::string::npos);

    // 26. Known lobby commands.
    std::vector<std::string> commands = knownLobbyCommands();
    CHECK(commands.size() >= 10);

    bool foundPrompt = false;
    bool foundTraining = false;

    for (const std::string& c : commands) {
        if (c == "prompt") foundPrompt = true;
        if (c == "training") foundTraining = true;
    }

    CHECK(foundPrompt);
    CHECK(foundTraining);

    if (failures == 0) {
        std::cout << "cli unit tests passed\n";
        return 0;
    }

    std::cerr << failures << " cli unit test(s) failed\n";
    return 1;
}

#endif // CLI_SELFTEST
