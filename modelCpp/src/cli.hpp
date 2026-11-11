#ifndef PROYEK_CLI_HPP
#define PROYEK_CLI_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class CliTokenType {
    Command,
    LongFlag,
    ShortFlag,
    Value,
    QuotedValue,
    Separator
};

struct CliToken {
    CliTokenType type;
    std::string text;
    int line;
    int column;
};

struct CliFlag {
    std::string name;
    std::vector<std::string> values;
    bool hasValue;
    int line;
    int column;
};

struct ParsedCommand {
    std::string name;
    std::vector<CliFlag> flags;
    std::vector<std::string> positional;
    std::vector<std::string> errors;

    bool ok() const {
        return errors.empty();
    }
};

enum class FlagType {
    Bool,
    Int,
    Double,
    String,
    KeyVal
};

struct FlagSpec {
    std::string name;
    std::vector<std::string> aliases;
    std::string shortName;
    FlagType type;
    bool required;
    std::string help;
};

struct CommandSpec {
    std::string name;
    std::vector<FlagSpec> flags;
    bool allowPositional;
    int minPositional;
    int maxPositional; // -1 berarti unlimited
    std::string help;
};

const CommandSpec& globalCommandSpec();

bool getLobbyCommandSpec(const std::string& name, CommandSpec& out);

std::vector<std::string> knownLobbyCommands();

std::vector<CliToken> tokenizeCliLine(
    const std::string& line,
    std::vector<std::string>& errors
);

std::vector<CliToken> tokenizeArgv(
    const std::vector<std::string>& args,
    std::vector<std::string>& errors
);

ParsedCommand parseLobbyLine(const std::string& line);

ParsedCommand parseGlobalArgs(const std::vector<std::string>& args);

ParsedCommand parseGlobalLine(const std::string& line);

void validateParsedCommand(
    ParsedCommand& cmd,
    const CommandSpec& spec
);

bool parseBoolValue(
    const std::string& value,
    bool& out,
    std::string& err
);

bool parseIntValue(
    const std::string& value,
    long long& out,
    std::string& err
);

bool parseDoubleValue(
    const std::string& value,
    double& out,
    std::string& err
);

bool hasFlag(
    const ParsedCommand& cmd,
    const std::string& name
);

bool getFlagValue(
    const ParsedCommand& cmd,
    const std::string& name,
    std::string& value
);

bool getLastFlagValue(
    const ParsedCommand& cmd,
    const std::string& name,
    std::string& value
);

std::vector<std::string> getAllFlagValues(
    const ParsedCommand& cmd,
    const std::string& name
);

bool getBoolFlag(
    const ParsedCommand& cmd,
    const std::string& name,
    bool& out
);

bool getIntFlag(
    const ParsedCommand& cmd,
    const std::string& name,
    long long& out
);

bool getDoubleFlag(
    const ParsedCommand& cmd,
    const std::string& name,
    double& out
);

std::string joinPositionals(
    const ParsedCommand& cmd,
    const std::string& separator = " "
);

std::string commandUsage(const CommandSpec& spec);

#endif // PROYEK_CLI_HPP
