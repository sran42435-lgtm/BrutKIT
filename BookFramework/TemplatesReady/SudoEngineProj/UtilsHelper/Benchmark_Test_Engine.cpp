#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <regex>
#include <chrono>
#include <iomanip>
#include <array>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/select.h>

// ==========================================
// STRUKTUR DATA & ENUM
// ==========================================
enum BenchmarkTechnique {
    BT_NAKED_SINGLE,
    BT_HIDDEN_SINGLE,
    BT_LOCKED_CANDIDATES,
    BT_NAKED_SUBSETS,
    BT_XWING,
    BT_CONTRADICTION_PROBE,
    BT_COUNT
};

constexpr const char* BENCHMARK_TECH_NAMES[BT_COUNT] = {
    "Naked Single",
    "Hidden Single",
    "Locked Candidates",
    "Naked Subsets",
    "X-Wing",
    "Contradiction Probe"
};

struct TechniqueSnapshot {
    double time_us = 0.0;
    long long calls = 0;
    uint32_t invoked = 0;
    uint32_t patterns = 0;
    uint32_t removed = 0;
    uint32_t filled = 0;
};

struct SolveSnapshot {
    double time = -1.0;
    int propagation = 0;
    int elimination = 0;
    int guesses = 0;
    int deadend = 0;
    int retry = 0;
    std::array<TechniqueSnapshot, BT_COUNT> techniques{};
    bool valid = false;
};

struct BenchmarkResult {
    std::string engine_name;
    std::string data_file;

    std::vector<SolveSnapshot> runs_data;

    int run_success = 0;
    int run_failure = 0;

    double min_time = 0, max_time = 0, mean_time = 0, median_time = 0, stddev_time = 0;

    double median_propagation = 0;
    double median_elimination = 0;
    double median_guesses = 0;
    double median_deadends = 0;
    double median_retries = 0;

    std::array<TechniqueSnapshot, BT_COUNT> technique_medians{};
};

struct EngineConfig {
    std::string path;
    std::string data_file;
};

// ==========================================
// PREFLIGHT CHECK SYSTEM
// ==========================================
enum class CoordinateStatus {
    NO_SESSION_ID,
    COORDINATE_READY,
    COORDINATE_MISSING_OR_INVALID
};

struct PreflightInfo {
    bool has_session_id = false;
    std::string session_id;
    CoordinateStatus coordinate_status = CoordinateStatus::NO_SESSION_ID;
    bool needs_calculate = true;
    std::string coordinate_file;
};

bool read_session_id_from_puzzle(const std::string& data_file, std::string& session_id) {
    std::ifstream file(data_file);
    if (!file.is_open()) return false;

    std::string line;
    // Mendukung format: [id_session: 12345] atau SESSION_ID: 12345
    std::regex session_regex1(R"(\[id_session:\s*([0-9]+)\])");
    std::regex session_regex2(R"((?:SESSION_ID|Session ID)\s*:\s*([0-9]+))");

    while (std::getline(file, line)) {
        std::smatch match;
        if (std::regex_search(line, match, session_regex1)) {
            session_id = match[1].str();
            return true;
        }
        if (std::regex_search(line, match, session_regex2)) {
            session_id = match[1].str();
            return true;
        }
    }
    return false;
}

std::string get_coordinate_filename(const std::string& session_id) {
    return "data_koordinat_" + session_id + ".txt";
}

bool validate_coordinate_file(const std::string& filename) {
    namespace fs = std::filesystem;

    if (!fs::exists(filename)) return false;
    if (!fs::is_regular_file(filename)) return false;
    if (fs::file_size(filename) == 0) return false;

    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    bool has_content = false;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            has_content = true;
            break;
        }
    }
    return has_content;
}

PreflightInfo inspect_puzzle_data(const std::string& data_file) {
    PreflightInfo info;

    if (!read_session_id_from_puzzle(data_file, info.session_id)) {
        info.has_session_id = false;
        info.coordinate_status = CoordinateStatus::NO_SESSION_ID;
        info.needs_calculate = true;
        return info;
    }

    info.has_session_id = true;
    info.coordinate_file = get_coordinate_filename(info.session_id);

    if (validate_coordinate_file(info.coordinate_file)) {
        info.coordinate_status = CoordinateStatus::COORDINATE_READY;
        info.needs_calculate = false;
    } else {
        info.coordinate_status = CoordinateStatus::COORDINATE_MISSING_OR_INVALID;
        info.needs_calculate = true;
    }

    return info;
}

// ==========================================
// HELPER: Pipe I/O & Parsing
// ==========================================
std::string read_until(int fd, const std::vector<std::string>& targets, int timeout_ms = 15000) {
    std::string buffer;
    char c;
    auto start = std::chrono::steady_clock::now();
    
    while (true) {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(fd, &set);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 50000; 

        int rv = select(fd + 1, &set, NULL, NULL, &tv);
        if (rv == -1) break;
        else if (rv == 0) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms) break;
            continue;
        } else {
            ssize_t bytes = read(fd, &c, 1);
            if (bytes > 0) {
                buffer += c;
                for (const auto& t : targets) {
                    if (buffer.find(t) != std::string::npos) return buffer;
                }
            } else {
                break; 
            }
        }
    }
    return buffer;
}

void write_cmd(int fd, const std::string& cmd) {
    write(fd, cmd.c_str(), cmd.length());
    write(fd, "\n", 1);
}

template <typename T>
double calculate_median(std::vector<T> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (static_cast<double>(values[n/2 - 1]) + static_cast<double>(values[n/2])) / 2.0;
    }
    return static_cast<double>(values[n/2]);
}

void parse_technique_block(const std::string& output, const std::string& name, TechniqueSnapshot& res) {
    std::string search_str = name + ":";
    size_t start = output.find(search_str);
    if (start == std::string::npos) return; 
    
    size_t end = output.find("\n\n", start);
    std::string block = (end == std::string::npos) ? output.substr(start) : output.substr(start, end - start);
    
    std::regex inv_r("Invoked\\s*:\\s*([0-9]+)");
    std::regex pat_r("Pattern Detected\\s*:\\s*([0-9]+)");
    std::regex rem_r("Candidate Removed\\s*:\\s*([0-9]+)");
    std::regex fil_r("Cell Filled\\s*:\\s*([0-9]+)");
    
    std::smatch m;
    if (std::regex_search(block, m, inv_r)) res.invoked = std::stoul(m[1]);
    if (std::regex_search(block, m, pat_r)) res.patterns = std::stoul(m[1]);
    if (std::regex_search(block, m, rem_r)) res.removed = std::stoul(m[1]);
    if (std::regex_search(block, m, fil_r)) res.filled = std::stoul(m[1]);
}

void parse_technique_timing(const std::string& output, const std::string& name, TechniqueSnapshot& res) {
    std::string pattern = name + "\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)\\s*\\S+\\s*\\|\\s*Calls\\s*:\\s*([0-9]+)";
    std::regex r(pattern);
    std::smatch m;
    if (std::regex_search(output, m, r)) {
        res.time_us = std::stod(m[1]);
        res.calls = std::stoll(m[2]);
    }
}

void add_stats(TechniqueSnapshot& target, const TechniqueSnapshot& source) {
    target.invoked += source.invoked;
    target.patterns += source.patterns;
    target.removed += source.removed;
    target.filled += source.filled;
}

bool run_calculate_phase(int p_in, int p_out) {
    write_cmd(p_in, "/calculate");
    std::string output = read_until(p_out, {"Tekan ENTER..."});
    write_cmd(p_in, "");
    read_until(p_out, {"Perintah:"});
    return !output.empty();
}

// ==========================================
// INTI: Menjalankan Engine via Fork & Pipe
// ==========================================
BenchmarkResult run_engine(const std::string& engine_path, const std::string& data_file, int warmups, int runs) {
    BenchmarkResult res;
    res.engine_name = engine_path;
    res.data_file = data_file;

    res.runs_data.reserve(runs);

    // ==========================================
    // 1. PRE-FLIGHT DATA CHECK
    // ==========================================
    PreflightInfo preflight = inspect_puzzle_data(data_file);
    
    std::cout << "\n=== PRE-FLIGHT DATA ===\n";
    if (!preflight.has_session_id) {
        std::cout << "[INFO] Session ID belum tersedia di file puzzle.\n";
        std::cout << "[INFO] /calculate diperlukan setelah /load.\n";
    } else {
        std::cout << "[INFO] Session ID ditemukan: " << preflight.session_id << "\n";
        std::cout << "[INFO] File koordinat: " << preflight.coordinate_file << "\n";
        if (preflight.coordinate_status == CoordinateStatus::COORDINATE_READY) {
            std::cout << "[OK] Data koordinat tersedia dan lolos validasi.\n";
            std::cout << "[INFO] /calculate akan dilewati.\n";
        } else {
            std::cout << "[WARNING] Data koordinat tidak tersedia atau tidak valid.\n";
            std::cout << "[INFO] /calculate diperlukan.\n";
        }
    }
    std::cout << "──────────────────────────\n";

    int stdin_pipe[2];
    int stdout_pipe[2];
    if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1) {
        std::cerr << "  [ERROR] Gagal membuat pipe.\n";
        return res;
    }

    pid_t pid = fork();
    if (pid == -1) {
        std::cerr << "  [ERROR] Gagal melakukan fork.\n";
        return res;
    }

    if (pid == 0) { 
        close(stdin_pipe[1]); close(stdout_pipe[0]);
        dup2(stdin_pipe[0], STDIN_FILENO); dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdin_pipe[0]); close(stdout_pipe[1]);
        execlp(engine_path.c_str(), engine_path.c_str(), NULL);
        exit(1);
    }

    close(stdin_pipe[0]); close(stdout_pipe[1]);
    int p_in = stdin_pipe[1];
    int p_out = stdout_pipe[0];

    // ==========================================
    // 2. LOAD PHASE
    // ==========================================
    read_until(p_out, {"Perintah:"});
    write_cmd(p_in, "/load " + data_file);
    read_until(p_out, {"Tekan ENTER..."});
    write_cmd(p_in, ""); 
    read_until(p_out, {"Perintah:"}); 

    // ==========================================
    // 3. PREPARATION PHASE (Calculate jika perlu)
    // ==========================================
    if (preflight.needs_calculate) {
        std::cout << "  [Preparation] Menjalankan /calculate..." << std::flush;
        run_calculate_phase(p_in, p_out);
        std::cout << " Selesai.\n";
    } else {
        std::cout << "  [Preparation] /calculate dilewati (koordinat sudah tersedia).\n";
    }

    // ==========================================
    // 4. BENCHMARK PHASE
    // ==========================================
    std::regex time_regex("(?:Waktu|SOLVE_TIME|Time)\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)\\s*ms");
    std::regex prop_regex("Propagasi\\s*:\\s*([0-9]+)");
    std::regex elim_regex("Eliminasi\\s*:\\s*([0-9]+)");
    std::regex guess_regex("Tebakan\\s*:\\s*([0-9]+)");
    std::regex dead_regex("deadend:\\s*([0-9]+)");
    std::regex retry_regex("retry:\\s*([0-9]+)");

    auto do_solve = [&]() -> SolveSnapshot {
        SolveSnapshot snapshot;
        write_cmd(p_in, "/solve");
        std::string solve_out = read_until(p_out, {"Tekan ENTER..."});
        write_cmd(p_in, ""); 
        read_until(p_out, {"Perintah:"}); 

        std::smatch match;
        if (!std::regex_search(solve_out, match, time_regex)) return snapshot;
        snapshot.time = std::stod(match[1].str());

        if (std::regex_search(solve_out, match, prop_regex)) snapshot.propagation = std::stoi(match[1].str());
        if (std::regex_search(solve_out, match, elim_regex)) snapshot.elimination = std::stoi(match[1].str());
        if (std::regex_search(solve_out, match, guess_regex)) snapshot.guesses = std::stoi(match[1].str());
        if (std::regex_search(solve_out, match, dead_regex)) snapshot.deadend = std::stoi(match[1].str());
        if (std::regex_search(solve_out, match, retry_regex)) snapshot.retry = std::stoi(match[1].str());

        // Naked Single
        parse_technique_block(solve_out, "Naked Single", snapshot.techniques[BT_NAKED_SINGLE]);
        parse_technique_timing(solve_out, "Naked Single", snapshot.techniques[BT_NAKED_SINGLE]);

        // Hidden Single (Aggregation)
        TechniqueSnapshot hs_row, hs_col, hs_box;
        parse_technique_block(solve_out, "Hidden Single (Baris)", hs_row);
        parse_technique_block(solve_out, "Hidden Single (Kolom)", hs_col);
        parse_technique_block(solve_out, "Hidden Single (Box)", hs_box);
        add_stats(snapshot.techniques[BT_HIDDEN_SINGLE], hs_row);
        add_stats(snapshot.techniques[BT_HIDDEN_SINGLE], hs_col);
        add_stats(snapshot.techniques[BT_HIDDEN_SINGLE], hs_box);
        parse_technique_timing(solve_out, "Hidden Single", snapshot.techniques[BT_HIDDEN_SINGLE]);

        // Locked Candidates (Aggregation)
        TechniqueSnapshot lc_claim, lc_point;
        parse_technique_block(solve_out, "Locked Claiming", lc_claim);
        parse_technique_block(solve_out, "Locked Pointing", lc_point);
        add_stats(snapshot.techniques[BT_LOCKED_CANDIDATES], lc_claim);
        add_stats(snapshot.techniques[BT_LOCKED_CANDIDATES], lc_point);
        parse_technique_timing(solve_out, "Locked Candidates", snapshot.techniques[BT_LOCKED_CANDIDATES]);

        // Naked Subsets (Aggregation)
        TechniqueSnapshot ns_pair, ns_triple, ns_quad;
        parse_technique_block(solve_out, "Naked Pair", ns_pair);
        parse_technique_block(solve_out, "Naked Triple", ns_triple);
        parse_technique_block(solve_out, "Naked Quad", ns_quad);
        add_stats(snapshot.techniques[BT_NAKED_SUBSETS], ns_pair);
        add_stats(snapshot.techniques[BT_NAKED_SUBSETS], ns_triple);
        add_stats(snapshot.techniques[BT_NAKED_SUBSETS], ns_quad);
        parse_technique_timing(solve_out, "Naked Subsets", snapshot.techniques[BT_NAKED_SUBSETS]);

        // X-Wing
        parse_technique_block(solve_out, "X-Wing", snapshot.techniques[BT_XWING]);
        parse_technique_timing(solve_out, "X-Wing", snapshot.techniques[BT_XWING]);

        // Contradiction Probe
        parse_technique_block(solve_out, "Contradiction Probe", snapshot.techniques[BT_CONTRADICTION_PROBE]);
        parse_technique_timing(solve_out, "Contradiction Probe", snapshot.techniques[BT_CONTRADICTION_PROBE]);

        snapshot.valid = true;
        return snapshot;
    };

    std::cout << "  [Warm-up " << warmups << "x] " << std::flush;
    for (int i = 0; i < warmups; ++i) { do_solve(); std::cout << "." << std::flush; }
    std::cout << " Selesai.\n";

    std::cout << "  [Benchmark " << runs << "x] " << std::flush;
    for (int i = 0; i < runs; ++i) {
        SolveSnapshot snapshot = do_solve();
        if (snapshot.valid) {
            res.runs_data.push_back(std::move(snapshot));
            res.run_success++;
        } else {
            res.run_failure++;
        }
        std::cout << "." << std::flush;
    }
    std::cout << " Selesai.\n";

    write_cmd(p_in, "/quit");
    close(p_in); close(p_out);
    waitpid(pid, NULL, 0);

    return res;
}

// ==========================================
// KALKULASI STATISTIK & MEDIAN
// ==========================================
void calculate_stats(BenchmarkResult& res) {
    if (res.runs_data.empty()) return;

    std::vector<double> times;
    std::vector<int> props, elims, guesses, deads, retries;
    
    size_t n = res.runs_data.size();
    times.reserve(n); props.reserve(n); elims.reserve(n); 
    guesses.reserve(n); deads.reserve(n); retries.reserve(n);

    for (const auto& run : res.runs_data) {
        times.push_back(run.time);
        props.push_back(run.propagation);
        elims.push_back(run.elimination);
        guesses.push_back(run.guesses);
        deads.push_back(run.deadend);
        retries.push_back(run.retry);
    }

    std::vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    
    res.min_time = sorted_times.front();
    res.max_time = sorted_times.back();
    res.median_time = calculate_median(times);

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    res.mean_time = sum / n;

    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - res.mean_time) * (t - res.mean_time);
    res.stddev_time = std::sqrt(sq_sum / n);

    res.median_propagation = calculate_median(props);
    res.median_elimination = calculate_median(elims);
    res.median_guesses = calculate_median(guesses);
    res.median_deadends = calculate_median(deads);
    res.median_retries = calculate_median(retries);

    for (int i = 0; i < BT_COUNT; ++i) {
        std::vector<double> v_time;
        std::vector<long long> v_calls;
        std::vector<uint32_t> v_inv, v_pat, v_rem, v_fil;
        
        v_time.reserve(n); v_calls.reserve(n); v_inv.reserve(n); 
        v_pat.reserve(n); v_rem.reserve(n); v_fil.reserve(n);

        for (const auto& run : res.runs_data) {
            const auto& tech = run.techniques[i];
            v_time.push_back(tech.time_us);
            v_calls.push_back(tech.calls);
            v_inv.push_back(tech.invoked);
            v_pat.push_back(tech.patterns);
            v_rem.push_back(tech.removed);
            v_fil.push_back(tech.filled);
        }

        res.technique_medians[i].time_us = calculate_median(v_time);
        res.technique_medians[i].calls = static_cast<long long>(calculate_median(v_calls));
        res.technique_medians[i].invoked = static_cast<uint32_t>(calculate_median(v_inv));
        res.technique_medians[i].patterns = static_cast<uint32_t>(calculate_median(v_pat));
        res.technique_medians[i].removed = static_cast<uint32_t>(calculate_median(v_rem));
        res.technique_medians[i].filled = static_cast<uint32_t>(calculate_median(v_fil));
    }
}

// ==========================================
// MAIN UI
// ==========================================
int main() {
    std::vector<EngineConfig> engines;
    int warmups = 5, runs = 100;

    std::cout << "========================================\n";
    std::cout << " SUDOKU ENGINE BENCHMARK SYSTEM v2.2\n";
    std::cout << "========================================\n\n";

    int engine_num = 1;
    while (true) {
        std::cout << "Masukkan file executable engine contoh ./main ./engine, atau ./main_ver #" << engine_num << ": ";
        std::string path; std::cin >> path;
        engines.push_back({path, ""});
        std::cout << "Tambahkan engine lain? (y/n): ";
        char choice; std::cin >> choice;
        if (choice != 'y' && choice != 'Y') break;
        engine_num++;
    }

    std::cout << "\n";
    for (size_t i = 0; i < engines.size(); ++i) {
        std::cout << "Masukkan data puzzle untuk Engine #" << (i + 1) << " (" << engines[i].path << "): ";
        std::cin >> engines[i].data_file;
    }

    std::cout << "\nJumlah warm-up: "; std::cin >> warmups;
    std::cout << "Jumlah benchmark runs: "; std::cin >> runs;
    std::cout << "\nMulai benchmark? (ketik 'continue'): ";
    std::string start_cmd; std::cin >> start_cmd;
    if (start_cmd != "continue") return 0;

    std::vector<BenchmarkResult> results;
    for (size_t i = 0; i < engines.size(); ++i) {
        std::cout << "\n--- Memulai Engine: " << engines[i].path << " ---\n";
        BenchmarkResult res = run_engine(engines[i].path, engines[i].data_file, warmups, runs);
        calculate_stats(res);
        results.push_back(res);
    }

    std::cout << "\n\n========================================\n";
    std::cout << "         HASIL ANALISIS AKHIR           \n";
    std::cout << "========================================\n";

    for (const auto& res : results) {
        std::cout << "\nENGINE: " << res.engine_name << "\n";
        std::cout << "DATA  : " << res.data_file << "\n";
        std::cout << "──────────────────────────\n";
        std::cout << "Runs       : " << runs << "\n";
        std::cout << "Success    : " << res.run_success << "\n";
        std::cout << "Failed     : " << res.run_failure << "\n";
        
        if (!res.runs_data.empty()) {
            std::cout << "\n=== PERFORMANCE ===\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Minimum    : " << res.min_time << " ms\n";
            std::cout << "Median     : " << res.median_time << " ms  <-- (Indikator Utama)\n";
            std::cout << "Mean       : " << res.mean_time << " ms\n";
            std::cout << "Maximum    : " << res.max_time << " ms\n";
            std::cout << "Std Dev    : " << res.stddev_time << " ms\n";
            
            std::cout << "\n=== CORE SOLVER WORKLOAD ===\n";
            std::cout << "Propagasi  : " << res.median_propagation << "  (Median)\n";
            std::cout << "Eliminasi  : " << res.median_elimination << " (Median)\n";
            std::cout << "Tebakan    : " << res.median_guesses << "   (Median)\n";
            std::cout << "Deadend    : " << res.median_deadends << "   (Median)\n";
            std::cout << "Retry      : " << res.median_retries << "   (Median)\n";
            
            std::cout << "\n=== TECHNIQUE IMPACT (MEDIAN PER SOLVE) ===\n";
            for (int i = 0; i < BT_COUNT; ++i) {
                const auto& tech = res.technique_medians[i];
                std::cout << "\n" << BENCHMARK_TECH_NAMES[i] << "\n";
                std::cout << "  Time       : " << std::fixed << std::setprecision(3) << tech.time_us << " µs\n";
                std::cout << "  Calls      : " << tech.calls << "\n";
                std::cout << "  Invoked    : " << tech.invoked << "\n";
                std::cout << "  Patterns   : " << tech.patterns << "\n";
                std::cout << "  Removed    : " << tech.removed << "\n";
                std::cout << "  Filled     : " << tech.filled << "\n";
            }
        } else {
            std::cout << "\n[GAGAL] Tidak ada waktu solve yang valid terekam.\n";
        }
    }
    std::cout << "\n========================================\n";
    return 0;
}
