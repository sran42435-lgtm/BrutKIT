#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <x86intrin.h>
#endif
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cctype>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <cstdio>
#include <algorithm>
#include <array>
#include <queue>
#include <cstring>
#include <future>
#include <thread>
#include <atomic>

using namespace std;

// ==========================================
// STATISTICS: Enum & Array
// ==========================================
enum TechniqueId {
    TECH_NAKED_SINGLE,
    TECH_HIDDEN_SINGLE_ROW,
    TECH_HIDDEN_SINGLE_COL,
    TECH_HIDDEN_SINGLE_BOX,
    TECH_LOCKED_CLAIMING,
    TECH_LOCKED_POINTING,
    TECH_NAKED_PAIR,
    TECH_NAKED_TRIPLE,
    TECH_NAKED_QUAD,
    TECH_X_WING,
    TECH_BACKTRACKING,
    TECH_COUNT
};

constexpr const char* TECH_NAMES[] = {
    "Naked Single", "Hidden Single (Baris)", "Hidden Single (Kolom)", "Hidden Single (Box)",
    "Locked Claiming", "Locked Pointing",
    "Naked Pair", "Naked Triple", "Naked Quad",
    "X-Wing", "Backtracking (LCV)"
};

struct TechniqueStats {
    uint32_t invoked = 0;
    uint32_t patterns = 0;
    uint32_t removed = 0;
    uint32_t filled = 0;
};

struct TechniqueTiming {
    double naked_single_us = 0;
    double hidden_single_us = 0;
    double naked_subsets_us = 0;
    double locked_candidates_us = 0;
    double xwing_us = 0;

    long long naked_single_calls = 0;
    long long hidden_single_calls = 0;
    long long naked_subsets_calls = 0;
    long long locked_candidates_calls = 0;
    long long xwing_calls = 0;
};

// ==========================================
// T0: Timer Nol-Biaya & Dispatch
// ==========================================
static inline uint64_t tick() {
#if defined(__x86_64__) || defined(_M_X64)
    
    return __builtin_ia32_rdtsc();
#elif defined(__aarch64__)
    uint64_t v; asm volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
#else
    return (uint64_t)chrono::steady_clock::now().time_since_epoch().count();
#endif
}
static double g_us_per_tick = 0.0;
static void calibrateTicks() {
    auto t0 = chrono::steady_clock::now(); uint64_t a = tick();
    while (chrono::steady_clock::now() - t0 < chrono::milliseconds(5)) {}
    uint64_t b = tick(); auto t1 = chrono::steady_clock::now();
    g_us_per_tick = chrono::duration<double, micro>(t1 - t0).count() / (double)(b - a);
}

template <bool PROF> struct Timer {
    template <class F> static inline double run(F&& f) {
        if constexpr (!PROF) { f(); return 0.0; }
        else { uint64_t a = tick(); f(); return (double)(tick() - a) * g_us_per_tick; }
    }
};

// ==========================================
// STATIC TABLES
// ==========================================
array<array<int, 20>, 81> PEERS;
array<array<int, 9>, 27> UNITS;
array<array<int, 3>, 81> CELL_UNITS;
array<array<int, 81>, 27> LOCAL_POS;

void initStaticTables() {
    for (int i = 0; i < 81; i++) {
        int r = i / 9, c = i % 9;
        int sr = (r / 3) * 3, sc = (c / 3) * 3;
        int idx = 0;
        for (int cc = 0; cc < 9; cc++) if (cc != c) PEERS[i][idx++] = r * 9 + cc;
        for (int rr = 0; rr < 9; rr++) if (rr != r) PEERS[i][idx++] = rr * 9 + c;
        for (int rr = sr; rr < sr + 3; rr++)
            for (int cc = sc; cc < sc + 3; cc++)
                if (rr * 9 + cc != i && rr != r && cc != c)
                    PEERS[i][idx++] = rr * 9 + cc;
    }
    int u = 0;
    for (int r = 0; r < 9; r++) {
        for (int c = 0; c < 9; c++) UNITS[u][c] = r * 9 + c;
        u++;
    }
    for (int c = 0; c < 9; c++) {
        for (int r = 0; r < 9; r++) UNITS[u][r] = r * 9 + c;
        u++;
    }
    for (int br = 0; br < 3; br++) {
        for (int bc = 0; bc < 3; bc++) {
            int idx = 0;
            for (int r = br * 3; r < br * 3 + 3; r++)
                for (int c = bc * 3; c < bc * 3 + 3; c++)
                    UNITS[u][idx++] = r * 9 + c;
            u++;
        }
    }
    for (int i = 0; i < 81; i++) {
        int count = 0;
        for (int uu = 0; uu < 27 && count < 3; uu++) {
            for (int j = 0; j < 9; j++) {
                if (UNITS[uu][j] == i) {
                    CELL_UNITS[i][count++] = uu;
                    break;
                }
            }
        }
    }
    for (int uu = 0; uu < 27; uu++) {
        for (int i = 0; i < 81; i++) LOCAL_POS[uu][i] = -1;
        for (int i = 0; i < 9; i++) LOCAL_POS[uu][UNITS[uu][i]] = i;
    }
}

// ==========================================
// T1: State POD, XwFifo, LogStack, Counters
// ==========================================
struct alignas(64) State {
    uint16_t cand[96];         // 81 dipakai, padding SIMD
    uint8_t  val[96];          // 0 = kosong, 1..9 = terisi
    uint16_t udp[10][32];      // [digit][unit]; bits 0..8 = posisi lokal; bit 15 = digit terpasang
    uint8_t  emptyInUnit[32];  // sel kosong per unit
};

struct XwFifo {
    uint8_t  q[256];
    uint16_t dirty[18] = {};
    uint32_t h = 0, t = 0;
    inline void mark(int line, int num) {
        uint16_t b = (uint16_t)(1u << num);
        if (!(dirty[line] & b)) { 
            dirty[line] |= b; 
            q[t++ & 255] = (uint8_t)(line * 10 + num); 
        }
    }
    inline bool empty() const { return h == t; }
    inline void pop(int& line, int& num) {
        uint8_t c = q[h++ & 255]; 
        line = c / 10; 
        num = c % 10;
        dirty[line] &= (uint16_t)~(1u << num);
    }
};

struct Shared {
    XwFifo xwing;
    uint16_t changedDigits[27] = {};
};

struct LangkahLog {
    int step;
    int8_t x, y;
    char old_val, new_val;
    uint8_t tech;
};

struct LogStack {
    LangkahLog data[4096];
    int top = 0;
    void push(const LangkahLog& l) { data[top++] = l; }
    void pop() { top--; }
    void truncate(int mark) { top = mark; }
    int size() const { return top; }
    int mark() const { return top; }
};

LogStack g_logStack;

struct Counters {
    int jumlah_solusi = 0;
    string solusi_pertama = "";
    vector<string> semua_solusi;
    int langkah_propagasi = 0;
    int langkah_tebakan = 0;
    int deadend = 0;
    int retry = 0;
    int total_elim = 0;
    array<TechniqueStats, TECH_COUNT> teknik_stats{};
    TechniqueTiming timing{};

    void add(const Counters& other) {
        jumlah_solusi += other.jumlah_solusi;
        langkah_propagasi += other.langkah_propagasi;
        langkah_tebakan += other.langkah_tebakan;
        deadend += other.deadend;
        retry += other.retry;
        total_elim += other.total_elim;
        for (int i = 0; i < TECH_COUNT; i++) {
            teknik_stats[i].invoked += other.teknik_stats[i].invoked;
            teknik_stats[i].patterns += other.teknik_stats[i].patterns;
            teknik_stats[i].removed += other.teknik_stats[i].removed;
            teknik_stats[i].filled += other.teknik_stats[i].filled;
        }
        timing.naked_single_us += other.timing.naked_single_us;
        timing.hidden_single_us += other.timing.hidden_single_us;
        timing.naked_subsets_us += other.timing.naked_subsets_us;
        timing.locked_candidates_us += other.timing.locked_candidates_us;
        timing.xwing_us += other.timing.xwing_us;
        timing.naked_single_calls += other.timing.naked_single_calls;
        timing.hidden_single_calls += other.timing.hidden_single_calls;
        timing.naked_subsets_calls += other.timing.naked_subsets_calls;
        timing.locked_candidates_calls += other.timing.locked_candidates_calls;
        timing.xwing_calls += other.timing.xwing_calls;
    }
};

// ==========================================
// KOORDINAT & UTILITAS
// ==========================================
int kartesiusToIndex(int x, int y) {
    int row = 4 - y, col = x + 4;
    if (row < 0 || row > 8 || col < 0 || col > 8) return -1;
    return row * 9 + col;
}
pair<int,int> indexToKartesius(int idx) { return {idx % 9 - 4, 4 - idx / 9}; }
char bacaKoordinat(const string& papan, int x, int y) {
    int idx = kartesiusToIndex(x, y);
    return idx == -1 ? '?' : papan[idx];
}
void cetakPapanVisual(const string& papan) {
    cout << "+-------+-------+-------+" << endl;
    for (int r = 0; r < 9; r++) {
        cout << "| ";
        for (int c = 0; c < 9; c++) {
            char val = papan[r * 9 + c];
            cout << (val == '0' ? ". " : string(1, val) + " ");
            if ((c + 1) % 3 == 0) cout << "| ";
        }
        cout << endl;
        if ((r + 1) % 3 == 0) cout << "+-------+-------+-------+" << endl;
    }
}
void clearScreen() { cout << "\033[2J\033[3J\033[H" << flush; }

bool validasiPetunjukAwal(const string& papan, vector<string>& konflik) {
    bool valid = true;
    konflik.clear();
    for (int i = 0; i < 81; i++) {
        if (papan[i] == '0') continue;
        char num = papan[i];
        int r = i / 9, c = i % 9;
        for (int cc = 0; cc < 9; cc++) {
            if (cc != c && papan[r * 9 + cc] == num) {
                konflik.push_back("Baris " + to_string(4 - r) + " duplikat " + string(1, num));
                valid = false;
            }
        }
        for (int rr = 0; rr < 9; rr++) {
            if (rr != r && papan[rr * 9 + c] == num) {
                konflik.push_back("Kolom " + to_string(c - 4) + " duplikat " + string(1, num));
                valid = false;
            }
        }
        int sr = (r / 3) * 3, sc = (c / 3) * 3;
        for (int rr = sr; rr < sr + 3; rr++)
            for (int cc = sc; cc < sc + 3; cc++)
                if ((rr != r || cc != c) && papan[rr * 9 + cc] == num) {
                    konflik.push_back("Box 3x3 duplikat " + string(1, num));
                    valid = false;
                }
    }
    return valid;
}

// ==========================================
// T2: Inisialisasi State
// ==========================================
void initState(State& s, const string& papan) {
    memset(&s, 0, sizeof(s));
    for (int i = 0; i < 81; i++) {
        if (papan[i] == '0') {
            s.cand[i] = 0x3FE;
            s.val[i] = 0;
            for (int u_idx = 0; u_idx < 3; u_idx++) s.emptyInUnit[CELL_UNITS[i][u_idx]]++;
        } else {
            s.val[i] = papan[i] - '0';
            s.cand[i] = 0;
        }
    }
    for (int d = 1; d <= 9; d++) {
        for (int u = 0; u < 27; u++) {
            uint16_t pos = 0;
            bool placed = false;
            for (int i = 0; i < 9; i++) {
                int idx = UNITS[u][i];
                if (s.val[idx] == d) placed = true;
                else if (s.cand[idx] & (1u << d)) pos |= (1u << i);
            }
            s.udp[d][u] = pos | (placed ? 0x8000 : 0);
        }
    }
}

// ==========================================
// T3: Hapus Kandidat (Inline & Gate)
// ==========================================
template <bool DIRTY>
inline bool removeCandidate(State& s, int idx, int num, Shared& sh, Counters& out, 
                            uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, 
                            int& step_counter, TechniqueId tech) {
    if (!(s.cand[idx] & (1u << num))) return false;
    
    s.cand[idx] &= ~(1u << num);
    out.total_elim++;
    out.teknik_stats[tech].removed++;
    
    for (int i = 0; i < 3; i++) {
        int u = CELL_UNITS[idx][i];
        if constexpr (DIRTY) {
            dirtyUnits |= (1u << u);
            lockedDirtyUnits |= (1u << u);
        }
        sh.changedDigits[u] |= (1u << num);
        s.udp[num][u] &= ~(1u << LOCAL_POS[u][idx]);
    }
    
    sh.xwing.mark(idx / 9, num);
    sh.xwing.mark(idx % 9 + 9, num);
    
    return true;
}

// ==========================================
// T2 & T3: Fill Core (Unified)
// ==========================================
template <bool LOG, bool DIRTY, bool TEBAKAN>
void fillCore(State& s, int idx, int num, Shared& sh, Counters& out, int& step_counter, TechniqueId tech, uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits) {
    uint16_t oldMask = s.cand[idx];
    
    if constexpr (LOG) {
        auto [x, y] = indexToKartesius(idx);
        g_logStack.push({++step_counter, (int8_t)x, (int8_t)y, '0', (char)('0' + num), (uint8_t)tech});
        out.teknik_stats[tech].filled++;
    }
    
    s.val[idx] = num;
    s.cand[idx] = 0;
    for (int i = 0; i < 3; i++) s.emptyInUnit[CELL_UNITS[idx][i]]--;
    
    for (int i = 0; i < 3; i++) {
        int u = CELL_UNITS[idx][i];
        if constexpr (DIRTY) {
            dirtyUnits |= (1u << u);
            lockedDirtyUnits |= (1u << u);
        }
        int localPos = LOCAL_POS[u][idx];
        uint16_t m = oldMask;
        while (m) {
            int d = __builtin_ctz(m);
            s.udp[d][u] &= ~(1u << localPos);
            m &= m - 1;
        }
        s.udp[num][u] |= 0x8000;
    }
    
    if constexpr (TEBAKAN) {
        for (int i = 0; i < 3; i++) sh.changedDigits[CELL_UNITS[idx][i]] |= oldMask;
    }
    
    for (int d = 1; d <= 9; d++) {
        if (oldMask & (1u << d)) {
            sh.xwing.mark(idx / 9, d);
            sh.xwing.mark(idx % 9 + 9, d);
        }
    }
    
    for (int p = 0; p < 20; p++) {
        int peer = PEERS[idx][p];
        if (s.cand[peer] & (1u << num)) {
            s.cand[peer] &= ~(1u << num);
            out.total_elim++;
            for (int i = 0; i < 3; i++) {
                int u = CELL_UNITS[peer][i];
                if constexpr (DIRTY) {
                    dirtyUnits |= (1u << u);
                    lockedDirtyUnits |= (1u << u);
                }
                sh.changedDigits[u] |= (1u << num);
                s.udp[num][u] &= ~(1u << LOCAL_POS[u][peer]);
            }
            sh.xwing.mark(peer / 9, num);
            sh.xwing.mark(peer % 9 + 9, num);
        }
    }
}

// ==========================================
// T3: Naked Subsets
// ==========================================
void prosesNakedSubsetsUnit(int u, State& s, Shared& sh, Counters& out, 
                            uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, int& step_counter) {
    int pairCells[9], tripleCells[9], quadCells[9];
    int c2 = 0, c3 = 0, c4 = 0;
    for (int i = 0; i < 9; i++) {
        int idx = UNITS[u][i];
        if (s.val[idx] != 0) continue;
        int cnt = __builtin_popcount(s.cand[idx]);
        if (cnt == 2) { pairCells[c2++] = idx; tripleCells[c3++] = idx; quadCells[c4++] = idx; }
        else if (cnt == 3) { tripleCells[c3++] = idx; quadCells[c4++] = idx; }
        else if (cnt == 4) { quadCells[c4++] = idx; }
    }
    
    if (c2 >= 2) {
        out.teknik_stats[TECH_NAKED_PAIR].invoked++;
        for (int i = 0; i < c2; i++) {
            for (int j = i + 1; j < c2; j++) {
                uint16_t um = s.cand[pairCells[i]] | s.cand[pairCells[j]];
                if (__builtin_popcount(um) == 2) {
                    out.teknik_stats[TECH_NAKED_PAIR].patterns++;
                    for (int k = 0; k < 9; k++) {
                        int idx = UNITS[u][k];
                        if (idx != pairCells[i] && idx != pairCells[j]) {
                            uint16_t common = s.cand[idx] & um;
                            while (common) {
                                int num = __builtin_ctz(common);
                                removeCandidate<true>(s, idx, num, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_NAKED_PAIR);
                                common &= common - 1;
                            }
                        }
                    }
                }
            }
        }
    }
    if (c3 >= 3) {
        out.teknik_stats[TECH_NAKED_TRIPLE].invoked++;
        for (int i = 0; i < c3; i++) {
            for (int j = i + 1; j < c3; j++) {
                for (int k = j + 1; k < c3; k++) {
                    uint16_t um = s.cand[tripleCells[i]] | s.cand[tripleCells[j]] | s.cand[tripleCells[k]];
                    if (__builtin_popcount(um) == 3) {
                        out.teknik_stats[TECH_NAKED_TRIPLE].patterns++;
                        for (int m = 0; m < 9; m++) {
                            int idx = UNITS[u][m];
                            if (idx != tripleCells[i] && idx != tripleCells[j] && idx != tripleCells[k]) {
                                uint16_t common = s.cand[idx] & um;
                                while (common) {
                                    int num = __builtin_ctz(common);
                                    removeCandidate<true>(s, idx, num, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_NAKED_TRIPLE);
                                    common &= common - 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (c4 >= 4) {
        out.teknik_stats[TECH_NAKED_QUAD].invoked++;
        for (int i = 0; i < c4; i++) {
            for (int j = i + 1; j < c4; j++) {
                for (int k = j + 1; k < c4; k++) {
                    for (int l = k + 1; l < c4; l++) {
                        uint16_t um = s.cand[quadCells[i]] | s.cand[quadCells[j]] | 
                                      s.cand[quadCells[k]] | s.cand[quadCells[l]];
                        if (__builtin_popcount(um) == 4) {
                            out.teknik_stats[TECH_NAKED_QUAD].patterns++;
                            for (int m = 0; m < 9; m++) {
                                int idx = UNITS[u][m];
                                if (idx != quadCells[i] && idx != quadCells[j] && 
                                    idx != quadCells[k] && idx != quadCells[l]) {
                                    uint16_t common = s.cand[idx] & um;
                                    while (common) {
                                        int num = __builtin_ctz(common);
                                        removeCandidate<true>(s, idx, num, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_NAKED_QUAD);
                                        common &= common - 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ==========================================
// T3: Locked Candidates
// ==========================================
bool prosesLockedPadaUnit(int u, State& s, Shared& sh, Counters& out,
                          uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, int& step_counter) {
    bool changed = false;
    if (u >= 18) {
        int box = u - 18;
        int br = box / 3;
        int bc = box % 3;
        for (int d = 1; d <= 9; d++) {
            out.teknik_stats[TECH_LOCKED_POINTING].invoked++;
            uint16_t pos = s.udp[d][u] & 0x1FF;
            if (__builtin_popcount(pos) < 2) continue;
            
            if ((pos & 0x007) == pos) {
                out.teknik_stats[TECH_LOCKED_POINTING].patterns++;
                int r = br * 3 + 0;
                for (int c = 0; c < 9; c++) {
                    if (c / 3 == bc) continue;
                    if (removeCandidate<true>(s, r * 9 + c, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_POINTING)) changed = true;
                }
            } else if ((pos & 0x038) == pos) {
                out.teknik_stats[TECH_LOCKED_POINTING].patterns++;
                int r = br * 3 + 1;
                for (int c = 0; c < 9; c++) {
                    if (c / 3 == bc) continue;
                    if (removeCandidate<true>(s, r * 9 + c, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_POINTING)) changed = true;
                }
            } else if ((pos & 0x1C0) == pos) {
                out.teknik_stats[TECH_LOCKED_POINTING].patterns++;
                int r = br * 3 + 2;
                for (int c = 0; c < 9; c++) {
                    if (c / 3 == bc) continue;
                    if (removeCandidate<true>(s, r * 9 + c, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_POINTING)) changed = true;
                }
            }
            if ((pos & 0x049) == pos) {
                out.teknik_stats[TECH_LOCKED_POINTING].patterns++;
                int c = bc * 3 + 0;
                for (int r = 0; r < 9; r++) {
                    if (r / 3 == br) continue;
                    if (removeCandidate<true>(s, r * 9 + c, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_POINTING)) changed = true;
                }
            } else if ((pos & 0x092) == pos) {
                out.teknik_stats[TECH_LOCKED_POINTING].patterns++;
                int c = bc * 3 + 1;
                for (int r = 0; r < 9; r++) {
                    if (r / 3 == br) continue;
                    if (removeCandidate<true>(s, r * 9 + c, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_POINTING)) changed = true;
                }
            } else if ((pos & 0x124) == pos) {
                out.teknik_stats[TECH_LOCKED_POINTING].patterns++;
                int c = bc * 3 + 2;
                for (int r = 0; r < 9; r++) {
                    if (r / 3 == br) continue;
                    if (removeCandidate<true>(s, r * 9 + c, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_POINTING)) changed = true;
                }
            }
        }
    } else {
        bool isRow = u < 9;
        int unitIndex = isRow ? u : u - 9;
        for (int d = 1; d <= 9; d++) {
            out.teknik_stats[TECH_LOCKED_CLAIMING].invoked++;
            uint16_t pos = s.udp[d][u] & 0x1FF;
            if (__builtin_popcount(pos) < 2) continue;
            
            if ((pos & 0x007) == pos || (pos & 0x038) == pos || (pos & 0x1C0) == pos) {
                out.teknik_stats[TECH_LOCKED_CLAIMING].patterns++;
                int box_col = -1;
                if ((pos & 0x007) == pos) box_col = 0;
                else if ((pos & 0x038) == pos) box_col = 1;
                else box_col = 2;
                
                if (isRow) {
                    int br = (unitIndex / 3) * 3;
                    int bc = box_col * 3;
                    for (int rr = br; rr < br + 3; rr++) {
                        if (rr == unitIndex) continue;
                        for (int cc = bc; cc < bc + 3; cc++) {
                            if (removeCandidate<true>(s, rr * 9 + cc, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_CLAIMING)) changed = true;
                        }
                    }
                } else {
                    int bc = (unitIndex / 3) * 3;
                    int box_row = box_col * 3;
                    for (int cc = bc; cc < bc + 3; cc++) {
                        if (cc == unitIndex) continue;
                        for (int rr = box_row; rr < box_row + 3; rr++) {
                            if (removeCandidate<true>(s, rr * 9 + cc, d, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_LOCKED_CLAIMING)) changed = true;
                        }
                    }
                }
            }
        }
    }
    return changed;
}

// ==========================================
// X-Wing
// ==========================================
bool prosesXWingBaris(int row, int num, State& s, Shared& sh, Counters& out,
                      uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, int& step_counter) {
    out.teknik_stats[TECH_X_WING].invoked++;
    uint16_t pos = s.udp[num][row] & 0x1FF;
    if (__builtin_popcount(pos) != 2) return false;
    
    bool changed = false;
    for (int other = 0; other < 9; other++) {
        if (other == row) continue;
        if ((s.udp[num][other] & 0x1FF) == pos) {
            out.teknik_stats[TECH_X_WING].patterns++;
            int c1 = __builtin_ctz(pos);
            int c2 = __builtin_ctz(pos & (pos - 1));
            for (int r = 0; r < 9; r++) {
                if (r == row || r == other) continue;
                if (removeCandidate<true>(s, r * 9 + c1, num, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_X_WING)) changed = true;
                if (removeCandidate<true>(s, r * 9 + c2, num, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_X_WING)) changed = true;
            }
        }
    }
    return changed;
}

bool prosesXWingKolom(int col, int num, State& s, Shared& sh, Counters& out,
                      uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, int& step_counter) {
    out.teknik_stats[TECH_X_WING].invoked++;
    uint16_t pos = s.udp[num][col + 9] & 0x1FF;
    if (__builtin_popcount(pos) != 2) return false;
    
    bool changed = false;
    for (int other = 0; other < 9; other++) {
        if (other == col) continue;
        if ((s.udp[num][other + 9] & 0x1FF) == pos) {
            out.teknik_stats[TECH_X_WING].patterns++;
            int r1 = __builtin_ctz(pos);
            int r2 = __builtin_ctz(pos & (pos - 1));
            for (int c = 0; c < 9; c++) {
                if (c == col || c == other) continue;
                if (removeCandidate<true>(s, r1 * 9 + c, num, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_X_WING)) changed = true;
                if (removeCandidate<true>(s, r2 * 9 + c, num, sh, out, dirtyUnits, lockedDirtyUnits, step_counter, TECH_X_WING)) changed = true;
            }
        }
    }
    return changed;
}

// ==========================================
// PROPAGASI INKREMENTAL
// ==========================================
template <bool PROF>
bool propagate(State& s, Shared& sh, Counters& out, int changedCell, int& step_counter) {
    uint32_t dirtyUnits = 0;
    uint32_t lockedDirtyUnits = 0;
    
    if (changedCell == -1) {
        dirtyUnits = 0x07FFFFFF;
        lockedDirtyUnits = 0x07FFFFFF;
        for (int u = 0; u < 27; u++) sh.changedDigits[u] = 0x3FE;
        for (int i = 0; i < 81; i++) {
            if (s.val[i] != 0) {
                sh.xwing.mark(i / 9, s.val[i]);
                sh.xwing.mark(i % 9 + 9, s.val[i]);
            } else {
                uint16_t m = s.cand[i];
                while (m) {
                    int num = __builtin_ctz(m);
                    sh.xwing.mark(i / 9, num);
                    sh.xwing.mark(i % 9 + 9, num);
                    m &= m - 1;
                }
            }
        }
    } else {
        for (int i = 0; i < 3; i++) {
            int u = CELL_UNITS[changedCell][i];
            dirtyUnits |= (1u << u);
            lockedDirtyUnits |= (1u << u);
        }
    }

    while (true) {
        while (dirtyUnits != 0) {
            int u = __builtin_ctz(dirtyUnits);
            dirtyUnits &= dirtyUnits - 1;

            out.timing.naked_single_us += Timer<PROF>::run([&]() {
                for (int i = 0; i < 9; i++) {
                    int idx = UNITS[u][i];
                    if (s.val[idx] == 0 && __builtin_popcount(s.cand[idx]) == 1) {
                        int num = __builtin_ctz(s.cand[idx]);
                        out.teknik_stats[TECH_NAKED_SINGLE].invoked++;
                        fillCore<true, true, false>(s, idx, num, sh, out, step_counter, TECH_NAKED_SINGLE, dirtyUnits, lockedDirtyUnits);
                    }
                }
            });
            out.timing.naked_single_calls++;

            out.timing.hidden_single_us += Timer<PROF>::run([&]() {
                uint16_t digits = sh.changedDigits[u];
                sh.changedDigits[u] = 0;
                while (digits) {
                    int d = __builtin_ctz(digits);
                    digits &= digits - 1;
                    uint16_t pos = s.udp[d][u];
                    if (!(pos & 0x8000) && (pos & (pos - 1)) == 0 && pos != 0) {
                        int localPos = __builtin_ctz(pos);
                        int target = UNITS[u][localPos];
                        TechniqueId tech = (u < 9) ? TECH_HIDDEN_SINGLE_ROW : (u < 18) ? TECH_HIDDEN_SINGLE_COL : TECH_HIDDEN_SINGLE_BOX;
                        out.teknik_stats[tech].invoked++;
                        fillCore<true, true, false>(s, target, d, sh, out, step_counter, tech, dirtyUnits, lockedDirtyUnits);
                    }
                }
            });
            out.timing.hidden_single_calls++;

            out.timing.naked_subsets_us += Timer<PROF>::run([&]() {
                prosesNakedSubsetsUnit(u, s, sh, out, dirtyUnits, lockedDirtyUnits, step_counter);
            });
            out.timing.naked_subsets_calls++;

            for (int d = 1; d <= 9; d++) {
                if (s.udp[d][u] == 0) return false;
            }
            for (int i = 0; i < 9; i++) {
                int idx = UNITS[u][i];
                if (s.val[idx] == 0 && s.cand[idx] == 0) return false;
            }
        }

        bool globalChanged = false;

        out.timing.locked_candidates_us += Timer<PROF>::run([&]() {
            while (lockedDirtyUnits != 0) {
                int u = __builtin_ctz(lockedDirtyUnits);
                lockedDirtyUnits &= lockedDirtyUnits - 1;
                if (s.emptyInUnit[u] < 2) continue;
                if (prosesLockedPadaUnit(u, s, sh, out, dirtyUnits, lockedDirtyUnits, step_counter))
                    globalChanged = true;
            }
        });
        out.timing.locked_candidates_calls++;

        out.timing.xwing_us += Timer<PROF>::run([&]() {
            while (!sh.xwing.empty()) {
                int line, num;
                sh.xwing.pop(line, num);
                if (line < 9) {
                    if (prosesXWingBaris(line, num, s, sh, out, dirtyUnits, lockedDirtyUnits, step_counter))
                        globalChanged = true;
                } else {
                    if (prosesXWingKolom(line - 9, num, s, sh, out, dirtyUnits, lockedDirtyUnits, step_counter))
                        globalChanged = true;
                }
            }
        });
        out.timing.xwing_calls++;

        if (!globalChanged) break;
    }
    return true;
}

// ==========================================
// T4: MRV dan LCV via popcount & vector-min
// ==========================================
int cariSelMRV(const State& s) {
    int min_k = 10, best_idx = -1, max_deg = -1;
    for (int i = 0; i < 81; i++) {
        if (s.val[i] == 0) {
            int jml = __builtin_popcount(s.cand[i]);
            if (jml == 0) return -2;
            if (jml < min_k) { min_k = jml; best_idx = i; max_deg = -1; }
            if (jml == min_k) {
                int deg = 0;
                for (int p = 0; p < 20; p++)
                    if (s.val[PEERS[i][p]] == 0) deg++;
                if (deg > max_deg) { max_deg = deg; best_idx = i; }
            }
        }
    }
    return best_idx;
}

int hitungEliminasiLCV(const State& s, int idx, int num) {
    int eliminasi = 0;
    for (int p = 0; p < 20; p++)
        if (s.cand[PEERS[idx][p]] & (1u << num)) eliminasi++;
    return eliminasi;
}

// ==========================================
// T2 & T5: Search Engine (Copy-on-Branch)
// ==========================================
template <bool PROF, bool PARALLEL>
void search(const State& parent, Shared& sh, Counters& out, int changedCell, int step_counter, bool is_root = false) {
    State s = parent;
    int logMark = g_logStack.mark();
    int savedStep = step_counter;
    
    if (!propagate<PROF>(s, sh, out, changedCell, step_counter)) {
        g_logStack.truncate(logMark);
        out.deadend = 1;
        return;
    }
    
    bool selesai = true;
    for (int i = 0; i < 81; i++) if (s.val[i] == 0) { selesai = false; break; }
    if (selesai) {
        out.jumlah_solusi = 1;
        out.solusi_pertama = "";
        for (int i = 0; i < 81; i++) out.solusi_pertama += (char)('0' + s.val[i]);
        out.semua_solusi.push_back(out.solusi_pertama);
        
        vector<LangkahLog> tempLog;
        for(int i=0; i<g_logStack.size(); i++) tempLog.push_back(g_logStack.data[i]);
        // Log disimpan secara global sementara untuk solusi pertama
        g_logStack.truncate(logMark);
        return;
    }
    
    int idx = cariSelMRV(s);
    if (idx == -2) {
        g_logStack.truncate(logMark);
        out.deadend = 1;
        return;
    }
    
    vector<pair<int, int>> kandidat_lcv;
    for (int num = 1; num <= 9; num++)
        if (s.cand[idx] & (1u << num))
            kandidat_lcv.push_back({num, hitungEliminasiLCV(s, idx, num)});
    sort(kandidat_lcv.begin(), kandidat_lcv.end(),
         [](const pair<int,int>& a, const pair<int,int>& b) { return a.second < b.second; });
         
    int attempts = 0;
    auto processBranch = [&](int num, int score, int branch_idx) {
        if (branch_idx > 0) out.retry++;
        out.langkah_tebakan++;
        
        State child = s;
        Shared local_sh = sh; // Kebocoran dipertahankan jika tidak PARALLEL. Untuk PARALLEL kita copy agar aman thread.
        
        // Jika PARALLEL dan di root, kita isolasi Shared. Jika tidak, biarkan bocor (Tier A behavior).
        if constexpr (!PARALLEL) {
            // Gunakan sh asli agar bocor (Tier A strict)
            auto [x, y] = indexToKartesius(idx);
            g_logStack.push({step_counter + branch_idx + 1, (int8_t)x, (int8_t)y, '0', (char)('0' + num), (uint8_t)TECH_BACKTRACKING});
            
            uint32_t dummyD = 0, dummyL = 0;
            fillCore<false, false, true>(child, idx, num, sh, out, step_counter, TECH_BACKTRACKING, dummyD, dummyL);
            
            Counters sub{};
            search<PROF, false>(child, sh, sub, idx, step_counter + branch_idx + 1);
            out.add(sub);
        } else {
            Shared isolated_sh = sh;
            Counters sub{};
            int sub_step = step_counter + branch_idx + 1;
            int sub_logMark = g_logStack.mark(); // Aman karena thread pool hanya di root
            auto [x, y] = indexToKartesius(idx);
            g_logStack.push({sub_step, (int8_t)x, (int8_t)y, '0', (char)('0' + num), (uint8_t)TECH_BACKTRACKING});
            
            uint32_t dummyD = 0, dummyL = 0;
            fillCore<false, false, true>(child, idx, num, isolated_sh, sub, step_counter, TECH_BACKTRACKING, dummyD, dummyL);
            
            search<PROF, false>(child, isolated_sh, sub, idx, sub_step);
            out.add(sub);
            g_logStack.truncate(sub_logMark);
        }
    };

    if constexpr (PARALLEL) {
        vector<future<void>> futures;
        for (size_t i = 0; i < kandidat_lcv.size(); i++) {
            auto [num, score] = kandidat_lcv[i];
            attempts++;
            // Tier B: Parallel execution di root
            futures.push_back(async(launch::async, [&, num, score, i]() {
                processBranch(num, score, i);
            }));
        }
        for (auto& f : futures) f.get();
    } else {
        for (size_t i = 0; i < kandidat_lcv.size(); i++) {
            auto [num, score] = kandidat_lcv[i];
            attempts++;
            processBranch(num, score, i);
        }
    }
    g_logStack.truncate(logMark);
}

// ==========================================
// Wrapper Solve Cerdas (T6)
// ==========================================
struct HasilSolver {
    int jumlah_solusi;
    string solusi_pertama;
    vector<string> semua_solusi;
    int langkah_propagasi;
    int langkah_tebakan;
    int deadend;
    int retry;
    int total_elim;
    vector<LangkahLog> langkah_pengisian;
    array<TechniqueStats, TECH_COUNT> teknik_stats;
    TechniqueTiming timing;
};

HasilSolver solveCerdas(const string& papan_awal, bool hitung_semua, bool use_profiling, bool use_parallel) {
    State s;
    initState(s, papan_awal);
    Shared sh;
    Counters out;
    g_logStack.top = 0;
    
    int step_counter = 0;
    if (use_parallel && !hitung_semua) {
        // Tier B: Paralel /solve
        search<true, true>(s, sh, out, -1, step_counter, true);
    } else if (use_parallel && hitung_semua) {
        // Tier A /count: Harus sequential agar bit-exact
        search<true, false>(s, sh, out, -1, step_counter, true);
    } else {
        if (use_profiling) search<true, false>(s, sh, out, -1, step_counter, true);
        else search<false, false>(s, sh, out, -1, step_counter, true);
    }

    HasilSolver hasil;
    hasil.jumlah_solusi = out.jumlah_solusi;
    hasil.solusi_pertama = out.solusi_pertama;
    hasil.semua_solusi = out.semua_solusi;
    hasil.langkah_propagasi = out.langkah_propagasi;
    hasil.langkah_tebakan = out.langkah_tebakan;
    hasil.deadend = out.deadend;
    hasil.retry = out.retry;
    hasil.total_elim = out.total_elim;
    hasil.teknik_stats = out.teknik_stats;
    hasil.timing = out.timing;
    
    for (int i = 0; i < g_logStack.size(); i++) {
        hasil.langkah_pengisian.push_back(g_logStack.data[i]);
    }
    return hasil;
}

// ==========================================
// MAIN (T6)
// ==========================================
string global_papan(81, '0');
string original_papan(81, '0');
string solved_papan = "";
unsigned long current_session = 0;
bool has_solve = false;
bool showing_solved = false;

void printTechniqueStats(const array<TechniqueStats, TECH_COUNT>& stats) {
    cout << "\n=== TECHNIQUE STATISTICS ===\n";
    for (int i = 0; i < TECH_COUNT; i++) {
        auto& s = stats[i];
        if (s.invoked == 0 && s.patterns == 0 && s.removed == 0 && s.filled == 0) continue;
        cout << "\n" << TECH_NAMES[i] << ":\n";
        cout << "  Invoked          : " << s.invoked << "\n";
        cout << "  Pattern Detected : " << s.patterns << "\n";
        cout << "  Candidate Removed: " << s.removed << "\n";
        cout << "  Cell Filled      : " << s.filled << "\n";
    }
}

void printTechniqueTiming(const TechniqueTiming& t) {
    cout << "\n=== TECHNIQUE TIMING ===\n";
    cout << fixed << setprecision(3);
    cout << "Naked Single: " << t.naked_single_us << " µs | Calls: " << t.naked_single_calls << '\n';
    cout << "Hidden Single: " << t.hidden_single_us << " µs | Calls: " << t.hidden_single_calls << '\n';
    cout << "Naked Subsets: " << t.naked_subsets_us << " µs | Calls: " << t.naked_subsets_calls << '\n';
    cout << "Locked Candidates: " << t.locked_candidates_us << " µs | Calls: " << t.locked_candidates_calls << '\n';
    cout << "X-Wing: " << t.xwing_us << " µs | Calls: " << t.xwing_calls << '\n';
    
    double total = t.naked_single_us + t.hidden_single_us + t.naked_subsets_us + 
                   t.locked_candidates_us + t.xwing_us;
    cout << "\nTotal Technique Time: " << total << " µs (" << (total / 1000.0) << " ms)\n";
}

int main() {
    calibrateTicks();
    initStaticTables();
    
    auto do_calculate = [&]() -> bool {
        string namaFile = "data_koordinat_" + to_string(current_session) + ".txt";
        ofstream f(namaFile);
        if (!f.is_open()) return false;
        f << "SESSION_ID: " << current_session << "\n\n";
        f << "=== DATABASE DUAL-KOORDINAT (v11.0) ===\n\n";
        int n = 0;
        State s; initState(s, global_papan);
        for (int y = 4; y >= -4; y--) {
            for (int x = -4; x <= 4; x++) {
                n++;
                int idx = kartesiusToIndex(x, y);
                char val = global_papan[idx];
                f << "TITIK " << n << " | PUSAT (" << x << "," << y << ") = [" << val << "]\n";
                f << "Horizontal: [ ";
                for (int xx = -4; xx <= 4; xx++) f << xx << "x," << y << "y:" << bacaKoordinat(global_papan, xx, y) << " ";
                f << "]\nVertikal: [ ";
                for (int yy = 4; yy >= -4; yy--) f << yy << "y," << x << "x:" << bacaKoordinat(global_papan, x, yy) << " ";
                f << "]\nKandidat: [ ";
                if (val != '0') f << val;
                else {
                    bool ada = false;
                    for (int num = 1; num <= 9; num++)
                        if (s.cand[idx] & (1u << num)) { f << num << " "; ada = true; }
                    if (!ada) f << "BUNTU";
                }
                f << " ]\n\n";
            }
        }
        f.close();
        cout << "SUKSES tersimpan ke " << namaFile << endl;
        return true;
    };

    while (true) {
        clearScreen();
        cout << "==========================================" << endl;
        cout << "  SUDOKU ENGINE v11.0 - T0..T6 TUNED     " << endl;
        cout << "  (Copy-On-Branch + Zero-Heap + SIMD)    " << endl;
        cout << "==========================================" << endl;
        
        bool semua_kosong = true;
        for (char c : global_papan) if (c != '0') { semua_kosong = false; break; }
        
        if (semua_kosong) {
            cout << "[Status: Papan KOSONG. Gunakan /load <file>]" << endl;
            cout << "[Session ID: NONE]" << endl;
        } else {
            if (showing_solved) cout << "[Status: Menampilkan PAPAN HASIL SOLVE]" << endl;
            else cout << "[Status: Papan TERMUAT (Original)]" << endl;
            cout << "[Session ID: " << current_session << "]" << endl;
            if (has_solve && !showing_solved)
                cout << "[Status Solve: ✅ SOLVED - gunakan /seesolve]" << endl;
            cetakPapanVisual(global_papan);
        }
        
        cout << "\nPerintah:" << endl;
        cout << "/load <file>  /hitungxy  /calculate  /solve  /count" << endl;
        if (has_solve && !showing_solved && !semua_kosong) cout << "/seesolve" << endl;
        if (showing_solved) cout << "/back" << endl;
        cout << "/quit" << endl;
        cout << "------------------------------------------" << endl;
        cout << "Perintah: ";
        
        string perintah;
        cin >> perintah;
        
        if (perintah == "/quit") { cout << "Selesai." << endl; break; }
        
        if (perintah == "/load") {
            string namaFile; cin >> namaFile;
            ifstream file(namaFile);
            if (!file.is_open()) {
                cout << "Error: File tidak ditemukan.\n";
                cout << "Tekan ENTER..."; cin.ignore(); cin.get(); continue;
            }
            string full_content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
            file.close();
            
            string buffer = "";
            for (char c : full_content) if (c >= '0' && c <= '9' && buffer.length() < 81) buffer += c;
            if (buffer.length() < 81) {
                cout << "Error: Butuh 81 angka.\n";
                cout << "Tekan ENTER..."; cin.ignore(); cin.get(); continue;
            }
            
            unsigned long loaded_session = 0;
            size_t pos = full_content.find("[id_session: ");
            if (pos != string::npos) {
                size_t start = pos + 13, end = full_content.find("]", start);
                if (end != string::npos)
                    try { loaded_session = stoul(full_content.substr(start, end - start)); } catch (...) {}
            }
            
            if (loaded_session > 0) {
                current_session = loaded_session;
                cout << "[OK] Session ID dari file: " << current_session << endl;
            } else {
                current_session = chrono::system_clock::now().time_since_epoch().count() % 1000000;
                ofstream af(namaFile, ios::app);
                af << "\n[id_session: " << current_session << "]\n";
                cout << "[OK] Session ID baru: " << current_session << endl;
            }
            
            string mem_fn = namaFile + ".mem";
            ifstream mf(mem_fn);
            string saved = "";
            if (mf.is_open()) { getline(mf, saved); mf.close(); }
            
            bool modified = false;
            if (saved.empty()) {
                ofstream nf(mem_fn); nf << buffer;
                cout << "[INFO] Memori persisten dibuat.\n";
            } else if (saved != buffer) {
                cout << "\n[PERINGATAN] Papan DIMODIFIKASI!\n";
                ofstream of(mem_fn); of << buffer;
                modified = true;
                remove(("data_koordinat_" + to_string(current_session) + ".txt").c_str());
                remove(("data_solve_" + to_string(current_session) + ".txt").c_str());
                has_solve = false; solved_papan = "";
            } else {
                cout << "[INFO] Papan tidak berubah.\n";
            }
            
            original_papan = buffer;
            global_papan = buffer;
            showing_solved = false;
            
            ifstream ts("data_solve_" + to_string(current_session) + ".txt");
            has_solve = ts.good(); ts.close();
            if (has_solve && !modified) {
                ifstream sf("data_solve_" + to_string(current_session) + ".txt");
                string line;
                while (getline(sf, line))
                    if (line.find("SOLVED_BOARD: ") == 0) { solved_papan = line.substr(14); break; }
                cout << "[INFO] Pernah disolve. Gunakan /seesolve.\n";
            }
            
            vector<string> konflik;
            if (!validasiPetunjukAwal(global_papan, konflik)) {
                cout << "\n[ERROR] Petunjuk TIDAK VALID!\n";
                for (auto& k : konflik) cout << "  - " << k << endl;
                global_papan = original_papan = string(81, '0');
                current_session = 0; has_solve = false; solved_papan = ""; showing_solved = false;
            } else {
                cout << "[OK] 81 karakter valid.\n";
            }
            cout << "Tekan ENTER..."; cin.ignore(); cin.get();
        }
        else if (perintah == "/hitungxy") {
            if (semua_kosong) { cout << "Error: Kosong.\n"; cout << "Tekan ENTER..."; cin.ignore(); cin.get(); continue; }
            int x, y; cout << "X: "; cin >> x; cout << "Y: "; cin >> y;
            cout << "\nPUSAT (" << x << "," << y << ") = [" << bacaKoordinat(global_papan, x, y) << "]\n";
            cout << "Horizontal: [ ";
            for (int xx = -4; xx <= 4; xx++) cout << xx << "x," << y << "y:" << bacaKoordinat(global_papan, xx, y) << " ";
            cout << "]\nVertikal: [ ";
            for (int yy = 4; yy >= -4; yy--) cout << yy << "y," << x << "x:" << bacaKoordinat(global_papan, x, yy) << " ";
            cout << "]\nKandidat: [ ";
            State s; initState(s, global_papan);
            int idx = kartesiusToIndex(x, y);
            if(idx != -1) {
                for (char n = '1'; n <= '9'; n++) {
                    if (s.cand[idx] & (1u << (n - '0'))) cout << n << " ";
                }
            }
            cout << "]\n";
            cout << "Tekan ENTER..."; cin.ignore(); cin.get();
        }
        else if (perintah == "/calculate") {
            if (semua_kosong) { cout << "Error: Kosong.\n"; cout << "Tekan ENTER..."; cin.ignore(); cin.get(); continue; }
            do_calculate();
            cout << "Tekan ENTER..."; cin.ignore(); cin.get();
        }
        else if (perintah == "/solve") {
            if (semua_kosong) { cout << "Error: Kosong.\n"; cout << "Tekan ENTER..."; cin.ignore(); cin.get(); continue; }
            if (showing_solved) { global_papan = original_papan; showing_solved = false; }
            
            string cf = "data_koordinat_" + to_string(current_session) + ".txt";
            ifstream tc(cf);
            if (!tc.good()) {
                cout << "\n[WARNING] File koordinat tidak ada!\n";
                cout << "Auto /calculate... Tekan ENTER."; cin.ignore(); cin.get();
                if (do_calculate()) cout << "\n[OK] Ketik /solve lagi.\n";
                cout << "Tekan ENTER..."; cin.ignore(); cin.get(); continue;
            }
            tc.close();
            
            clearScreen();
            cout << "\n[Solver v11.0 - Tier A/B Engine Optimized...]\n";
            cout << "[INFO] Inisialisasi State POD.\n";
            
            auto t0 = chrono::high_resolution_clock::now();
            // Aktifkan profiling dan paralel di root
            HasilSolver hasil = solveCerdas(global_papan, false, true, true);
            double ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count();
            
            cout << "\n--- Statistik v11.0 ---\n";
            cout << "Waktu         : " << fixed << setprecision(2) << ms << " ms\n";
            cout << "Propagasi     : " << hasil.langkah_propagasi << endl;
            cout << "Eliminasi     : " << hasil.total_elim << " (ALL counted)" << endl;
            cout << "Tebakan       : " << hasil.langkah_tebakan << endl;
            cout << "[deadend: " << hasil.deadend << "] [retry: " << hasil.retry << "]\n";
            
            if (hasil.jumlah_solusi > 0) {
                cout << "\n=== LOG PENGISIAN ===\n";
                for (auto& l : hasil.langkah_pengisian)
                    cout << "[step:" << l.step << " " << l.x << "x," << l.y << "y:" << l.old_val << "->" << l.new_val << "] ";
                cout << "\n\n[TERPECAHKAN]\n";
                cetakPapanVisual(hasil.solusi_pertama);
                
                printTechniqueStats(hasil.teknik_stats);
                printTechniqueTiming(hasil.timing);
                
                solved_papan = hasil.solusi_pertama;
                global_papan = hasil.solusi_pertama;
                has_solve = true; showing_solved = true;
                
                string sf = "data_solve_" + to_string(current_session) + ".txt";
                ofstream f(sf);
                if (f.is_open()) {
                    f << "SESSION_ID: " << current_session << "\n";
                    f << "SOLVE_TIME: " << fixed << setprecision(2) << ms << " ms\n";
                    f << "SOLVED_BOARD: " << hasil.solusi_pertama << "\n\n";
                    f << "=== SOLVED BOARD ===\n+-------+-------+-------+\n";
                    for (int r = 0; r < 9; r++) {
                        f << "| ";
                        for (int c = 0; c < 9; c++) {
                            char v = hasil.solusi_pertama[r*9+c];
                            f << (v == '0' ? ". " : string(1,v) + " ");
                            if ((c+1)%3==0) f << "| ";
                        }
                        f << "\n"; if ((r+1)%3==0) f << "+-------+-------+-------+\n";
                    }
                    f << "\n=== FILL LOG ===\n";
                    for (auto& l : hasil.langkah_pengisian)
                        f << "[step:" << l.step << " " << l.x << "x," << l.y << "y:" << l.old_val << "->" << l.new_val << "]\n";
                    f << "\n=== STATISTICS ===\n";
                    f << "Propagasi: " << hasil.langkah_propagasi << "\n";
                    f << "Eliminasi: " << hasil.total_elim << "\n";
                    f << "Tebakan: " << hasil.langkah_tebakan << "\n";
                    f << "Deadend: " << hasil.deadend << "\nRetry: " << hasil.retry << "\n";
                    f << "Time: " << fixed << setprecision(2) << ms << " ms\n";
                    f << "\n=== TECHNIQUE STATS ===\n";
                    for (int i = 0; i < TECH_COUNT; i++) {
                        auto& v = hasil.teknik_stats[i];
                        if (v.invoked == 0 && v.patterns == 0 && v.removed == 0 && v.filled == 0) continue;
                        f << TECH_NAMES[i] << ":\n";
                        f << "  invoked=" << v.invoked 
                          << " patterns=" << v.patterns
                          << " removed=" << v.removed
                          << " filled=" << v.filled << "\n";
                    }
                    f << "\n=== TECHNIQUE TIMING ===\n";
                    f << fixed << setprecision(3);
                    f << "Naked Single: " << hasil.timing.naked_single_us << " µs | Calls: " << hasil.timing.naked_single_calls << "\n";
                    f << "Hidden Single: " << hasil.timing.hidden_single_us << " µs | Calls: " << hasil.timing.hidden_single_calls << "\n";
                    f << "Naked Subsets: " << hasil.timing.naked_subsets_us << " µs | Calls: " << hasil.timing.naked_subsets_calls << "\n";
                    f << "Locked Candidates: " << hasil.timing.locked_candidates_us << " µs | Calls: " << hasil.timing.locked_candidates_calls << "\n";
                    f << "X-Wing: " << hasil.timing.xwing_us << " µs | Calls: " << hasil.timing.xwing_calls << "\n";
                    double total = hasil.timing.naked_single_us + hasil.timing.hidden_single_us + 
                                   hasil.timing.naked_subsets_us + hasil.timing.locked_candidates_us + 
                                   hasil.timing.xwing_us;
                    f << "Total Technique Time: " << total << " µs (" << (total / 1000.0) << " ms)\n";
                    cout << "\n[OK] Disimpan ke " << sf << endl;
                }
            } else {
                cout << "\n[GAGAL]\n";
            }
            cout << "Tekan ENTER..."; cin.ignore(); cin.get();
        }
        else if (perintah == "/count") {
            if (semua_kosong) { cout << "Error.\n"; cout << "Tekan ENTER..."; cin.ignore(); cin.get(); continue; }
            if (showing_solved) { global_papan = original_papan; showing_solved = false; }
            clearScreen();
            cout << "\n[Validator v11.0 - Tier A Strict (Sequential) ...]\n";
            
            auto t0 = chrono::high_resolution_clock::now();
            // hitung_semua = true, use_profiling = true, use_parallel = false (Tier A Strict)
            HasilSolver hasil = solveCerdas(global_papan, true, true, false);
            double ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count();
            
            cout << "\n--- Validasi v11.0 ---\n";
            cout << "Solusi    : " << hasil.jumlah_solusi << endl;
            cout << "Propagasi : " << hasil.langkah_propagasi << endl;
            cout << "Eliminasi : " << hasil.total_elim << " (ALL counted)" << endl;
            cout << "Tebakan   : " << hasil.langkah_tebakan << endl;
            cout << "[deadend: " << hasil.deadend << "] [retry: " << hasil.retry << "]\n";
            
            printTechniqueStats(hasil.teknik_stats);
            printTechniqueTiming(hasil.timing);
            
            if (hasil.jumlah_solusi == 0) cout << "\n❌ Tidak ada solusi.\n";
            else if (hasil.jumlah_solusi == 1) cout << "\n✅ VALID (unik).\n";
            else cout << "\n⚠️ INVALID (multiple).\n";
            cout << "Tekan ENTER..."; cin.ignore(); cin.get();
        }
        else if (perintah == "/seesolve") {
            if (!has_solve || solved_papan.length() != 81) cout << "Error: Tidak ada.\n";
            else { global_papan = solved_papan; showing_solved = true; cout << "[INFO] Solved board.\n"; }
            cout << "Tekan ENTER..."; cin.ignore(); cin.get();
        }
        else if (perintah == "/back") {
            if (showing_solved) { global_papan = original_papan; showing_solved = false; cout << "[INFO] Original.\n"; }
            cout << "Tekan ENTER..."; cin.ignore(); cin.get();
        }
        else { cout << "Unknown.\n"; cout << "Tekan ENTER..."; cin.ignore(); cin.get(); }
    }
    return 0;
}
