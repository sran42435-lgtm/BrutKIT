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

using namespace std;

// ==========================================
// STRUKTUR DATA: Kandidat berbasis Bitmask
// ==========================================
struct Kandidat {
    uint16_t mask;
    Kandidat() : mask(0) {}
    Kandidat(uint16_t m) : mask(m) {}
    bool ada(int num) const { return (mask & (1 << num)) != 0; }
    void tambah(int num) { mask |= (1 << num); }
    void hapus(int num) { mask &= ~(1 << num); }
    int jumlah() const { return __builtin_popcount(mask); }
    bool kosong() const { return mask == 0; }
    int pertama() const {
        for (int i = 1; i <= 9; i++) if (ada(i)) return i;
        return 0;
    }
};

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
    TECH_CONTRADICTION_PROBE,
    TECH_COUNT
};

constexpr const char* TECH_NAMES[] = {
    "Naked Single", "Hidden Single (Baris)", "Hidden Single (Kolom)", "Hidden Single (Box)",
    "Locked Claiming", "Locked Pointing",
    "Naked Pair", "Naked Triple", "Naked Quad",
    "X-Wing", "Backtracking (LCV)",
    "Contradiction Probe"
};

struct TechniqueStats {
    uint32_t invoked = 0;
    uint32_t patterns = 0;
    uint32_t removed = 0;
    uint32_t filled = 0;
};

// ==========================================
// STATIC TABLES
// ==========================================
array<array<int, 20>, 81> PEERS;
array<array<int, 9>, 27> UNITS;
array<array<int, 3>, 81> CELL_UNITS;
array<array<int, 81>, 27> LOCAL_POS;

// ==========================================
// DirtyXWing — event-driven X-Wing queue
// ==========================================
struct DirtyXWing {
    queue<pair<int,int>> rows;
    queue<pair<int,int>> cols;
    bool dirtyRows[9][10]{};
    bool dirtyCols[9][10]{};
    
    void markRow(int row, int num) {
        if (!dirtyRows[row][num]) {
            rows.push({row, num});
            dirtyRows[row][num] = true;
        }
    }
    void markCol(int col, int num) {
        if (!dirtyCols[col][num]) {
            cols.push({col, num});
            dirtyCols[col][num] = true;
        }
    }
    bool empty() const { return rows.empty() && cols.empty(); }
};

// ==========================================
// TechniqueTiming — profiling data
// ==========================================
struct TechniqueTiming {
    double naked_single_us = 0;
    double hidden_single_us = 0;
    double naked_subsets_us = 0;
    double locked_candidates_us = 0;
    double xwing_us = 0;
    double probe_us = 0;

    long long naked_single_calls = 0;
    long long hidden_single_calls = 0;
    long long naked_subsets_calls = 0;
    long long locked_candidates_calls = 0;
    long long xwing_calls = 0;
    long long probe_calls = 0;
};

template <typename Func>
double ukurWaktu(Func&& fungsi) {
    auto start = chrono::steady_clock::now();
    fungsi();
    auto finish = chrono::steady_clock::now();
    return chrono::duration<double, micro>(finish - start).count();
}

// ==========================================
// Trail for Backtracking
// ==========================================
struct TrailEntry {
    int idx;
    char old_papan;
    uint16_t old_mask;
};
vector<TrailEntry> g_trail;

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
// FUNGSI JEMBATAN KOORDINAT
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
void tulisKoordinat(string& papan, int x, int y, char val) {
    int idx = kartesiusToIndex(x, y);
    if (idx != -1) papan[idx] = val;
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

vector<Kandidat> inisialisasiKandidat(const string& papan) {
    vector<Kandidat> kand(81);
    for (int i = 0; i < 81; i++) if (papan[i] == '0') kand[i].mask = 0x3FE;
    for (int i = 0; i < 81; i++) {
        if (papan[i] != '0') {
            int num = papan[i] - '0';
            for (int p = 0; p < 20; p++) kand[PEERS[i][p]].hapus(num);
        }
    }
    return kand;
}

struct LangkahLog {
    int step;
    int x, y;
    char old_val;
    char new_val;
    string teknik;
};

void trailUndo(size_t mark, string& papan, vector<Kandidat>& kand) {
    while (g_trail.size() > mark) {
        auto& e = g_trail.back();
        papan[e.idx] = e.old_papan;
        kand[e.idx].mask = e.old_mask;
        g_trail.pop_back();
    }
}

void rebuildUDP(const string& papan, const vector<Kandidat>& kand, array<array<uint16_t, 10>, 27>& udp) {
    for (int u = 0; u < 27; u++) {
        for (int d = 1; d <= 9; d++) udp[u][d] = 0;
        for (int i = 0; i < 9; i++) {
            int idx = UNITS[u][i];
            if (papan[idx] == '0') {
                uint16_t m = kand[idx].mask;
                while (m) {
                    int d = __builtin_ctz(m);
                    udp[u][d] |= (1u << i);
                    m &= m - 1;
                }
            }
        }
    }
}

inline void markDirty(uint32_t& mask, int u) { mask |= (1u << u); }
inline int popDirty(uint32_t& mask) {
    int u = __builtin_ctz(mask);
    mask &= (mask - 1);
    return u;
}

bool hapusKandidatInc(int idx, int num, string& papan, vector<Kandidat>& kand,
                      uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, array<uint16_t, 27>& changedDigits,
                      int& total_elim, array<TechniqueStats, TECH_COUNT>& stats, TechniqueId tech,
                      DirtyXWing& xwing, array<array<uint16_t, 10>, 27>& udp) {
    if (!kand[idx].ada(num)) return false;
    g_trail.push_back({idx, papan[idx], kand[idx].mask});
    kand[idx].hapus(num);
    total_elim++;
    if (tech != TECH_COUNT) stats[tech].removed++;
    
    for (int i = 0; i < 3; i++) {
        int u = CELL_UNITS[idx][i];
        markDirty(dirtyUnits, u);
        markDirty(lockedDirtyUnits, u);
        changedDigits[u] |= (1 << num);
        udp[u][num] &= ~(1u << LOCAL_POS[u][idx]);
    }
    xwing.markRow(idx / 9, num);
    xwing.markCol(idx % 9, num);
    return true;
}

void isiSelInc(int idx, int num, string& papan, vector<Kandidat>& kand,
               uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, array<uint16_t, 27>& changedDigits,
               int& step_counter, vector<LangkahLog>& log,
               array<TechniqueStats, TECH_COUNT>& stats, TechniqueId tech, 
               int& total_elim, DirtyXWing& xwing, array<array<uint16_t, 10>, 27>& udp) {
    uint16_t oldMask = kand[idx].mask;
    g_trail.push_back({idx, papan[idx], oldMask});
    papan[idx] = '0' + num;
    kand[idx].mask = 0;
    auto [x, y] = indexToKartesius(idx);
    log.push_back({++step_counter, x, y, '0', papan[idx], TECH_NAMES[tech]});
    stats[tech].filled++;
    
    for (int i = 0; i < 3; i++) {
        int u = CELL_UNITS[idx][i];
        markDirty(dirtyUnits, u);
        markDirty(lockedDirtyUnits, u);
        uint16_t m = oldMask;
        while (m) {
            int d = __builtin_ctz(m);
            udp[u][d] &= ~(1u << LOCAL_POS[u][idx]);
            m &= m - 1;
        }
    }
    
    for (int d = 1; d <= 9; d++) {
        if (oldMask & (1 << d)) {
            xwing.markRow(idx / 9, d);
            xwing.markCol(idx % 9, d);
        }
    }
    
    for (int p = 0; p < 20; p++)
        hapusKandidatInc(PEERS[idx][p], num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_COUNT, xwing, udp);
}

void isiSelTrail(int idx, int num, string& papan, vector<Kandidat>& kand, int& total_elim, 
                 DirtyXWing& xwing, array<uint16_t, 27>& changedDigits) {
    uint16_t oldMask = kand[idx].mask;
    g_trail.push_back({idx, papan[idx], oldMask});
    papan[idx] = '0' + num;
    kand[idx].mask = 0;
    
    for (int d = 1; d <= 9; d++) {
        if (oldMask & (1 << d)) {
            xwing.markRow(idx / 9, d);
            xwing.markCol(idx % 9, d);
        }
    }
    for (int i = 0; i < 3; i++) {
        changedDigits[CELL_UNITS[idx][i]] |= oldMask;
    }
    
    for (int p = 0; p < 20; p++) {
        int peer = PEERS[idx][p];
        if (kand[peer].ada(num)) {
            g_trail.push_back({peer, papan[peer], kand[peer].mask});
            kand[peer].hapus(num);
            total_elim++;
            xwing.markRow(peer / 9, num);
            xwing.markCol(peer % 9, num);
            for (int i = 0; i < 3; i++) {
                changedDigits[CELL_UNITS[peer][i]] |= (1 << num);
            }
        }
    }
}

void prosesNakedSubsetsUnit(int u, string& papan, vector<Kandidat>& kand, 
                            uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, array<uint16_t, 27>& changedDigits,
                            array<TechniqueStats, TECH_COUNT>& stats, int& total_elim, DirtyXWing& xwing, array<array<uint16_t, 10>, 27>& udp) {
    int pairCells[9], tripleCells[9], quadCells[9];
    int c2 = 0, c3 = 0, c4 = 0;
    for (int i = 0; i < 9; i++) {
        int idx = UNITS[u][i];
        if (papan[idx] != '0') continue;
        int cnt = __builtin_popcount(kand[idx].mask);
        if (cnt == 2) { pairCells[c2++] = idx; tripleCells[c3++] = idx; quadCells[c4++] = idx; }
        else if (cnt == 3) { tripleCells[c3++] = idx; quadCells[c4++] = idx; }
        else if (cnt == 4) { quadCells[c4++] = idx; }
    }
    
    if (c2 >= 2) {
        stats[TECH_NAKED_PAIR].invoked++;
        for (int i = 0; i < c2; i++) {
            for (int j = i + 1; j < c2; j++) {
                uint16_t um = kand[pairCells[i]].mask | kand[pairCells[j]].mask;
                if (__builtin_popcount(um) == 2) {
                    stats[TECH_NAKED_PAIR].patterns++;
                    for (int k = 0; k < 9; k++) {
                        int idx = UNITS[u][k];
                        if (idx != pairCells[i] && idx != pairCells[j]) {
                            uint16_t common = kand[idx].mask & um;
                            while (common) {
                                int num = __builtin_ctz(common);
                                hapusKandidatInc(idx, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_NAKED_PAIR, xwing, udp);
                                common &= common - 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (c3 >= 3) {
        stats[TECH_NAKED_TRIPLE].invoked++;
        for (int i = 0; i < c3; i++) {
            for (int j = i + 1; j < c3; j++) {
                for (int k = j + 1; k < c3; k++) {
                    uint16_t um = kand[tripleCells[i]].mask | kand[tripleCells[j]].mask | kand[tripleCells[k]].mask;
                    if (__builtin_popcount(um) == 3) {
                        stats[TECH_NAKED_TRIPLE].patterns++;
                        for (int m = 0; m < 9; m++) {
                            int idx = UNITS[u][m];
                            if (idx != tripleCells[i] && idx != tripleCells[j] && idx != tripleCells[k]) {
                                uint16_t common = kand[idx].mask & um;
                                while (common) {
                                    int num = __builtin_ctz(common);
                                    hapusKandidatInc(idx, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_NAKED_TRIPLE, xwing, udp);
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
        stats[TECH_NAKED_QUAD].invoked++;
        for (int i = 0; i < c4; i++) {
            for (int j = i + 1; j < c4; j++) {
                for (int k = j + 1; k < c4; k++) {
                    for (int l = k + 1; l < c4; l++) {
                        uint16_t um = kand[quadCells[i]].mask | kand[quadCells[j]].mask | 
                                      kand[quadCells[k]].mask | kand[quadCells[l]].mask;
                        if (__builtin_popcount(um) == 4) {
                            stats[TECH_NAKED_QUAD].patterns++;
                            for (int m = 0; m < 9; m++) {
                                int idx = UNITS[u][m];
                                if (idx != quadCells[i] && idx != quadCells[j] && 
                                    idx != quadCells[k] && idx != quadCells[l]) {
                                    uint16_t common = kand[idx].mask & um;
                                    while (common) {
                                        int num = __builtin_ctz(common);
                                        hapusKandidatInc(idx, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_NAKED_QUAD, xwing, udp);
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

bool prosesLockedPadaUnit(int u, string& papan, vector<Kandidat>& kand, 
                          uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, array<uint16_t, 27>& changedDigits,
                          array<TechniqueStats, TECH_COUNT>& stats, int& total_elim, DirtyXWing& xwing, array<array<uint16_t, 10>, 27>& udp) {
    bool changed = false;
    
    if (u >= 18) {
        int box = u - 18;
        int br = box / 3;
        int bc = box % 3;
        
        for (int d = 1; d <= 9; d++) {
            stats[TECH_LOCKED_POINTING].invoked++;
            uint16_t pos = udp[u][d];
            if (__builtin_popcount(pos) < 2) continue;
            
            if ((pos & 0x007) == pos) {
                stats[TECH_LOCKED_POINTING].patterns++;
                int r = br * 3 + 0;
                for (int c = 0; c < 9; c++) {
                    if (c / 3 == bc) continue;
                    if (hapusKandidatInc(r * 9 + c, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_POINTING, xwing, udp)) changed = true;
                }
            } else if ((pos & 0x038) == pos) {
                stats[TECH_LOCKED_POINTING].patterns++;
                int r = br * 3 + 1;
                for (int c = 0; c < 9; c++) {
                    if (c / 3 == bc) continue;
                    if (hapusKandidatInc(r * 9 + c, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_POINTING, xwing, udp)) changed = true;
                }
            } else if ((pos & 0x1C0) == pos) {
                stats[TECH_LOCKED_POINTING].patterns++;
                int r = br * 3 + 2;
                for (int c = 0; c < 9; c++) {
                    if (c / 3 == bc) continue;
                    if (hapusKandidatInc(r * 9 + c, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_POINTING, xwing, udp)) changed = true;
                }
            }
            
            if ((pos & 0x049) == pos) {
                stats[TECH_LOCKED_POINTING].patterns++;
                int c = bc * 3 + 0;
                for (int r = 0; r < 9; r++) {
                    if (r / 3 == br) continue;
                    if (hapusKandidatInc(r * 9 + c, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_POINTING, xwing, udp)) changed = true;
                }
            } else if ((pos & 0x092) == pos) {
                stats[TECH_LOCKED_POINTING].patterns++;
                int c = bc * 3 + 1;
                for (int r = 0; r < 9; r++) {
                    if (r / 3 == br) continue;
                    if (hapusKandidatInc(r * 9 + c, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_POINTING, xwing, udp)) changed = true;
                }
            } else if ((pos & 0x124) == pos) {
                stats[TECH_LOCKED_POINTING].patterns++;
                int c = bc * 3 + 2;
                for (int r = 0; r < 9; r++) {
                    if (r / 3 == br) continue;
                    if (hapusKandidatInc(r * 9 + c, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_POINTING, xwing, udp)) changed = true;
                }
            }
        }
    } else {
        bool isRow = u < 9;
        int unitIndex = isRow ? u : u - 9;
        
        for (int d = 1; d <= 9; d++) {
            stats[TECH_LOCKED_CLAIMING].invoked++;
            uint16_t pos = udp[u][d];
            if (__builtin_popcount(pos) < 2) continue;
            
            if ((pos & 0x007) == pos || (pos & 0x038) == pos || (pos & 0x1C0) == pos) {
                stats[TECH_LOCKED_CLAIMING].patterns++;
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
                            if (hapusKandidatInc(rr * 9 + cc, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_CLAIMING, xwing, udp)) changed = true;
                        }
                    }
                } else {
                    int bc = (unitIndex / 3) * 3;
                    int box_row = box_col * 3;
                    for (int cc = bc; cc < bc + 3; cc++) {
                        if (cc == unitIndex) continue;
                        for (int rr = box_row; rr < box_row + 3; rr++) {
                            if (hapusKandidatInc(rr * 9 + cc, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_LOCKED_CLAIMING, xwing, udp)) changed = true;
                        }
                    }
                }
            }
        }
    }
    return changed;
}

bool prosesXWingBaris(int row, int num, string& papan, vector<Kandidat>& kand,
                      uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, array<uint16_t, 27>& changedDigits,
                      int& total_elim, array<TechniqueStats, TECH_COUNT>& stats, DirtyXWing& xwing, array<array<uint16_t, 10>, 27>& udp) {
    stats[TECH_X_WING].invoked++;
    uint16_t pos = udp[row][num];
    if (__builtin_popcount(pos) != 2) return false;
    
    bool changed = false;
    for (int other = 0; other < 9; other++) {
        if (other == row) continue;
        if (udp[other][num] == pos) {
            stats[TECH_X_WING].patterns++;
            int c1 = __builtin_ctz(pos);
            int c2 = __builtin_ctz(pos & (pos - 1));
            for (int r = 0; r < 9; r++) {
                if (r == row || r == other) continue;
                if (hapusKandidatInc(r * 9 + c1, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_X_WING, xwing, udp)) changed = true;
                if (hapusKandidatInc(r * 9 + c2, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_X_WING, xwing, udp)) changed = true;
            }
        }
    }
    return changed;
}

bool prosesXWingKolom(int col, int num, string& papan, vector<Kandidat>& kand,
                      uint32_t& dirtyUnits, uint32_t& lockedDirtyUnits, array<uint16_t, 27>& changedDigits,
                      int& total_elim, array<TechniqueStats, TECH_COUNT>& stats, DirtyXWing& xwing, array<array<uint16_t, 10>, 27>& udp) {
    stats[TECH_X_WING].invoked++;
    uint16_t pos = udp[col + 9][num];
    if (__builtin_popcount(pos) != 2) return false;
    
    bool changed = false;
    for (int other = 0; other < 9; other++) {
        if (other == col) continue;
        if (udp[other + 9][num] == pos) {
            stats[TECH_X_WING].patterns++;
            int r1 = __builtin_ctz(pos);
            int r2 = __builtin_ctz(pos & (pos - 1));
            for (int c = 0; c < 9; c++) {
                if (c == col || c == other) continue;
                if (hapusKandidatInc(r1 * 9 + c, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_X_WING, xwing, udp)) changed = true;
                if (hapusKandidatInc(r2 * 9 + c, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, TECH_X_WING, xwing, udp)) changed = true;
            }
        }
    }
    return changed;
}

bool propagasiIncremental(string& papan, vector<Kandidat>& kand, int& step_counter,
                          vector<LangkahLog>& log, array<TechniqueStats, TECH_COUNT>& stats, 
                          int& total_elim, TechniqueTiming& timing, 
                          DirtyXWing& xwing, array<uint16_t, 27>& changedDigits, 
                          array<array<uint16_t, 10>, 27>& udp, int changedCell = -1) {
    uint32_t dirtyUnits = 0;
    uint32_t lockedDirtyUnits = 0;
    
    rebuildUDP(papan, kand, udp);
    
    if (changedCell == -1) {
        dirtyUnits = 0x07FFFFFF;
        lockedDirtyUnits = 0x07FFFFFF;
        for (int u = 0; u < 27; u++) changedDigits[u] = 0x3FE;
        
        for (int i = 0; i < 81; i++) {
            if (papan[i] != '0') {
                int num = papan[i] - '0';
                xwing.markRow(i / 9, num);
                xwing.markCol(i % 9, num);
            } else {
                uint16_t m = kand[i].mask;
                while (m) {
                    int num = __builtin_ctz(m);
                    xwing.markRow(i / 9, num);
                    xwing.markCol(i % 9, num);
                    m &= m - 1;
                }
            }
        }
    } else {
        for (int i = 0; i < 3; i++) {
            int u = CELL_UNITS[changedCell][i];
            markDirty(dirtyUnits, u);
            markDirty(lockedDirtyUnits, u);
        }
    }
    
    while (true) {
        while (dirtyUnits != 0) {
            int u = popDirty(dirtyUnits);
            
            timing.naked_single_us += ukurWaktu([&]() {
                for (int i = 0; i < 9; i++) {
                    int idx = UNITS[u][i];
                    if (papan[idx] == '0' && __builtin_popcount(kand[idx].mask) == 1) {
                        int num = kand[idx].pertama();
                        stats[TECH_NAKED_SINGLE].invoked++;
                        isiSelInc(idx, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, step_counter, log, stats, TECH_NAKED_SINGLE, total_elim, xwing, udp);
                    }
                }
            });
            timing.naked_single_calls++;
            
            timing.hidden_single_us += ukurWaktu([&]() {
                uint16_t digits = changedDigits[u];
                changedDigits[u] = 0;
                while (digits) {
                    int d = __builtin_ctz(digits);
                    digits &= digits - 1;
                    
                    uint16_t pos = udp[u][d];
                    if (pos != 0 && (pos & (pos - 1)) == 0) {
                        bool placed = false;
                        for (int i = 0; i < 9; i++) if (papan[UNITS[u][i]] == '0' + d) { placed = true; break; }
                        if (!placed) {
                            int localPos = __builtin_ctz(pos);
                            int target = UNITS[u][localPos];
                            TechniqueId tech = (u < 9) ? TECH_HIDDEN_SINGLE_ROW : (u < 18) ? TECH_HIDDEN_SINGLE_COL : TECH_HIDDEN_SINGLE_BOX;
                            stats[tech].invoked++;
                            isiSelInc(target, d, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, step_counter, log, stats, tech, total_elim, xwing, udp);
                        }
                    }
                }
            });
            timing.hidden_single_calls++;
            
            timing.naked_subsets_us += ukurWaktu([&]() {
                prosesNakedSubsetsUnit(u, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, stats, total_elim, xwing, udp);
            });
            timing.naked_subsets_calls++;
            
            for (int d = 1; d <= 9; d++) {
                if (udp[u][d] == 0) {
                    bool placed = false;
                    for (int i = 0; i < 9; i++) if (papan[UNITS[u][i]] == '0' + d) { placed = true; break; }
                    if (!placed) return false;
                }
            }
            
            for (int i = 0; i < 9; i++) {
                int idx = UNITS[u][i];
                if (papan[idx] == '0' && kand[idx].kosong()) return false;
            }
        }
        
        bool globalChanged = false;
        
        timing.locked_candidates_us += ukurWaktu([&]() {
            while (lockedDirtyUnits != 0) {
                int u = popDirty(lockedDirtyUnits);
                int emptyCount = 0;
                for (int i = 0; i < 9; i++) if (papan[UNITS[u][i]] == '0') emptyCount++;
                if (emptyCount < 2) continue;
                
                if (prosesLockedPadaUnit(u, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, stats, total_elim, xwing, udp))
                    globalChanged = true;
            }
        });
        timing.locked_candidates_calls++;
        
        timing.xwing_us += ukurWaktu([&]() {
            while (!xwing.empty()) {
                if (!xwing.rows.empty()) {
                    auto [row, num] = xwing.rows.front();
                    xwing.rows.pop();
                    xwing.dirtyRows[row][num] = false;
                    if (prosesXWingBaris(row, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, xwing, udp))
                        globalChanged = true;
                }
                if (!xwing.cols.empty()) {
                    auto [col, num] = xwing.cols.front();
                    xwing.cols.pop();
                    xwing.dirtyCols[col][num] = false;
                    if (prosesXWingKolom(col, num, papan, kand, dirtyUnits, lockedDirtyUnits, changedDigits, total_elim, stats, xwing, udp))
                        globalChanged = true;
                }
            }
        });
        timing.xwing_calls++;
        
        if (!globalChanged) break;
    }
    return true;
}

// ==========================================
// [v10.6] MRVResult & Enhanced MRV
// ==========================================
struct MRVResult {
    int idx = -1;
    int candidate_count = 0;
    int degree = 0;
};

MRVResult cariSelMRV(const string& papan, const vector<Kandidat>& kand) {
    MRVResult result;
    int min_k = 10;
    int max_deg = -1;

    for (int i = 0; i < 81; i++) {
        if (papan[i] != '0') continue;
        int jml = __builtin_popcount(kand[i].mask);
        if (jml == 0) {
            result.idx = -2;
            return result;
        }
        if (jml < min_k) {
            min_k = jml;
            max_deg = -1;
            result.idx = i;
            result.candidate_count = jml;
        }
        if (jml == min_k) {
            int deg = 0;
            for (int p = 0; p < 20; p++) {
                if (papan[PEERS[i][p]] == '0') deg++;
            }
            if (deg > max_deg) {
                max_deg = deg;
                result.idx = i;
                result.candidate_count = jml;
                result.degree = deg;
            }
        }
    }
    return result;
}

// ==========================================
// [v10.6] Probe Eligibility Gate
// ==========================================
inline bool layakContradictionProbe(const MRVResult& mrv) {
    if (mrv.candidate_count > 3) return false;
    if (mrv.candidate_count < 2) return false;
    if (mrv.candidate_count == 2) return mrv.degree >= 8;
    if (mrv.candidate_count == 3) return mrv.degree >= 12;
    return false;
}

int hitungEliminasiLCV(const vector<Kandidat>& kand, int idx, int num) {
    int eliminasi = 0;
    for (int p = 0; p < 20; p++)
        if (kand[PEERS[idx][p]].ada(num)) eliminasi++;
    return eliminasi;
}

// ==========================================
// [v10.6] Adaptive MRV Contradiction Probe with Queue Deduplication
// ==========================================
bool probeContradiction(int idx, int num, string& papan, vector<Kandidat>& kand) {
    size_t probeMark = g_trail.size();
    bool contradiction = false;

    g_trail.push_back({idx, papan[idx], kand[idx].mask});
    papan[idx] = '0' + num;
    kand[idx].mask = 0;

    int q[81];
    int q_head = 0, q_tail = 0;
    uint8_t queued[81] = {}; // Deduplication array

    // Level 1: Peer elimination & immediate empty mask check
    for (int p = 0; p < 20; p++) {
        int peer = PEERS[idx][p];
        if (kand[peer].ada(num)) {
            g_trail.push_back({peer, papan[peer], kand[peer].mask});
            kand[peer].hapus(num);
            if (kand[peer].kosong() && papan[peer] == '0') {
                contradiction = true;
                break;
            }
            if (__builtin_popcount(kand[peer].mask) == 1 && papan[peer] == '0' && !queued[peer]) {
                queued[peer] = 1;
                q[q_tail++] = peer;
            }
        }
    }

    // Level 2: Naked Single cascade (bounded local propagation)
    while (!contradiction && q_head < q_tail) {
        int curr = q[q_head++];
        queued[curr] = 0;
        if (papan[curr] != '0') continue;
        if (kand[curr].kosong()) {
            contradiction = true;
            break;
        }
        int val = kand[curr].pertama();
        
        g_trail.push_back({curr, papan[curr], kand[curr].mask});
        papan[curr] = '0' + val;
        kand[curr].mask = 0;
        
        for (int p = 0; p < 20; p++) {
            int peer = PEERS[curr][p];
            if (kand[peer].ada(val)) {
                g_trail.push_back({peer, papan[peer], kand[peer].mask});
                kand[peer].hapus(val);
                if (kand[peer].kosong() && papan[peer] == '0') {
                    contradiction = true;
                    break;
                }
                if (__builtin_popcount(kand[peer].mask) == 1 && papan[peer] == '0' && !queued[peer]) {
                    queued[peer] = 1;
                    q[q_tail++] = peer;
                }
            }
        }
        if (contradiction) break;
    }

    // Level 3: Unit feasibility check (only 3 units of the target cell)
    if (!contradiction) {
        for (int i = 0; i < 3; i++) {
            int u = CELL_UNITS[idx][i];
            for (int d = 1; d <= 9; d++) {
                bool found = false;
                for (int j = 0; j < 9; j++) {
                    int c_idx = UNITS[u][j];
                    if (papan[c_idx] == '0' + d || (papan[c_idx] == '0' && kand[c_idx].ada(d))) {
                        found = true; break;
                    }
                }
                if (!found) { contradiction = true; break; }
            }
            if (contradiction) break;
        }
    }

    // Undo all probe changes perfectly using Trail
    while (g_trail.size() > probeMark) {
        auto& e = g_trail.back();
        papan[e.idx] = e.old_papan;
        kand[e.idx].mask = e.old_mask;
        g_trail.pop_back();
    }

    return contradiction;
}

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

HasilSolver solveCerdas(string& papan, vector<Kandidat>& kand,
                         bool hitung_semua, int batas = 1,
                         int lvl = 0, int step_counter = 0, int changedCell = -1,
                         DirtyXWing* ext_xwing = nullptr, array<uint16_t, 27>* ext_changedDigits = nullptr,
                         array<array<uint16_t, 10>, 27>* ext_udp = nullptr) {
    HasilSolver hasil = {0, "", {}, 0, 0, 0, 0, 0, {}, {}, {}};
    
    size_t nodeMark = g_trail.size();
    int savedStep = step_counter;
    
    DirtyXWing local_xwing;
    array<uint16_t, 27> local_changedDigits{};
    array<array<uint16_t, 10>, 27> local_udp{};
    
    DirtyXWing& xwing = ext_xwing ? *ext_xwing : local_xwing;
    array<uint16_t, 27>& changedDigits = ext_changedDigits ? *ext_changedDigits : local_changedDigits;
    array<array<uint16_t, 10>, 27>& udp = ext_udp ? *ext_udp : local_udp;
    
    vector<LangkahLog> localLog;
    int localElim = 0;
    
    bool ok = propagasiIncremental(papan, kand, step_counter, localLog, hasil.teknik_stats, 
                                    localElim, hasil.timing, xwing, changedDigits, udp, changedCell);
    
    if (!ok) {
        trailUndo(nodeMark, papan, kand);
        hasil.deadend = 1;
        hasil.langkah_propagasi = step_counter - savedStep;
        hasil.total_elim = localElim;
        return hasil;
    }
    
    // ==========================================
    // [v10.6] DECISION CYCLE LOOP
    // ==========================================
    int idx = -1;
    while (true) {
        bool selesai = true;
        for (int i = 0; i < 81; i++) if (papan[i] == '0') { selesai = false; break; }
        if (selesai) {
            hasil.jumlah_solusi = 1;
            hasil.solusi_pertama = papan;
            hasil.semua_solusi.push_back(papan);
            hasil.langkah_pengisian = localLog;
            trailUndo(nodeMark, papan, kand);
            hasil.langkah_propagasi = step_counter - savedStep;
            hasil.total_elim = localElim;
            return hasil;
        }
        
        MRVResult mrv = cariSelMRV(papan, kand);
        if (mrv.idx == -2) {
            trailUndo(nodeMark, papan, kand);
            hasil.deadend = 1;
            hasil.langkah_propagasi = step_counter - savedStep;
            hasil.total_elim = localElim;
            return hasil;
        }
        
        idx = mrv.idx;
        bool probe_changed = false;
        
        if (layakContradictionProbe(mrv)) {
            uint16_t mask_copy = kand[idx].mask;
            hasil.timing.probe_us += ukurWaktu([&]() {
                hasil.teknik_stats[TECH_CONTRADICTION_PROBE].invoked++;
                while (mask_copy) {
                    int num = __builtin_ctz(mask_copy);
                    mask_copy &= mask_copy - 1;
                    
                    if (probeContradiction(idx, num, papan, kand)) {
                        g_trail.push_back({idx, papan[idx], kand[idx].mask});
                        kand[idx].hapus(num);
                        probe_changed = true;
                        hasil.teknik_stats[TECH_CONTRADICTION_PROBE].removed++;
                    }
                }
            });
            hasil.timing.probe_calls++;
            
            if (probe_changed) {
                if (kand[idx].kosong()) {
                    trailUndo(nodeMark, papan, kand);
                    hasil.deadend = 1;
                    hasil.langkah_propagasi = step_counter - savedStep;
                    hasil.total_elim = localElim;
                    return hasil;
                }
                
                bool ok_after_probe = propagasiIncremental(papan, kand, step_counter, localLog, 
                                                           hasil.teknik_stats, localElim, hasil.timing, 
                                                           xwing, changedDigits, udp, idx);
                
                if (!ok_after_probe) {
                    trailUndo(nodeMark, papan, kand);
                    hasil.deadend = 1;
                    hasil.langkah_propagasi = step_counter - savedStep;
                    hasil.total_elim = localElim;
                    return hasil;
                }
                continue; // Re-evaluate selesai and MRV with the new state
            }
        }
        break; // Proceed to LCV
    }
    
    hasil.langkah_pengisian = localLog;
    
    vector<pair<int, int>> kandidat_lcv;
    for (int num = 1; num <= 9; num++)
        if (kand[idx].ada(num))
            kandidat_lcv.push_back({num, hitungEliminasiLCV(kand, idx, num)});
    sort(kandidat_lcv.begin(), kandidat_lcv.end(),
         [](const pair<int,int>& a, const pair<int,int>& b) { return a.second < b.second; });
    
    int attempts = 0;
    for (auto& [num, score] : kandidat_lcv) {
        attempts++;
        if (attempts > 1) hasil.retry++;
        hasil.langkah_tebakan++;
        
        size_t branchMark = g_trail.size();
        int branchStep = step_counter;
        int branchElim = 0;
        
        isiSelTrail(idx, num, papan, kand, branchElim, xwing, changedDigits);
        
        auto [x, y] = indexToKartesius(idx);
        LangkahLog guess_log = {step_counter + attempts, x, y, '0', (char)('0' + num), "Backtracking (LCV)"};
        
        HasilSolver sub = solveCerdas(papan, kand, hitung_semua, batas, lvl + 1, step_counter + attempts, idx, &xwing, &changedDigits, &udp);
        
        hasil.jumlah_solusi += sub.jumlah_solusi;
        hasil.langkah_propagasi += sub.langkah_propagasi;
        hasil.langkah_tebakan += sub.langkah_tebakan;
        hasil.deadend += sub.deadend;
        hasil.retry += sub.retry;
        hasil.total_elim += sub.total_elim + branchElim;
        
        for (int i = 0; i < TECH_COUNT; i++) {
            hasil.teknik_stats[i].invoked += sub.teknik_stats[i].invoked;
            hasil.teknik_stats[i].patterns += sub.teknik_stats[i].patterns;
            hasil.teknik_stats[i].removed += sub.teknik_stats[i].removed;
            hasil.teknik_stats[i].filled += sub.teknik_stats[i].filled;
        }
        
        hasil.timing.naked_single_us += sub.timing.naked_single_us;
        hasil.timing.hidden_single_us += sub.timing.hidden_single_us;
        hasil.timing.naked_subsets_us += sub.timing.naked_subsets_us;
        hasil.timing.locked_candidates_us += sub.timing.locked_candidates_us;
        hasil.timing.xwing_us += sub.timing.xwing_us;
        hasil.timing.probe_us += sub.timing.probe_us;
        
        hasil.timing.naked_single_calls += sub.timing.naked_single_calls;
        hasil.timing.hidden_single_calls += sub.timing.hidden_single_calls;
        hasil.timing.naked_subsets_calls += sub.timing.naked_subsets_calls;
        hasil.timing.locked_candidates_calls += sub.timing.locked_candidates_calls;
        hasil.timing.xwing_calls += sub.timing.xwing_calls;
        hasil.timing.probe_calls += sub.timing.probe_calls;
        
        if (sub.jumlah_solusi > 0) {
            hasil.langkah_pengisian.push_back(guess_log);
            for (auto& l : sub.langkah_pengisian) hasil.langkah_pengisian.push_back(l);
            if (hasil.solusi_pertama.empty()) hasil.solusi_pertama = sub.solusi_pertama;
            for (const auto& s : sub.semua_solusi) hasil.semua_solusi.push_back(s);
            if (!hitung_semua && hasil.jumlah_solusi >= batas) {
                trailUndo(nodeMark, papan, kand);
                hasil.langkah_propagasi += (step_counter - savedStep); // Accumulate current node propagasi
                hasil.total_elim += localElim;
                return hasil;
            }
        }
        
        trailUndo(branchMark, papan, kand);
        step_counter = branchStep;
    }
    
    trailUndo(nodeMark, papan, kand);
    hasil.langkah_propagasi += (step_counter - savedStep);
    hasil.total_elim += localElim;
    return hasil;
}

bool isValidCartesian(const string& papan, int tx, int ty, char num) {
    for (int x = -4; x <= 4; x++) if (x != tx && bacaKoordinat(papan, x, ty) == num) return false;
    for (int y = 4; y >= -4; y--) if (y != ty && bacaKoordinat(papan, tx, y) == num) return false;
    int r = 4 - ty, c = tx + 4;
    int sr = r - r % 3, sc = c - c % 3;
    for (int rr = 0; rr < 3; rr++)
        for (int cc = 0; cc < 3; cc++) {
            int cx = sc + cc - 4, cy = 4 - (sr + rr);
            if ((cx != tx || cy != ty) && bacaKoordinat(papan, cx, cy) == num) return false;
        }
    return true;
}

string global_papan(81, '0');
string original_papan(81, '0');
string solved_papan = "";
unsigned long current_session = 0;
vector<Kandidat> cached_kand;
bool has_cache = false;
unsigned long cache_session = 0;
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
    cout << "Contradiction Probe: " << t.probe_us << " µs | Calls: " << t.probe_calls << '\n';
    
    double total = t.naked_single_us + t.hidden_single_us + t.naked_subsets_us + 
                   t.locked_candidates_us + t.xwing_us + t.probe_us;
    cout << "\nTotal Technique Time: " << total << " µs (" << (total / 1000.0) << " ms)\n";
}

int main() {
    initStaticTables();
    
    auto do_calculate = [&]() -> bool {
        string namaFile = "data_koordinat_" + to_string(current_session) + ".txt";
        ofstream f(namaFile);
        if (!f.is_open()) return false;
        f << "SESSION_ID: " << current_session << "\n\n";
        f << "=== DATABASE DUAL-KOORDINAT (v10.6) ===\n\n";
        int n = 0;
        cached_kand = inisialisasiKandidat(global_papan);
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
                        if (cached_kand[idx].ada(num)) { f << num << " "; ada = true; }
                    if (!ada) f << "BUNTU";
                }
                f << " ]\n\n";
            }
        }
        f.close();
        has_cache = true;
        cache_session = current_session;
        cout << "SUKSES tersimpan ke " << namaFile << endl;
        return true;
    };

    while (true) {
        clearScreen();
        cout << "==========================================" << endl;
        cout << "  SUDOKU ENGINE v10.6 - PRECISION CYCLE  " << endl;
        cout << "  (MRV Metadata + Re-Propagation)        " << endl;
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
                has_cache = false; has_solve = false; solved_papan = "";
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
                current_session = 0; has_cache = has_solve = false; solved_papan = ""; showing_solved = false;
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
            for (char n = '1'; n <= '9'; n++) if (isValidCartesian(global_papan, x, y, n)) cout << n << " ";
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
            cout << "\n[Solver v10.6 - Precision Decision Cycle...]\n";
            
            vector<Kandidat> kand;
            if (has_cache && cache_session == current_session) {
                cout << "[INFO] Cache digunakan.\n";
                kand = cached_kand;
            } else {
                cout << "[INFO] Hitung kandidat baru.\n";
                kand = inisialisasiKandidat(global_papan);
            }
            
            g_trail.clear();
            auto t0 = chrono::high_resolution_clock::now();
            HasilSolver hasil = solveCerdas(global_papan, kand, false, 1, 0, 0, -1);
            double ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count();
            
            cout << "\n--- Statistik v10.6 ---\n";
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
                    f << "Contradiction Probe: " << hasil.timing.probe_us << " µs | Calls: " << hasil.timing.probe_calls << "\n";
                    double total = hasil.timing.naked_single_us + hasil.timing.hidden_single_us + 
                                   hasil.timing.naked_subsets_us + hasil.timing.locked_candidates_us + 
                                   hasil.timing.xwing_us + hasil.timing.probe_us;
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
            
            vector<Kandidat> kand;
            if (has_cache && cache_session == current_session) kand = cached_kand;
            else kand = inisialisasiKandidat(global_papan);
            
            g_trail.clear();
            HasilSolver hasil = solveCerdas(global_papan, kand, true, 9999, 0, 0, -1);
            
            cout << "\n--- Validasi v10.6 ---\n";
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
