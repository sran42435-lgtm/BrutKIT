#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp> // WAJIB DI OPENCV 5 untuk getPerspectiveTransform

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <chrono>
#include <sstream>
#include <cmath>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

class SudokuExtractor {
private:
    static constexpr int GRID_SIZE = 9;
    static constexpr int BOARD_SIZE = 900;
    static constexpr bool ALLOW_SAVE_WITH_UNCERTAIN = true;

    Mat original;
    Mat board;
    array<Point2f, 4> corners;
    string inputPath;
    string outputPath;
    fs::path tempDir;

    struct OcrResult { char digit = '?'; float confidence = -1.0f; };

    string shellQuote(const string& value) {
#ifdef _WIN32
        string result = "\"";
        for (char c : value) {
            if (c == '"' || c == '\\') result += '\\';
            result += c;
        }
        return result + "\"";
#else
        string result = "'";
        for (char c : value) {
            if (c == '\'') result += "'\\''";
            else result += c;
        }
        return result + "'";
#endif
    }

    bool loadImage() {
        original = imread(inputPath, IMREAD_COLOR);
        if (original.empty()) {
            cerr << "ERROR: Gambar tidak dapat dibaca.\n";
            return false;
        }
        cout << "[OK] Image loaded: " << original.cols << "x" << original.rows << '\n';
        return true;
    }

    array<Point2f, 4> orderCorners(const vector<Point>& points) {
        array<Point2f, 4> result;
        float minSum = 1e9f, maxSum = -1e9f;
        float minDiff = 1e9f, maxDiff = -1e9f;

        for (const auto& p : points) {
            float sum = static_cast<float>(p.x + p.y);
            float diff = static_cast<float>(p.y - p.x);
            if (sum < minSum) { minSum = sum; result[0] = p; }
            if (sum > maxSum) { maxSum = sum; result[2] = p; }
            if (diff < minDiff) { minDiff = diff; result[1] = p; }
            if (diff > maxDiff) { maxDiff = diff; result[3] = p; }
        }
        return result;
    }

    bool detectGrid() {
        Mat gray, blurred, binary;
        cvtColor(original, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(5, 5), 0);
        adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);

        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        double largestArea = 0;
        vector<Point> bestContour;
        double imageArea = static_cast<double>(original.cols) * original.rows;

        for (const auto& contour : contours) {
            // PERBAIKAN OPENCV 5: Gunakan cv::contourArea
            double area = cv::contourArea(contour);
            if (area < imageArea * 0.08) continue;

            // PERBAIKAN OPENCV 5: Gunakan cv::arcLength
            double perimeter = cv::arcLength(contour, true);
            vector<Point> approx;
            
            // PERBAIKAN OPENCV 5: Gunakan cv::approxPolyDP
            cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

            if (approx.size() != 4) continue;

            // PERBAIKAN OPENCV 5: Gunakan cv::isContourConvex
            if (!cv::isContourConvex(approx)) continue;

            // PERBAIKAN OPENCV 5: Gunakan cv::boundingRect
            Rect bounds = cv::boundingRect(approx);
            double ratio = static_cast<double>(bounds.width) / max(1, bounds.height);

            if (ratio < 0.65 || ratio > 1.45) continue;
            if (area > largestArea) {
                largestArea = area;
                bestContour = approx;
            }
        }

        if (bestContour.empty()) {
            cout << "[WARNING] Grid contour tidak ditemukan. Menggunakan seluruh gambar.\n";
            corners = {{
                Point2f(0, 0),
                Point2f(static_cast<float>(original.cols - 1), 0),
                Point2f(static_cast<float>(original.cols - 1), static_cast<float>(original.rows - 1)),
                Point2f(0, static_cast<float>(original.rows - 1))
            }};
            return true;
        }

        corners = orderCorners(bestContour);
        cout << "[OK] Sudoku grid detected.\n";
        return true;
    }

    bool correctPerspective() {
        vector<Point2f> source(corners.begin(), corners.end());
        vector<Point2f> destination = {
            Point2f(0, 0),
            Point2f(static_cast<float>(BOARD_SIZE - 1), 0),
            Point2f(static_cast<float>(BOARD_SIZE - 1), static_cast<float>(BOARD_SIZE - 1)),
            Point2f(0, static_cast<float>(BOARD_SIZE - 1))
        };

        // PERBAIKAN OPENCV 5: Wajib dari modul calib3d
        Mat transform = cv::getPerspectiveTransform(source, destination);
        warpPerspective(original, board, transform, Size(BOARD_SIZE, BOARD_SIZE));
        
        cout << "[OK] Perspective correction completed.\n";
        return !board.empty();
    }

    void drawGrid(Mat& img) {
        int cellSize = BOARD_SIZE / GRID_SIZE;
        for (int i = 0; i <= GRID_SIZE; ++i) {
            int pos = min(BOARD_SIZE - 1, i * cellSize);
            int thickness = (i % 3 == 0) ? 4 : 2;
            line(img, Point(pos, 0), Point(pos, BOARD_SIZE - 1), Scalar(0, 0, 255), thickness, LINE_AA);
            line(img, Point(0, pos), Point(BOARD_SIZE - 1, pos), Scalar(0, 0, 255), thickness, LINE_AA);
        }
        rectangle(img, Rect(0, 0, BOARD_SIZE, BOARD_SIZE), Scalar(0, 0, 255), 5, LINE_AA);
    }

    fs::path makeDebugPath(const string& suffix) {
        fs::path out(outputPath);
        string stem = out.stem().string();
        if (stem.empty()) stem = "sudoku";
        fs::path dir = out.parent_path();
        fs::path filename(stem + suffix + ".png");
        return dir.empty() ? filename : dir / filename;
    }

    void saveGridDebug() {
        if (board.empty()) return;
        Mat debug;
        if (board.channels() == 1) cvtColor(board, debug, COLOR_GRAY2BGR);
        else debug = board.clone();
        drawGrid(debug);
        fs::path debugPath = makeDebugPath("_grid");
        if (imwrite(debugPath.string(), debug))
            cout << "[OK] Garis kotak diperjelas & disimpan di: " << debugPath.string() << '\n';
    }

    void saveResultDebug(const string& extracted) {
        if (board.empty() || extracted.size() != 81) return;
        Mat debug;
        if (board.channels() == 1) cvtColor(board, debug, COLOR_GRAY2BGR);
        else debug = board.clone();
        drawGrid(debug);

        int cellSize = BOARD_SIZE / GRID_SIZE;
        for (int row = 0; row < GRID_SIZE; ++row) {
            for (int col = 0; col < GRID_SIZE; ++col) {
                char c = extracted[row * GRID_SIZE + col];
                string text(1, c);
                int baseline = 0;
                Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 1.3, 3, &baseline);
                int x = col * cellSize + (cellSize - textSize.width) / 2;
                int y = row * cellSize + (cellSize + textSize.height) / 2;
                Scalar color = (c == '0') ? Scalar(120, 120, 120) : Scalar(255, 0, 0);
                putText(debug, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 1.3, color, 3, LINE_AA);
            }
        }
        imwrite(makeDebugPath("_hasil").string(), debug);
    }

    Mat normalizeCell(const Mat& cell) {
        Mat gray;
        if (cell.channels() == 3) cvtColor(cell, gray, COLOR_BGR2GRAY);
        else gray = cell.clone();

        vector<uchar> borderPixels;
        const int thickness = max(1, min(gray.rows, gray.cols) / 10);
        for (int y = 0; y < gray.rows; ++y) {
            for (int x = 0; x < gray.cols; ++x) {
                if (x < thickness || x >= gray.cols - thickness || y < thickness || y >= gray.rows - thickness)
                    borderPixels.push_back(gray.at<uchar>(y, x));
            }
        }

        uchar backgroundMedian = 255;
        if (!borderPixels.empty()) {
            size_t middle = borderPixels.size() / 2;
            nth_element(borderPixels.begin(), borderPixels.begin() + middle, borderPixels.end());
            backgroundMedian = borderPixels[middle];
        }

        if (backgroundMedian < 127) bitwise_not(gray, gray);
        GaussianBlur(gray, gray, Size(3, 3), 0);
        return gray;
    }

    double borderBlackRatio(const Mat& binary) {
        if (binary.empty()) return 0.0;
        int border = max(1, min(binary.rows, binary.cols) / 8);
        int total = 0, black = 0;
        for (int y = 0; y < binary.rows; ++y) {
            for (int x = 0; x < binary.cols; ++x) {
                if (x < border || x >= binary.cols - border || y < border || y >= binary.rows - border) {
                    total++;
                    if (binary.at<uchar>(y, x) < 127) black++;
                }
            }
        }
        return total == 0 ? 0.0 : static_cast<double>(black) / total;
    }

    Mat ensureWhiteBackground(const Mat& binary) {
        Mat out = binary.clone();
        if (borderBlackRatio(out) > 0.5) bitwise_not(out, out);
        return out;
    }

    Mat binarizeCell(const Mat& gray) {
        if (gray.empty()) return Mat();
        Mat otsu;
        threshold(gray, otsu, 0, 255, THRESH_BINARY | THRESH_OTSU);
        otsu = ensureWhiteBackground(otsu);

        Mat fg; bitwise_not(otsu, fg);
        double fgRatio = countNonZero(fg) / static_cast<double>(fg.total());

        if (fgRatio > 0.35) {
            Mat adaptive;
            int blockSize = max(15, min(gray.rows, gray.cols) / 4);
            if (blockSize % 2 == 0) ++blockSize;
            adaptiveThreshold(gray, adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize, 15);
            adaptive = ensureWhiteBackground(adaptive);
            medianBlur(adaptive, adaptive, 3);
            return adaptive;
        }
        medianBlur(otsu, otsu, 3);
        return otsu;
    }

    void clearBorder(Mat& mask, int border) {
        if (mask.empty()) return;
        int b = min(border, min(mask.rows, mask.cols) / 2);
        if (b <= 0) return;
        rectangle(mask, Rect(0, 0, mask.cols, b), Scalar(0), FILLED);
        rectangle(mask, Rect(0, mask.rows - b, mask.cols, b), Scalar(0), FILLED);
        rectangle(mask, Rect(0, 0, b, mask.rows), Scalar(0), FILLED);
        rectangle(mask, Rect(mask.cols - b, 0, b, mask.rows), Scalar(0), FILLED);
    }

    Rect expandRect(const Rect& r, int pad, Size size) {
        int x1 = max(0, r.x - pad);
        int y1 = max(0, r.y - pad);
        int x2 = min(size.width, r.x + r.width + pad);
        int y2 = min(size.height, r.y + r.height + pad);
        return Rect(x1, y1, x2 - x1, y2 - y1);
    }

    bool findBestDigitBox(const Mat& binary, Rect& outBox, double& outArea) {
        if (binary.empty()) return false;
        Mat fg; bitwise_not(binary, fg);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
        morphologyEx(fg, fg, MORPH_OPEN, kernel, Point(-1, -1), 1);
        morphologyEx(fg, fg, MORPH_CLOSE, kernel, Point(-1, -1), 1);
        clearBorder(fg, max(2, min(fg.rows, fg.cols) / 10));

        vector<vector<Point>> contours;
        findContours(fg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours.empty()) return false;

        float centerX = fg.cols / 2.0f, centerY = fg.rows / 2.0f;
        float maxDist = max(fg.cols, fg.rows) / 2.0f;
        double totalArea = static_cast<double>(fg.total());
        double bestScore = -1.0;

        for (const auto& contour : contours) {
            // PERBAIKAN OPENCV 5: cv::contourArea
            double area = cv::contourArea(contour);
            if (area < totalArea * 0.002) continue;

            // PERBAIKAN OPENCV 5: cv::boundingRect
            Rect r = cv::boundingRect(contour);
            if (r.width < 3 || r.height < 3) continue;

            float aspect = static_cast<float>(r.width) / max(1, r.height);
            if (aspect < 0.15f || aspect > 6.0f) continue;
            if (area > totalArea * 0.85) continue;

            float cx = r.x + r.width / 2.0f, cy = r.y + r.height / 2.0f;
            float dist = sqrtf((cx - centerX) * (cx - centerX) + (cy - centerY) * (cy - centerY));
            float central = 1.0f - min(1.0f, dist / maxDist);
            double score = (area / totalArea) * (0.30 + 0.70 * central);
            if (aspect > 3.5f || aspect < 0.25f) score *= 0.25;

            if (score > bestScore) {
                bestScore = score;
                outBox = r;
                outArea = area;
            }
        }
        return bestScore > 0.0;
    }

    bool prepareOcrImage(const Mat& cell, Mat& ocrImage) {
        if (cell.empty()) return false;
        Mat gray = normalizeCell(cell);
        Mat binary = binarizeCell(gray);
        Rect box; double area = 0.0;
        if (!findBestDigitBox(binary, box, area)) return false;
        if (area < max(25, static_cast<int>(cell.total() * 0.004))) return false;

        Rect roi = expandRect(box, max(3, static_cast<int>(0.18f * max(box.width, box.height))), binary.size());
        Mat digit = binary(roi).clone();
        int side = max(digit.rows, digit.cols);
        Mat square(side, side, CV_8UC1, Scalar(255));
        digit.copyTo(square(Rect((side - digit.cols) / 2, (side - digit.rows) / 2, digit.cols, digit.rows)));

        resize(square, square, Size(160, 160), 0, 0, square.cols > 160 ? INTER_AREA : INTER_CUBIC);
        threshold(square, square, 0, 255, THRESH_BINARY | THRESH_OTSU);
        square = ensureWhiteBackground(square);
        copyMakeBorder(square, square, 70, 70, 70, 70, BORDER_CONSTANT, Scalar(255));
        ocrImage = square;
        return true;
    }

    Mat makeFullOcrImage(const Mat& cell) {
        Mat gray = normalizeCell(cell);
        Mat binary = binarizeCell(gray);
        resize(binary, binary, Size(), 4.0, 4.0, INTER_CUBIC);
        copyMakeBorder(binary, binary, 60, 60, 60, 60, BORDER_CONSTANT, Scalar(255));
        threshold(binary, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
        return ensureWhiteBackground(binary);
    }

    bool isEmptyCell(const Mat& cell) {
        if (cell.empty()) return true;
        Mat gray = normalizeCell(cell);
        Mat binary = binarizeCell(gray);
        Mat fg; bitwise_not(binary, fg);
        morphologyEx(fg, fg, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(2, 2)), Point(-1, -1), 1);
        clearBorder(fg, max(2, min(fg.rows, fg.cols) / 8));
        return countNonZero(fg) < max(25, static_cast<int>(cell.total() * 0.005));
    }

    OcrResult runTesseract(const fs::path& imageFile, const fs::path& outputBase, int psm, int oem) {
        OcrResult best;
        fs::path tsvFile(outputBase.string() + ".tsv");
        fs::path txtFile(outputBase.string() + ".txt");

#ifdef _WIN32
        string redirect = " 2>NUL";
#else
        string redirect = " 2>/dev/null";
#endif

        string command = "tesseract " + shellQuote(imageFile.string()) + " " + shellQuote(outputBase.string()) +
                         " --oem " + to_string(oem) + " --psm " + to_string(psm) +
                         " -l eng -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1 txt tsv" + redirect;
        system(command.c_str());

        ifstream tsv(tsvFile);
        if (tsv.is_open()) {
            string line; getline(tsv, line);
            while (getline(tsv, line)) {
                if (line.empty()) continue;
                stringstream ss(line);
                int level, page_num, block_num, par_num, line_num, word_num, left, top, width, height;
                float conf;
                ss >> level >> page_num >> block_num >> par_num >> line_num >> word_num >> left >> top >> width >> height >> conf;
                if (!ss) continue;
                string text; ss >> text;
                for (char c : text) {
                    if (c >= '1' && c <= '9') {
                        float usedConf = max(0.0f, conf);
                        if (usedConf > best.confidence) { best.digit = c; best.confidence = usedConf; }
                        break;
                    }
                }
            }
        }
        if (best.digit != '?') return best;

        ifstream txt(txtFile);
        if (txt.is_open()) {
            string result; getline(txt, result);
            for (char c : result) {
                if (c >= '1' && c <= '9') { best.digit = c; best.confidence = 1.0f; break; }
            }
        }
        return best;
    }

    char recognizeFromImage(const Mat& img, int index, const string& suffix) {
        if (img.empty()) return '?';
        fs::path imageFile = tempDir / ("cell_" + to_string(index) + suffix + ".png");
        if (!imwrite(imageFile.string(), img)) return '?';

        OcrResult best;
        auto tryConfig = [&, this](int psm, int oem) {
            fs::path outputBase = tempDir / ("ocr_" + to_string(index) + suffix + "_oem" + to_string(oem) + "_psm" + to_string(psm));
            OcrResult current = runTesseract(imageFile, outputBase, psm, oem);
            if (current.digit != '?' && current.confidence > best.confidence) best = current;
        };

        tryConfig(10, 1);
        if (best.digit != '?' && best.confidence >= 85.0f) return best.digit;
        tryConfig(7, 1); tryConfig(13, 1);
        if (best.digit != '?' && best.confidence >= 80.0f) return best.digit;
        tryConfig(10, 3); tryConfig(13, 3);
        return best.digit;
    }

    bool extractCells() {
        vector<string> unknownCells;
        string extracted; extracted.reserve(81);
        const int cellSize = BOARD_SIZE / GRID_SIZE;
        cout << "\nRecognizing digits...\n";
        int emptyCount = 0, recognizedCount = 0, uncertainCount = 0, index = 0;

        for (int row = 0; row < GRID_SIZE; ++row) {
            for (int col = 0; col < GRID_SIZE; ++col) {
                int margin = cellSize * 12 / 100;
                Rect region(col * cellSize + margin, row * cellSize + margin, cellSize - 2 * margin, cellSize - 2 * margin);
                Mat cell = board(region).clone();
                char digit = '?';

                if (isEmptyCell(cell)) {
                    digit = '0'; emptyCount++;
                } else {
                    Mat ocrImage;
                    if (prepareOcrImage(cell, ocrImage)) digit = recognizeFromImage(ocrImage, index, "");
                    if (digit == '?') digit = recognizeFromImage(makeFullOcrImage(cell), index, "_full");

                    if (digit == '?') {
                        unknownCells.push_back("Row " + to_string(row + 1) + ", Column " + to_string(col + 1));
                        uncertainCount++;
                        if (ALLOW_SAVE_WITH_UNCERTAIN) digit = '0';
                    } else if (digit == '0') emptyCount++;
                    else recognizedCount++;
                }
                extracted.push_back(digit);
                index++;
            }
        }

        cout << "\n========== EXTRACTION REPORT ==========\n";
        cout << "Total cells : " << extracted.size() << '\n';
        cout << "Empty cells : " << emptyCount << '\n';
        cout << "Recognized  : " << recognizedCount << '\n';
        cout << "Uncertain   : " << uncertainCount << '\n';

        if (!unknownCells.empty()) {
            cout << "\nCells requiring correction:\n";
            for (const auto& cell : unknownCells) cout << " - " << cell << '\n';
            if (!ALLOW_SAVE_WITH_UNCERTAIN) {
                cout << "\nExtraction aborted: OCR uncertain.\n";
                return false;
            }
            cout << "\n[WARNING] Sel gagal OCR ditulis sebagai 0.\n";
        }

        if (extracted.size() != 81) { cerr << "ERROR: Invalid cell count.\n"; return false; }

        fs::path destination(outputPath);
        if (!destination.parent_path().empty()) fs::create_directories(destination.parent_path());
        ofstream output(outputPath);
        if (!output.is_open()) { cerr << "ERROR: Cannot create output file.\n"; return false; }

        // OUTPUT PANJANG KE SAMPING (BUKAN TURUN KEBAWAH)
        output << extracted << '\n';

        if (!output.good()) { cerr << "ERROR: Failed writing output file.\n"; return false; }

        cout << "\n[OK] Extraction completed!\nOutput: " << outputPath << '\n';
        saveResultDebug(extracted);

        cout << "\nExtracted Sudoku (Visual):\n";
        for (int i = 0; i < 81; ++i) {
            cout << extracted[i];
            if (i % 9 == 8) cout << '\n';
        }
        return true;
    }

public:
    bool run() {
        cout << "====================================\n       SUDOKU IMAGE EXTRACTOR\n====================================\n\n";
        cout << "Input image path: "; getline(cin, inputPath);
        cout << "Output file name: "; getline(cin, outputPath);
        if (inputPath.empty() || outputPath.empty()) { cerr << "ERROR: Invalid input.\n"; return false; }

        try {
            if (fs::path(outputPath).extension().empty()) outputPath += ".txt";
            fs::path outParent = fs::path(outputPath).parent_path();
            if (!outParent.empty()) fs::create_directories(outParent);

            if (!loadImage()) return false;
            auto timestamp = chrono::steady_clock::now().time_since_epoch().count();
            tempDir = fs::temp_directory_path() / ("sudoku_ocr_" + to_string(timestamp));
            fs::create_directories(tempDir);

            if (!detectGrid()) { fs::remove_all(tempDir); return false; }
            if (!correctPerspective()) { fs::remove_all(tempDir); return false; }
            
            saveGridDebug(); // Memperjelas garis kotak
            
            bool success = extractCells();
            fs::remove_all(tempDir);
            return success;
        } catch (const exception& e) {
            cerr << "ERROR: " << e.what() << '\n';
            error_code ec;
            if (!tempDir.empty()) fs::remove_all(tempDir, ec);
            return false;
        }
    }
};

int main() {
    SudokuExtractor extractor;
    return extractor.run() ? 0 : 1;
}
