#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <string_view>
#include <vector>
#include <unordered_set>
#include <cctype>
#include <iterator>

namespace {
struct Token { size_t begin, end; std::string_view text; };
Token nextToken(const std::string& text, size_t& position)
{
    while (position < text.size())
    {
        if (std::isspace(static_cast<unsigned char>(text[position]))) { ++position; continue; }
        if (text.compare(position, 2, "//") == 0)
        {
            position = text.find('\n', position);
            if (position == std::string::npos) position = text.size();
            continue;
        }
        if (text.compare(position, 2, "/*") == 0)
        {
            auto end = text.find("*/", position + 2);
            if (end == std::string::npos) throw std::runtime_error("Unterminated PTX comment");
            position = end + 2; continue;
        }
        break;
    }
    size_t begin = position;
    if (position < text.size())
    {
        char ch = text[position++];
        if (ch == '"')
        {
            bool closed = false;
            while (position < text.size())
            {
                char value = text[position++];
                if (value == '\\' && position < text.size()) ++position;
                else if (value == '"') { closed = true; break; }
            }
            if (!closed) throw std::runtime_error("Unterminated PTX string");
        }
        else if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '.' || ch == '$')
        {
            while (position < text.size())
            {
                char value = text[position];
                if (!std::isalnum(static_cast<unsigned char>(value)) && value != '_' && value != '.' && value != '$') break;
                ++position;
            }
        }
    }
    return {begin, position, std::string_view(text).substr(begin, position - begin)};
}
struct Entry { std::string name; size_t begin, end; };
std::vector<Entry> findEntries(const std::string& text)
{
    std::vector<Entry> entries;
    std::unordered_set<std::string> names;
    size_t position = 0;
    Token previous{};
    while (position < text.size())
    {
        auto token = nextToken(text, position);
        if (token.text == ".entry")
        {
            size_t begin = previous.text == ".visible" ? previous.begin : token.begin;
            auto name = nextToken(text, position);
            if (name.text.empty() || name.text.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$") != std::string_view::npos)
                throw std::runtime_error("Invalid PTX entry name");
            if (!names.insert(std::string(name.text)).second) throw std::runtime_error("Duplicate PTX entry");
            do { token = nextToken(text, position); }
            while (!token.text.empty() && token.text != "{" && token.text != ";");
            if (token.text != "{") throw std::runtime_error("Missing PTX entry body");
            unsigned depth = 1;
            while (depth)
            {
                token = nextToken(text, position);
                if (token.text.empty()) throw std::runtime_error("Unterminated PTX entry body");
                if (token.text == "{") ++depth;
                if (token.text == "}") --depth;
            }
            entries.push_back({std::string(name.text), begin, position});
        }
        previous = token;
    }
    if (entries.empty()) throw std::runtime_error("PTX contains no kernel entries");
    return entries;
}
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    try
    {
        if (argc != 3) throw std::runtime_error("Usage: neural_embed_ptx input.ptx output.h");
        const std::filesystem::path input(argv[1]), output(argv[2]);
        auto temporary = output; temporary += ".tmp";
        std::ifstream src(input, std::ios::binary);
        if (!src) throw std::runtime_error("Cannot open PTX input");
        std::string source((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
        if (src.bad()) throw std::runtime_error("Failed reading PTX");
        // C++ normalizes raw-literal newlines. Index that same normalized text
        // so offsets are correct for both Windows and Unix compiler output.
        std::string text; text.reserve(source.size());
        for (size_t i = 0; i < source.size(); ++i)
            if (source[i] != '\r' || i + 1 == source.size() || source[i + 1] != '\n') text += source[i];
        const auto entries = findEntries(text);
        if (text.find(")MPMATPTX\"") != std::string::npos) throw std::runtime_error("PTX conflicts with raw literal delimiter");
        std::ofstream dst(temporary, std::ios::binary | std::ios::trunc);
        if (!src || !dst) throw std::runtime_error("Cannot open PTX input/output");
        dst << "#pragma once\n#include <string>\n#include <string_view>\n#include <stdexcept>\nstatic const std::string mpmat_coop_ptx = []{std::string text; text.reserve("
            << text.size() << ");\n";
        for (size_t offset = 0; offset < text.size(); offset += 8000)
        {
            dst << "text.append(R\"MPMATPTX(";
            dst.write(text.data() + offset, std::min(size_t(8000), text.size() - offset));
            dst << ")MPMATPTX\");\n";
        }
        dst << "return text;}();\n";
        // Keep shared declarations/functions, but submit only the selected
        // kernel to OptiX. Building this index here avoids runtime PTX parsing.
        dst << "struct MpmatPtxEntry { const char* name; size_t begin, end; };\n"
               "static const MpmatPtxEntry mpmat_coop_entries[] = {\n";
        for (const auto& entry : entries)
            dst << "{\"" << entry.name << "\", " << entry.begin << ", " << entry.end << "},\n";
        dst << R"CPP(};
static std::string mpmat_coop_ptx_for_entry(std::string_view name)
{
    const MpmatPtxEntry* selected = nullptr;
    size_t commonSize = mpmat_coop_ptx.size();
    for (const auto& entry : mpmat_coop_entries)
    {
        commonSize -= entry.end - entry.begin;
        if (name == entry.name) selected = &entry;
    }
    if (!selected) throw std::runtime_error("Missing cooperative PTX entry: " + std::string(name));
    std::string result; result.reserve(commonSize + selected->end - selected->begin);
    size_t previous = 0;
    for (const auto& entry : mpmat_coop_entries)
    {
        result.append(mpmat_coop_ptx, previous, entry.begin - previous);
        if (&entry == selected) result.append(mpmat_coop_ptx, entry.begin, entry.end - entry.begin);
        previous = entry.end;
    }
    result.append(mpmat_coop_ptx, previous, std::string::npos);
    return result;
}
)CPP";
        dst.close();
        if (!dst) throw std::runtime_error("Failed writing embedded PTX header");
        std::error_code error;
        for (unsigned attempt = 0; attempt != 40; ++attempt)
        {
            std::filesystem::rename(temporary, output, error);
            if (!error) return 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        throw std::runtime_error("Cannot replace embedded PTX header: " + error.message());
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
