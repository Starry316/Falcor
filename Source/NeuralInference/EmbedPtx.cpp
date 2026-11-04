#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

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
        std::ofstream dst(temporary, std::ios::binary | std::ios::trunc);
        if (!src || !dst) throw std::runtime_error("Cannot open PTX input/output");
        // Identical string and literal boundaries to the original generator.
        // One streaming write avoids both Windows append-open races and CMake's
        // quadratic string concatenation for the large cooperative device code.
        dst << "#pragma once\n#include <string>\nstatic const std::string mpmat_coop_ptx = []{std::string text; text.reserve("
            << std::filesystem::file_size(input) << ");\n";
        std::array<char, 8000> chunk;
        while (src.read(chunk.data(), chunk.size()) || src.gcount())
        {
            dst << "text.append(R\"MPMATPTX(";
            dst.write(chunk.data(), src.gcount());
            dst << ")MPMATPTX\");\n";
        }
        if (!src.eof()) throw std::runtime_error("Failed reading PTX");
        dst << "return text;}();\n";
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
