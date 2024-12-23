#include "Core/Error.h"
#include "IOHelper.h"
#include <fstream>
namespace Falcor
{
std::vector<float> readBinaryFile(const char* filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        logError("[IOHelper] Unable to open file {}", filename);
        return std::vector<float>();
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        logError("[IOHelper] Error reading file {}", filename);
        return std::vector<float>();
    }
    file.close();
    return buffer;
}
}
