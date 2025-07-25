
#include "Histogram.h"

#include "external/Utils.h"

#include <format>
FALCOR_EXPORT_D3D12_AGILITY_SDK
static void imwrite(const std::filesystem::path& path, uint32_t width, uint32_t height, float* pData)
{
    Bitmap::saveImage(
        path, width, height, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Uncompressed, ResourceFormat::RGB32Float, false, pData
    );
}

void extractBTF(BTF* pBTF, int viewID, int lightID, std::vector<float>& data)
{
    if (viewID >= 151 || viewID < 0 || lightID >= 151 || lightID < 0)
    {
        logError("Invalid view or light index");
        return;
    }
    for (uint32_t btf_y = 0; btf_y < pBTF->Height; ++btf_y)
    {
        for (uint32_t btf_x = 0; btf_x < pBTF->Width; ++btf_x)
        {
            auto spec = BTFFetchSpectrum(pBTF, lightID, viewID, btf_x, btf_y);
            int idx = ((pBTF->Height - btf_y - 1) * pBTF->Width + btf_x) * 3;
            data[idx + 0] = spec.x;
            data[idx + 1] = spec.y;
            data[idx + 2] = spec.z;
        }
    }
}

void extractBTF(BTF* pBTF, int viewID, int lightID, std::vector<float>& data, uint offset)
{
    if (viewID >= 151 || viewID < 0 || lightID >= 151 || lightID < 0)
    {
        logError("Invalid view or light index");
        return;
    }
    uint count = 0;
    for (uint32_t btf_y = 0; btf_y < pBTF->Height; ++btf_y)
    {
        for (uint32_t btf_x = 0; btf_x < pBTF->Width; ++btf_x)
        {
            auto spec = BTFFetchSpectrum(pBTF, lightID, viewID, btf_x, pBTF->Height - btf_y - 1);

            data[offset + count] = spec.x;
            data[offset + count + 1] = spec.y;
            data[offset + count + 2] = spec.z;
            count += 3;
        }
    }
}

float lookUp(float x, TextureDataFloat& lut, uint channel)
{
    uint length = lut.width;
    float pos = x * (length - 1);
    uint left = fmin(floor(pos), length - 1);
    uint right = left + 1;
    float a = lut.GetPixel(left, 0, channel);
    float b = lut.GetPixel(right, 0, channel);
    return math::lerp(a, b, pos - left);
}

void reconstructSingle()
{
    // auto pBitmap = Bitmap::createFromFile("C:/Projects/NNMat/data/leather11_histo/0.exr", false);
    auto pBitmap = Bitmap::createFromFile("C:/Projects/NNMat/outputs/Histo/BTFMLP_8x8_leather11_U8_H4_D4_12_L1/test_354_0.exr", false);
    uint width = pBitmap->getWidth();
    uint height = pBitmap->getHeight();
    logInfo("{} {}", width, height);

    auto pLUTBitmap = Bitmap::createFromFile("C:/Projects/NNMat/data/leather11_histo/LUT_0.exr", true);
    float* pData = (float*)pBitmap->getData();
    float* pLUTData = (float*)pLUTBitmap->getData();
    TextureDataFloat lut(160000, 1, 3);
    for (size_t i = 0; i < 160000; i++)
    {
        lut.SetPixel(i, 0, 0, pLUTData[4 * i]);
        lut.SetPixel(i, 0, 1, pLUTData[4 * i + 1]);
        lut.SetPixel(i, 0, 2, pLUTData[4 * i + 2]);
    }
    float* recon = new float[height * width * 3];
    for (size_t x = 0; x < height * width; x++)
    {
        recon[3 * x + 0] = lookUp(pData[4 * x + 0], lut, 0);
        recon[3 * x + 1] = lookUp(pData[4 * x + 1], lut, 1);
        recon[3 * x + 2] = lookUp(pData[4 * x + 2], lut, 2);
    }

    imwrite(fmt::format("C:/Projects/NNMat/outputs/recon.exr"), width, height, recon);
    // imwrite(fmt::format("C:/Projects/NNMat/data/test/recon.exr"), 40, 40, recon);
}

void reconstructSingle(std::string imgPath, std::string LUTPath, std::string outPath)
{
    // auto pBitmap = Bitmap::createFromFile("C:/Projects/NNMat/data/leather11_histo/0.exr", false);
    auto pBitmap = Bitmap::createFromFile(imgPath, false);
    uint width = pBitmap->getWidth();
    uint height = pBitmap->getHeight();
    logInfo("{} {}", width, height);

    auto pLUTBitmap = Bitmap::createFromFile(LUTPath, true);
    float* pData = (float*)pBitmap->getData();
    float* pLUTData = (float*)pLUTBitmap->getData();
    TextureDataFloat lut(160000, 1, 3);
    for (size_t i = 0; i < 160000; i++)
    {
        lut.SetPixel(i, 0, 0, pLUTData[4 * i]);
        lut.SetPixel(i, 0, 1, pLUTData[4 * i + 1]);
        lut.SetPixel(i, 0, 2, pLUTData[4 * i + 2]);
    }
    float* recon = new float[height * width * 3];
    for (size_t x = 0; x < height * width; x++)
    {
        recon[3 * x + 0] = lookUp(pData[4 * x + 0], lut, 0);
        recon[3 * x + 1] = lookUp(pData[4 * x + 1], lut, 1);
        recon[3 * x + 2] = lookUp(pData[4 * x + 2], lut, 2);
    }

    imwrite(fmt::format(outPath), width, height, recon);
}

int runMain(int argc, char** argv)
{
    // for (size_t i = 0; i < 300; i++)
    // {
    //     reconstructSingle(
    //         fmt::format("C:/Projects/NNMat/outputs/test/{}.exr", i),
    //         fmt::format("C:/Projects/NNMat/data/leather11_histo/LUT_{}.exr", i / 151),
    //         fmt::format("C:/Projects/NNMat/outputs/recon/{}.exr", i)
    //     );
    // }
    // return 0;

    // reconstructSingle();
    // return 0;

    std::string BTFName = "leather11";
    std::string inputPath = fmt::format("C:/Projects/NNMat/data/{}_W400xH400_L151xV151.btf", BTFName);
    auto pBTF = LoadBTF(inputPath.c_str());

    int height = 400;
    int width = 400;
    int imageSize = 3 * height * width;

    TextureDataFloat input(151, width * height, 3);
    TextureDataFloat recon(width, height, 3);
    TextureDataFloat Tinput;
    TextureDataFloat lut;

    for (size_t j = 0; j < 151; j++)
    {
        logInfo("{} / {}", j, 151);
        for (size_t i = 0; i < 151; i++)
        {
            uint offset = i * imageSize;
            extractBTF(pBTF, j, i, input.data, offset);
        }

        logInfo("[Synthesis] Precomputing Gaussian T and Inv.");
        Precomputations(input, Tinput, lut);
        logInfo("[Synthesis] Precomputation done!");
        logInfo("[Synthesis] Outputing!");
        imwrite(fmt::format("C:/Projects/NNMat/data/leather11_histo/LUT_{}.exr", j), height, width, lut.data.data());

        for (size_t i = 0; i < 151; i++)
        {
            uint offset = i * imageSize;
            uint count = j * 151 + i;
            imwrite(fmt::format("C:/Projects/NNMat/data/leather11_histo/{}.exr", count), height, width, Tinput.data.data() + offset);
        }

        logInfo("[Synthesis] Output done!");
    }

    return 0;
}

int histogramBasedOnH()
{
    std::string BTFName = "leather11";
    std::string inputPath = fmt::format("C:/Projects/NNMat/data/{}_W400xH400_L151xV151.btf", BTFName);
    auto pBTF = LoadBTF(inputPath.c_str());

    int height = 400;
    int width = 400;
    int imageSize = 3 * height * width;

    for (int binID = 0; binID < 42; binID++)
    {
        // first round, count how many data inside the bin
        int count = 0;
        for (int viewID = 0; viewID < 151; viewID++)
        {
            for (int lightID = 0; lightID < 151; lightID++)
            {
                float3 l = float3(pBTF->Lights[lightID].x, pBTF->Lights[lightID].y, pBTF->Lights[lightID].z);
                float3 v = float3(pBTF->Views[viewID].x, pBTF->Views[viewID].y, pBTF->Views[viewID].z);
                float3 h = normalize(l + v);
                // Normalized theta [0, 1]
                float theta = acosf(h.z) / M_PI * 2;
                if (theta >= binID * 0.02f && theta < binID * 0.02f + 0.02f)
                {
                    count++;
                }
            }
        }
        if (count == 0)
            continue;
        logInfo("Bin: {} Count: {}", binID, count);

        TextureDataFloat input(count, width * height, 3);
        TextureDataFloat Tinput;
        TextureDataFloat lut;
        uint* indexList = new uint[count];
        uint globalCount = 0;
        count = 0;
        for (int viewID = 0; viewID < 151; viewID++)
        {
            for (int lightID = 0; lightID < 151; lightID++)
            {
                float3 l = float3(pBTF->Lights[lightID].x, pBTF->Lights[lightID].y, pBTF->Lights[lightID].z);
                float3 v = float3(pBTF->Views[viewID].x, pBTF->Views[viewID].y, pBTF->Views[viewID].z);
                float3 h = normalize(l + v);
                float theta = acosf(h.z) / M_PI * 2;
                if (theta >= binID * 0.02f && theta < binID * 0.02f + 0.02f)
                {
                    uint offset = count * imageSize;
                    extractBTF(pBTF, viewID, lightID, input.data, offset);
                    indexList[count] = globalCount;
                    count++;
                }
                globalCount++;
            }
        }
        logInfo("[Synthesis] Precomputing Gaussian T and Inv.");
        Precomputations(input, Tinput, lut);
        logInfo("[Synthesis] Precomputation done!");
        imwrite(fmt::format("C:/Projects/NNMat/data/leather11_histo_h/LUT_{}.exr", binID), height, width, lut.data.data());

        for (size_t i = 0; i < count; i++)
        {
            uint offset = i * imageSize;
            imwrite(
                fmt::format("C:/Projects/NNMat/data/leather11_histo_h/{}.exr", indexList[i]), height, width, Tinput.data.data() + offset
            );
        }
        // delete input.data.data();
        // delete indexList;
    }
    return 0;
}

int histogramBasedOnWi(int startBin)
{
    std::string BTFName = "leather11";
    std::string inputPath = fmt::format("C:/Projects/NNMat/data/{}_W400xH400_L151xV151.btf", BTFName);
    auto pBTF = LoadBTF(inputPath.c_str());

    int height = 400;
    int width = 400;
    int imageSize = 3 * height * width;

    for (int binID = startBin; binID < startBin + 1; binID++)
    {
        // first round, count how many data inside the bin
        int count = 0;
        for (int viewID = 0; viewID < 151; viewID++)
        {
            for (int lightID = 0; lightID < 151; lightID++)
            {
                float3 l = float3(pBTF->Lights[lightID].x, pBTF->Lights[lightID].y, pBTF->Lights[lightID].z);
                float3 v = float3(pBTF->Views[viewID].x, pBTF->Views[viewID].y, pBTF->Views[viewID].z);
                // Normalized theta [0, 1]
                float theta = acosf(l.z) / M_PI * 2;
                // float theta = l.z;
                if (theta >= binID * 0.1f && theta < binID * 0.1f + 0.1f)
                {
                    count++;
                }
            }
        }
        if (count == 0)
            continue;
        logInfo("Bin: {} Range: [{}, {}) Count: {}", binID, binID * 0.1f, binID * 0.1f + 0.1f, count);
        // if (binID < 195)
        //     continue;
        TextureDataFloat input(count, width * height, 3);
        // logInfo("{}  {}",input.data.size(), count * width * height * 3);
        TextureDataFloat Tinput;
        TextureDataFloat lut;
        uint* indexList = new uint[count];
        uint globalCount = 0;
        count = 0;
        for (int viewID = 0; viewID < 151; viewID++)
        {
            for (int lightID = 0; lightID < 151; lightID++)
            {
                float3 l = float3(pBTF->Lights[lightID].x, pBTF->Lights[lightID].y, pBTF->Lights[lightID].z);
                float3 v = float3(pBTF->Views[viewID].x, pBTF->Views[viewID].y, pBTF->Views[viewID].z);
                // float theta = l.z;
                float theta = acosf(l.z) / M_PI * 2;
                if (theta >= binID * 0.1f && theta < binID * 0.1f + 0.1f)
                {
                    uint offset = count * imageSize;
                    extractBTF(pBTF, viewID, lightID, input.data, offset);
                    indexList[count] = globalCount;
                    count++;
                }
                globalCount++;
            }
        }
        logInfo("[Synthesis] Precomputing Gaussian T and Inv.");
        Precomputations(input, Tinput, lut);
        logInfo("[Synthesis] Precomputation done!");
        imwrite(fmt::format("C:/Projects/NNMat/data/leather11_histo_wi/LUT_{}.exr", binID), height, width, lut.data.data());

        for (size_t i = 0; i < count; i++)
        {
            if (i % 10 == 1)
                logInfo("{} / {}", i, count);
            uint offset = i * imageSize;
            imwrite(
                fmt::format("C:/Projects/NNMat/data/leather11_histo_wi/{}.exr", indexList[i]), height, width, Tinput.data.data() + offset
                // fmt::format("C:/Projects/NNMat/data/leather11_histo_wi/{}.exr", indexList[i]), height, width, input.data.data() + offset
            );
        }
        // break;
    }
    return 0;
}

int histogramTest(int startBin)
{
    std::string BTFName = "leather11";
    std::string inputPath = fmt::format("C:/Projects/NNMat/data/{}_W400xH400_L151xV151.btf", BTFName);
    auto pBTF = LoadBTF(inputPath.c_str());

    int height = 400;
    int width = 400;
    int imageSize = 3 * height * width;

    for (int binID = startBin; binID < 42; binID++)
    {
        // first round, count how many data inside the bin
        int count = 0;
        for (int viewID = 0; viewID < 151; viewID++)
        {
            for (int lightID = 0; lightID < 151; lightID++)
            {
                float3 l = float3(pBTF->Lights[lightID].x, pBTF->Lights[lightID].y, pBTF->Lights[lightID].z);
                float3 v = float3(pBTF->Views[viewID].x, pBTF->Views[viewID].y, pBTF->Views[viewID].z);
                // Normalized theta [0, 1]
                float3 h = normalize(l + v);
                float theta = acosf(h.z) / M_PI * 2;
                if (theta >= binID * 0.02f && theta < binID * 0.02f + 0.02f)
                {
                    count++;
                }
            }
        }
        if (count == 0)
            continue;
        logInfo("Bin: {} Range: [{}, {}) Count: {}", binID, binID * 0.02f, binID * 0.02f + 0.02f, count);
        // if (binID < 195)
        //     continue;
        TextureDataFloat input(count, width * height, 3);
        // logInfo("{}  {}",input.data.size(), count * width * height * 3);
        TextureDataFloat Tinput;
        TextureDataFloat lut;
        uint* indexList = new uint[count];
        uint globalCount = 0;
        count = 0;
        for (int viewID = 0; viewID < 151; viewID++)
        {
            for (int lightID = 0; lightID < 151; lightID++)
            {
                float3 l = float3(pBTF->Lights[lightID].x, pBTF->Lights[lightID].y, pBTF->Lights[lightID].z);
                float3 v = float3(pBTF->Views[viewID].x, pBTF->Views[viewID].y, pBTF->Views[viewID].z);
                // float theta = l.z;
                float3 h = normalize(l + v);
                float theta = acosf(h.z) / M_PI * 2;
                if (theta >= binID * 0.02f && theta < binID * 0.02f + 0.02f)
                {
                    uint offset = count * imageSize;
                    extractBTF(pBTF, viewID, lightID, input.data, offset);
                    indexList[count] = globalCount;
                    count++;
                }
                globalCount++;
            }
        }
        logInfo("[Synthesis] Precomputing Gaussian T and Inv.");
        Precomputations(input, Tinput, lut);
        logInfo("[Synthesis] Precomputation done!");
        imwrite(fmt::format("C:/Projects/NNMat/data/leather11_histo_test/LUT_{}.exr", binID), height, width, lut.data.data());

        for (size_t i = 0; i < count; i++)
        {
            if (i % 10 == 1)
                logInfo("{} / {}", i, count);
            uint offset = i * imageSize;
            imwrite(
                fmt::format("C:/Projects/NNMat/data/leather11_histo_test/{}.exr", indexList[i]), height, width, Tinput.data.data() + offset
                // fmt::format("C:/Projects/NNMat/data/leather11_histo_wi/{}.exr", indexList[i]), height, width, input.data.data() + offset
            );
        }
        // break;
    }
    return 0;
}

int main(int argc, char** argv)
{
    // int threadNum = std::stoi(argv[1]);
    // logInfo("[Histogram] Running Indx {}", threadNum);
    return histogramTest(0);
    // return histogramBasedOnWi(threadNum);
    // return histogramBasedOnH();
    // return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
