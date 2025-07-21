/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "BTFExtract.h"
#define BTF_IMPLEMENTATION
#include "external/BTF.h"

#include <format>
FALCOR_EXPORT_D3D12_AGILITY_SDK
static void imwrite(const std::filesystem::path& path, uint32_t width, uint32_t height, float* pData)
{
    Bitmap::saveImage(
        path, width, height, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Uncompressed, ResourceFormat::RGB32Float, false, pData
    );
}

int runMain(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <BTFName> <threadNum>" << std::endl;
        return 1;
    }
    std::string BTFName = argv[1];

    int threadNum = std::stoi(argv[2]);
    if (threadNum > 9)
    {
        std::cout << "threadNum cant exceed 9" << std::endl;
        return 1;
    }
    std::string inputPath = fmt::format("C:/Projects/NNMat/data/{}_W400xH400_L151xV151.btf", BTFName);

    auto* btf = LoadBTF(inputPath.c_str());
    if (!btf)
    {
        fprintf(stderr, "Failed to load btf: %s", argv[0]);
        return 1;
    }
    int minView = 0;
    int maxView = 151;
    minView = threadNum * 15;
    if (threadNum < 9)
    {
        maxView = minView + 15;
    }

    auto area = btf->Width * btf->Height;
    uint32_t* tex_data = new uint32_t[area];

    float* data = new float[area * 3];

    int count = 1;
    for (int viewID = minView; viewID < maxView; viewID++)
        for (int lightID = 0; lightID < 151; lightID++)
        {
            count = viewID * 151 + lightID;
            for (uint32_t btf_y = 0; btf_y < btf->Height; ++btf_y)
            {
                for (uint32_t btf_x = 0; btf_x < btf->Width; ++btf_x)
                {
                    auto spec = BTFFetchSpectrum(btf, lightID, viewID, btf_x, btf_y);
                    int idx = ((btf->Height - btf_y - 1) * btf->Width + btf_x) * 3;
                    data[idx + 0] = spec.x;
                    data[idx + 1] = spec.y;
                    data[idx + 2] = spec.z;
                }
            }
            std::string outputPath = fmt::format(
                "C:/Projects/NNMat/data/{}/{}_{}_{}_{}_{}_{}_{}.exr",
                BTFName,
                fmt::format("{:05}", count),
                btf->Lights[lightID].x,
                btf->Lights[lightID].y,
                btf->Lights[lightID].z,
                btf->Views[viewID].x,
                btf->Views[viewID].y,
                btf->Views[viewID].z
            );
            imwrite(outputPath.c_str(), btf->Width, btf->Height, data);
        }

    delete[] tex_data;

    DestroyBTF(btf);

    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
