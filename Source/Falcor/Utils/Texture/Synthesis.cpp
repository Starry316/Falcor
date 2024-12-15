#include "Synthesis.h"
#include "SynthesisUtils.h"
#include "Core/Error.h"
#include "Core/API/Device.h"
namespace Falcor
{
TextureSynthesis::TextureSynthesis()
{
    // std::string hfPath = "D:/textures/ubo/leather11.png";
}
void TextureSynthesis::bindShaderData(const ShaderVar& var) const
{
    var["color"] = mpColor;
}
void TextureSynthesis::readHFData(std::string hfPath, ref<Device> pDevice)
{
    Bitmap::UniqueConstPtr pBitmap = Bitmap::createFromFile(hfPath, true);
    FALCOR_ASSERT(pBitmap);
    logInfo("[Synthesis] Input Image Path: {}", hfPath);
    logInfo("[Synthesis] Input Image Format: {}", to_string(pBitmap->getFormat()));
    logInfo("[Synthesis] Input Image Width:  {}", pBitmap->getWidth());
    logInfo("[Synthesis] Input Image Height: {}", pBitmap->getHeight());

    TextureDataFloat input(pBitmap->getHeight(), pBitmap->getWidth(), 1);
    int bitMapChannels = 1;
    if (pBitmap->getFormat() == ResourceFormat::BGRX8Unorm)
    {
        bitMapChannels = 4;
        for (size_t i = 0; i < pBitmap->getWidth() * pBitmap->getHeight(); i++)
        {
            input.data[i] = pBitmap->getData()[i * bitMapChannels + 0] / 255.0f;
        }
    }

    else if (pBitmap->getFormat() == ResourceFormat::R16Unorm)
    {
        bitMapChannels = 1;
        auto pBitData = reinterpret_cast<const uint16_t*>(pBitmap->getData());
        for (size_t i = 0; i < pBitmap->getWidth() * pBitmap->getHeight(); i++)
        {
            input.data[i] = pBitData[i * bitMapChannels + 0] / 65535.0f;
        }
    }

    TextureDataFloat Tinput;
    TextureDataFloat lut;
    logInfo("[Synthesis] Precomputing Gaussian T and Inv.");
    Precomputations(input, Tinput, lut);
    logInfo("[Synthesis] Precomputation done!");
    // TODO generate max mipmap
    mpHFT = pDevice->createTexture2D(
        Tinput.width,
        Tinput.height,
        ResourceFormat::R32Float,
        1,
        Resource::kMaxPossible,
        Tinput.data.data(),
        ResourceBindFlags::ShaderResource
    );

    mpHFInvT =
        pDevice->createTexture2D(lut.width, lut.height, ResourceFormat::R32Float, 1, 1, lut.data.data(), ResourceBindFlags::ShaderResource);
}

void TextureSynthesis::precomputeFeatureData(std::vector<float> data, uint2 dataDim, ref<Device> pDevice)
{
    logInfo("[Synthesis] Input Feature Dim: {}", dataDim);
    logInfo("[Synthesis] Input Feature Size:  {}", data.size());
    std::vector<float> TData(data.size());
    std::vector<float> invTData(LUT_WIDTH * 4 * dataDim.y);
    TextureDataFloat acf = TextureDataFloat(dataDim.x, dataDim.x, 1);
    logInfo("[Synthesis] Precomputing Feature Gaussian T and Inv.");
    for(uint i = 0; i < dataDim.y; i++)
    {
        uint offset = i * dataDim.x * dataDim.x * 4;
        uint singleDataSize = dataDim.x * dataDim.x * 4;
        TextureDataFloat input(dataDim.x, dataDim.x, 4);
        std::copy(data.begin()+ offset, data.begin()+ offset + singleDataSize, input.data.begin());

        TextureDataFloat Tinput;
        TextureDataFloat lut;
        Precomputations(input, Tinput, lut);
        std::copy(Tinput.data.begin(), Tinput.data.end(), TData.begin()+offset);
        std::copy(lut.data.begin(), lut.data.end(), invTData.begin() + LUT_WIDTH * i * 4);

        // if(i == 0)
        //     calculateAutocovariance(input, acf);
    }

    logInfo("[Synthesis] Precomputation done!");

    // // TODO generate max mipmap
    mpFeatureT = pDevice->createTexture2D(
        dataDim.x,  dataDim.x, ResourceFormat::RGBA32Float, dataDim.y, 1, TData.data(), ResourceBindFlags::ShaderResource
    );
    mpFeatureInvT =
        pDevice->createTexture2D(LUT_WIDTH, 1, ResourceFormat::RGBA32Float, dataDim.y, 1, invTData.data(), ResourceBindFlags::ShaderResource);
    mpACF =
        pDevice->createTexture2D(dataDim.x, dataDim.x, ResourceFormat::R32Float, 1, 1, acf.data.data(), ResourceBindFlags::ShaderResource);

}

void TextureSynthesis::bindHFData(const ShaderVar& var)
{
    var["tex"] = mpHFT;
    var["invTex"] = mpHFInvT;
}
void TextureSynthesis::bindFeatureData(const ShaderVar& var)
{
    var["tex"] = mpFeatureT;
    var["invTex"] = mpFeatureInvT;
    var["acf"] = mpACF;
}
} // namespace Falcor
