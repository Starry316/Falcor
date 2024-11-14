#include "Synthesis.h"
#include "Core/Error.h"
#include "Core/API/Device.h"
namespace Falcor
{

TextureSynthesis::TextureSynthesis(ref<Device> pDevice)
{
    mpColor = Texture::createFromFile(pDevice, "D:/Picture1.png", true, true, ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);


    // // Use >= since we reserve 0xFFFFFFFFu as an invalid flag marker during construction.
    // if (weights.size() >= std::numeric_limits<uint32_t>::max())
    //     FALCOR_THROW("Too many entries for alias table.");

    // std::uniform_int_distribution<uint32_t> rngDist;

    // mpWeights =
    //     pDevice->createStructuredBuffer(sizeof(float), mCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, weights.data());

    // // Our working set / intermediate buffers (underweight & overweight); initialize to "invalid"
    // std::vector<uint32_t> lowIdx(mCount, 0xFFFFFFFFu);
    // std::vector<uint32_t> highIdx(mCount, 0xFFFFFFFFu);

    // // Sum element weights, use double to minimize precision issues
    // mWeightSum = 0.0;
    // for (float f : weights)
    //     mWeightSum += f;

    // // Find the average weight
    // float avgWeight = float(mWeightSum / double(mCount));

    // // Initialize working set. Inset inputs into our lists of above-average or below-average weight elements.
    // int lowCount = 0;
    // int highCount = 0;
    // for (uint32_t i = 0; i < mCount; ++i)
    // {
    //     if (weights[i] < avgWeight)
    //         lowIdx[lowCount++] = i;
    //     else
    //         highIdx[highCount++] = i;
    // }

    // // Create alias table entries by merging above- and below-average samples
    // std::vector<AliasTable::Item> items(mCount);
    // for (uint32_t i = 0; i < mCount; ++i)
    // {
    //     // Usual case:  We have an above-average and below-average sample we can combine into one alias table entry
    //     if ((lowIdx[i] != 0xFFFFFFFFu) && (highIdx[i] != 0xFFFFFFFFu))
    //     {
    //         // Create an alias table tuple:
    //         items[i] = {weights[lowIdx[i]] / avgWeight, highIdx[i], lowIdx[i], 0};

    //         // We've removed some weight from element highIdx[i]; update it's weight, then re-enter it
    //         // on the end of either the above-average or below-average lists.
    //         float updatedWeight = (weights[lowIdx[i]] + weights[highIdx[i]]) - avgWeight;
    //         weights[highIdx[i]] = updatedWeight;
    //         if (updatedWeight < avgWeight)
    //             lowIdx[lowCount++] = highIdx[i];
    //         else
    //             highIdx[highCount++] = highIdx[i];
    //     }

    //     // The next two cases can only occur towards the end of table creation, because either:
    //     //    (a) all the remaining possible alias table entries have weight *exactly* equal to avgWeight,
    //     //        which means these alias table entries only have one input item that is selected
    //     //        with 100% probability
    //     //    (b) all the remaining alias table entires have *almost* avgWeight, but due to (compounding)
    //     //        precision issues throughout the process, they don't have *quite* that value.  In this case
    //     //        treating these entries as having exactly avgWeight (as in case (a)) is the only right
    //     //        thing to do mathematically (other than re-generating the alias table using higher precision
    //     //        or trying to reduce catasrophic numerical cancellation in the "updatedWeight" computation above).
    //     else if (highIdx[i] != 0xFFFFFFFFu)
    //     {
    //         items[i] = {1.0f, highIdx[i], highIdx[i], 0};
    //     }
    //     else if (lowIdx[i] != 0xFFFFFFFFu)
    //     {
    //         items[i] = {1.0f, lowIdx[i], lowIdx[i], 0};
    //     }

    //     // If there is neither a highIdx[i] or lowIdx[i] for some array element(s).  By construction,
    //     // this cannot occur (without some logic bug above).
    //     else
    //     {
    //         FALCOR_ASSERT(false); // Should not occur
    //     }
    // }

    // // TODO: We can simplify the alias table to implicitly store indexB (aka lowIdx[i]), so the AliasTable::Item
    // // structure would be 1 float + 1 uint32_t, rather than 128 bits.  This, of course, would change usage in shaders
    // // and elsewhere.  To do this, here you'd need to sort elements by indexB so that when looking up mpItems[j],
    // // indexB==j.  This works since, by construction, only one element in the table has indexB==j (for any j
    // // in [0...mCount-1]).  Alternatively, during the loop above, you could directly enter elements into the
    // // correct location in the alias table.

    // // Stash the alias table in our GPU buffer
    // mpItems = pDevice->createStructuredBuffer(
    //     sizeof(AliasTable::Item), mCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, items.data()
    // );
}
void TextureSynthesis::bindShaderData(const ShaderVar& var) const
{
    var["color"] = mpColor;
    // var["weights"] = mpWeights;
    // var["count"] = mCount;
    // var["weightSum"] = (float)mWeightSum;
}

} // namespace Falcor