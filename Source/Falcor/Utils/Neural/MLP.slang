import Utils.Math.MathHelpers;
import Utils.Neural.ActivationFunc;
import Utils.Math.FormatConversion;
#ifndef MAX_NN_DIM
#define MAX_NN_DIM 8
#endif
struct MLP
{
    int layerNum;
    StructuredBuffer<float4x4> weights;
    StructuredBuffer<float4> bias;
    StructuredBuffer<float> meta;
    float4 eval(inout float4 x[MAX_NN_DIM])
    {
        float4 y[MAX_NN_DIM];
        int offset = 0;
        int biasOffset = 0;
        for (int i = 0; i < layerNum; ++i)
        {
            if (i % 2 == 0)
            {
                float h = meta[2 * i + 1];
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        y[k] += mul(weights[outNum * j + k + offset], x[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] += bias[j + biasOffset];
                    y[j] = leakyrelu(y[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
            else
            {
                float h = meta[2 * i + 1];
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        x[k] += mul(weights[outNum * j + k + offset], y[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] += bias[j + biasOffset];
                    x[j] = leakyrelu(x[j]);
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
        }
        // final layer
        if (layerNum % 2 == 1)
        {
            return relu(y[0]);
        }
        else
        {
            return relu(x[0]);
        }
    }

    float4 evalfp16(inout float4 x[MAX_NN_DIM])
    {
        float4 y[MAX_NN_DIM];
        int offset = 0;
        int biasOffset = 0;
        for (int i = 0; i < layerNum; ++i)
        {
            if (i % 2 == 0)
            {
                float h = meta[2 * i + 1];
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        y[k] += mul(weights[outNum * j + k + offset], x[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    y[j] += bias[j + biasOffset];
                    y[j] = leakyrelu(y[j]);
                    y[j].xy = unpackSnorm2x16(packSnorm2x16(y[j].xy));
                    y[j].zw = unpackSnorm2x16(packSnorm2x16(y[j].zw));
                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
            else
            {
                float h = meta[2 * i + 1];
                float w = meta[2 * i + 2];

                int outNum = floor(w / 4);
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] = float4(0.0f);
                }
                for (int j = 0; j < h / 4; ++j)
                {
                    for (int k = 0; k < w / 4; ++k)
                    {
                        x[k] += mul(weights[outNum * j + k + offset], y[j]);
                    }
                }
                for (int j = 0; j < outNum; ++j)
                {
                    x[j] += bias[j + biasOffset];
                    x[j] = leakyrelu(x[j]);
                    x[j].xy = unpackSnorm2x16(packSnorm2x16(x[j].xy));
                    x[j].zw = unpackSnorm2x16(packSnorm2x16(x[j].zw));

                }
                biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
        }
        // final layer
        if (layerNum % 2 == 1)
        {
            return relu(y[0]);
        }
        else
        {
            return relu(x[0]);
        }
    }
}
