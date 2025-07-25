import Utils.Math.MathHelpers;
import Utils.Neural.ActivationFunc;
import Utils.Math.FormatConversion;
#ifndef MAX_NN_DIM
#define MAX_NN_DIM 4
#endif
struct MLP
{
    int layerNum;
    StructuredBuffer<float4x4> weights;
    StructuredBuffer<float4> bias;
    StructuredBuffer<float> meta;

    Buffer<float> debugWeights;
    Buffer<float> debugBias;

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
                    y[j] = relu(y[j]);
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
                    x[j] = relu(x[j]);
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

    float4 eval(inout float inputVal[24])
    {
        float val1[32];
        float val2[32];
        int offset = 0;
        int biasOffset = 0;

        int inNumFirst = 24;
        int inNum = 32;
        int outNum = 32;

        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNumFirst; ++j)
            {
                sum += debugWeights[inNumFirst * k + j + offset] * inputVal[j];
            }
            // val2[k] = sum + debugBias[k + biasOffset];
            val2[k] = relu(sum);
        }
        offset += outNum * inNumFirst;
        biasOffset += outNum;

        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val2[j];
            }
            // val1[k] = sum + debugBias[k + biasOffset];
            val1[k] = relu(sum);
        }
        offset += outNum * inNum;
        biasOffset += outNum;

        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val1[j];
            }
            // val2[k] = sum + debugBias[k + biasOffset];
            val2[k] = relu(sum);
        }
        offset += outNum * inNum;
        biasOffset += outNum;

        for (int k = 0; k < 3; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val2[j];
            }
            // val1[k] = sum + debugBias[k + biasOffset];
            val1[k] = relu(sum);
        }
        return float4(val1[0], val1[1], val1[2], 1.0);
    }

    float4 evalBTF(inout float4 x[MAX_NN_DIM], float4 uFine[2])
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
                    // y[j] += bias[j + biasOffset];
                    y[j] = leakyrelu(y[j]);
                }
                // biasOffset += outNum;
                offset += (int)(h * w) / 16;
            }
            else
            {
                if (i == 3)
                {
                    y[2] = uFine[0];
                    y[3] = uFine[1];
                }
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
                    // x[j] += bias[j + biasOffset];
                    x[j] = leakyrelu(x[j]);
                }
                // biasOffset += outNum;
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

    float4 evalBTF(inout float inputVal[24], float4 uFine[2], int inNumFirst)
    {
        float val1[16];
        float val2[16];
        int offset = 0;

   
        int inSecondNum = 16;
        int inNum = 8;

        int outNum = 8;
        // m1
        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNumFirst; ++j)
            {
                sum += debugWeights[inNumFirst * k + j + offset] * inputVal[j];
            }
            val2[k] = leakyrelu(sum);
        }
        offset += outNum * inNumFirst;

        // m2
        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val2[j];
            }
            val1[k] = leakyrelu(sum);
        }
        offset += outNum * inNum;

        // m3
        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val1[j];
            }
            val2[k] = leakyrelu(sum);
        }
        offset += outNum * inNum;

        val2[8] = uFine[0].x;
        val2[9] = uFine[0].y;
        val2[10] = uFine[0].z;
        val2[11] = uFine[0].w;

        val2[12] = uFine[1].x;
        val2[13] = uFine[1].y;
        val2[14] = uFine[1].z;
        val2[15] = uFine[1].w;

        for (int k = 0; k < outNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inSecondNum; ++j)
            {
                sum += debugWeights[inSecondNum * k + j + offset] * val2[j];
            }
            val1[k] = leakyrelu(sum);
        }
        offset += outNum * inSecondNum;

        for (int k = 0; k < 3; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += debugWeights[inNum * k + j + offset] * val1[j];
            }
            val2[k] = relu(sum);
        }
        return float4(val2[0], val2[1], val2[2], 1.0);
    }
}
