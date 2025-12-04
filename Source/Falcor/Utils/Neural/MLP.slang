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

    Buffer<float> tWeights;
    Buffer<float> tBias;

    void evalT(inout float input[8], inout float output[10])
    {
        float val1[16];


        int offset = 0;
        int biasOffset = 0;

        // t1
        for (int k = 0; k < 16; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 5; ++j)
            {
                sum += tWeights[5 * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += 16 * 5;
        biasOffset += 16;
        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 16; ++j)
            {
                sum += tWeights[16 * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            output[k + 0] = tanh(sum);
        }


        offset += 2 * 16;
        biasOffset += 2;
        // t2
        for (int k = 0; k < 16; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 5; ++j)
            {
                sum += tWeights[5 * k + j + offset] * input[j];
            }
            val1[k] = tBias[k + biasOffset] + sum;
            val1[k] = leakyrelu(val1[k]);
        }
        offset += 16 * 5;
        biasOffset += 16;
        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 16; ++j)
            {
                sum += tWeights[16 * k + j + offset] * val1[j];
            }
            output[k + 2] = tBias[k + biasOffset] + sum;
            output[k + 2] = tanh(output[k + 2]);
        }
        offset += 2 * 16;
        biasOffset += 2;

        // t3
        for (int k = 0; k < 16; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 5; ++j)
            {
                sum += tWeights[5 * k + j + offset] * input[j];
            }
            val1[k] = tBias[k + biasOffset] + sum;
            val1[k] = leakyrelu(val1[k]);
        }
        offset += 16 * 5;
        biasOffset += 16;
        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 16; ++j)
            {
                sum += tWeights[16 * k + j + offset] * val1[j];
            }
            output[k + 4] = tBias[k + biasOffset] + sum;
            output[k + 4] = tanh(output[k + 4]);
        }
        offset += 2 * 16;
        biasOffset += 2;

        // t4
        for (int k = 0; k < 16; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 5; ++j)
            {
                sum += tWeights[5 * k + j + offset] * input[j];
            }
            val1[k] = tBias[k + biasOffset] + sum;
            val1[k] = leakyrelu(val1[k]);
        }
        offset += 16 * 5;
        biasOffset += 16;
        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 16; ++j)
            {
                sum += tWeights[16 * k + j + offset] * val1[j];
            }
            output[k + 6] = tBias[k + biasOffset] + sum;
            output[k + 6] = tanh(output[k + 6]);
        }
        offset += 2 * 16;
        biasOffset += 2;

        // t5
        for (int k = 0; k < 16; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 5; ++j)
            {
                sum += tWeights[5 * k + j + offset] * input[j];
            }
            val1[k] = tBias[k + biasOffset] + sum;
            val1[k] = leakyrelu(val1[k]);
        }
        offset += 16 * 5;
        biasOffset += 16;
        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < 16; ++j)
            {
                sum += tWeights[16 * k + j + offset] * val1[j];
            }
            output[k + 8] = tBias[k + biasOffset] + sum;
            output[k + 8] = tanh(output[k + 8]);
        }
    }
    float2 evalTTriMLP(inout float input[6])
    {
        float val1[16];
        float val2[16];
        float2 t;
        int offset = 0;
        int biasOffset = 0;
        int hiddenNum = 16;
        int inNum = 6;
        // t1
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * inNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 1; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            t.x = leakyrelu(sum);
        }
        offset += hiddenNum * 3;
        biasOffset += 3;

        // t2
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * inNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 1; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            t.y = leakyrelu(sum);
        }
        return t;
    }
    void eval8T(inout float input[8], inout float output[10])
    {
        float val1[16];
        float val2[16];

        int offset = 0;
        int biasOffset = 0;
        int hiddenNum = 16;
        // int inNum = 5;
        int inNum = 6;
        // t1
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * inNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            output[k + 0] = tanh(sum);
        }
        offset += hiddenNum * 2;
        biasOffset += 2;

        // t2
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * inNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            output[k + 2] = tanh(sum);
        }
        offset += hiddenNum * 2;
        biasOffset += 2;

        // t3
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * inNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            output[k + 4] = tanh(sum);
        }
        offset += hiddenNum * 2;
        biasOffset += 2;

        // t4
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * inNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            output[k + 6] = tanh(sum);
        }
        offset += hiddenNum * 2;
        biasOffset += 2;

        // tinNum
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * input[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * inNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 2; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            output[k + 8] = tanh(sum);
        }
        offset += hiddenNum * 2;
        biasOffset += 2;
    }

    float3 evalBTFMLP(inout float val2[66])
    {
        float val1[64];

        int offset = 0;
        int biasOffset = 0;
        // int inNum = 43;
        int inNum = 63;
        // int inNum = 66;
        // int inNum = 26;
        int hiddenNum = 64;
        // int hiddenNum = 32;
        // l1
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += inNum * hiddenNum;
        biasOffset += hiddenNum;
        // l2
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        // l3
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        // l4
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        // l5
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 3; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = relu(sum);
        }

        return float3(val2[0], val2[1], val2[2]);
    }
    float3 evalBTFMLP(inout float val2[64], int inNum)
    {
        float val1[64];

        int offset = 0;
        int biasOffset = 0;
        // int inNum = 43;

        // int inNum = 66;
        // int inNum = 26;
        int hiddenNum = 64;
        // int hiddenNum = 32;
        // l1
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < inNum; ++j)
            {
                sum += tWeights[inNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += inNum * hiddenNum;
        biasOffset += hiddenNum;
        // l2
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        // l3
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        // l4
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        // l5
        for (int k = 0; k < hiddenNum; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val2[j];
            }
            sum += tBias[k + biasOffset];
            val1[k] = leakyrelu(sum);
        }
        offset += hiddenNum * hiddenNum;
        biasOffset += hiddenNum;

        for (int k = 0; k < 3; ++k)
        {
            float sum = 0;
            for (int j = 0; j < hiddenNum; ++j)
            {
                sum += tWeights[hiddenNum * k + j + offset] * val1[j];
            }
            sum += tBias[k + biasOffset];
            val2[k] = relu(sum);
        }

        return float3(val2[0], val2[1], val2[2]);
    }
    float4 eval(inout float4 x[MAX_NN_DIM])
    {
        return float4(0);
    }

    float4 eval(inout float inputVal[24])
    {
       return float4(0);
    }

    float4 evalBTF(inout float4 x[MAX_NN_DIM], float4 uFine[2])
    {
        return float4(0);
    }

    float4 evalBTF(inout float inputVal[24], float4 uFine[2], int inNumFirst)
    {
        return float4(0);
    }
}
