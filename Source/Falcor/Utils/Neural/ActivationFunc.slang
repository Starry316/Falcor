
float4 relu(inout float4 x) { return max(x, 0.0f); }
float relu(inout float x) { return max(x, 0.0f); }

float4 leakyrelu(inout float4 x) { return max(x, 0.0f) + min(x, 0.0f) * 0.01f; }
float leakyrelu(inout float x) { return max(x, 0.0f) + min(x, 0.0f) * 0.01f; }

float tanh(inout float x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
float softClip(float x, float a = 100) {
    return x - (1.0 / a) * log(1 + exp(a * (x - 1.0))) + (1.0 / a) * log(1 + exp(a * (-1.0 - x)));
}
inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

