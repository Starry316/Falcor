import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;


// pack inputs into one uint4 using float16
inline uint4 packInput(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy)
{
    uint4 ret;
    float2 h, d, ddxy;
    dirToRusinkiewicz(wi, wo, h, d);
    float3 vh = spherical_to_cartesian_rad(h);
    float3 vd = spherical_to_cartesian_rad(d);

    h = hemisphereDirToDiskSquare(vh);
    d = hemisphereDirToDiskSquare(vd);


    ddxy.x = abs(ddx.x) > abs(ddx.y) ? ddx.x : ddx.y;
    ddxy.y = abs(ddy.x) > abs(ddy.y) ? ddy.x : ddy.y;
    ret.x = packUnorm2x16(h);
    ret.y = packUnorm2x16(d);
    ret.z = packUnorm2x16(uv);
    ret.w = packSnorm2x16(ddxy);
    return ret;
}

inline uint4 packInputSyn(float3 wi, float3 wo, float2 uv1, float2 uv2)
{
    uint4 ret;
    float2 h, d;
    dirToRusinkiewicz(wi, wo, h, d);
    float3 vh = spherical_to_cartesian_rad(h);
    float3 vd = spherical_to_cartesian_rad(d);

    h = hemisphereDirToDiskSquare(vh);
    d = hemisphereDirToDiskSquare(vd);


    ret.x = packUnorm2x16(h);
    ret.y = packUnorm2x16(d);
    ret.z = packUnorm2x16(uv1);
    ret.w = packSnorm2x16(uv2);
    return ret;
}

// unpack input from the packed uint4
inline void unpackInput(uint4 packed, out float2 h, out float2 d, out float2 uv, out float2 ddx, out float2 ddy)
{
    float2 ddxy;
    h =     unpackUnorm2x16(packed.x);
    d =     unpackUnorm2x16(packed.y);
    uv =    unpackUnorm2x16(packed.z);
    ddxy =  unpackSnorm2x16(packed.w);
    ddx = ddxy.xx;
    ddy = ddxy.yy;
}
