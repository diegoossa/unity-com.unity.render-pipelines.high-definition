Shader "Hidden/HDRP/TemporalAntialiasing"
{
    HLSLINCLUDE

        #pragma target 4.5
        #pragma multi_compile_local _ ORTHOGRAPHIC
        #pragma multi_compile_local _ REDUCED_HISTORY_CONTRIB
        #pragma only_renderers d3d11 ps4 xboxone vulkan metal switch

        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
        #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Material/Builtin/BuiltinData.hlsl"
        #include "Packages/com.unity.render-pipelines.high-definition/Runtime/ShaderLibrary/ShaderVariables.hlsl"
        #include "Packages/com.unity.render-pipelines.high-definition/Runtime/PostProcessing/Shaders/TemporalAntialiasing.hlsl"

        #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Filtering.hlsl"

        float3 YcocgFromRgb(const in float3 rgb)
        {
            float g = rgb.g * 0.5f;
            float rb = rgb.r + rgb.b;

            return float3(
                rb * 0.25f + g,
                0.5f * (rgb.r - rgb.b),
                rb * -0.25f + g
            );
        }

        float3 RgbFromYcocg(const in float3 ycocg)
        {
            float xz = ycocg.x - ycocg.z;

            return float3(
                xz + ycocg.y,
                ycocg.x + ycocg.z,
                xz - ycocg.y
            );
        }

        float TAAReconstructionFilterWeightGaussianFitBlackmanHarris(const in float x2)
        {
            return exp(-2.29f * x2);
        }

        // Brent Burley: Filtering in PRMan
        // https://web.archive.org/web/20080908072950/http://www.renderman.org/RMR/st/PRMan_Filtering/Filtering_In_PRMan.html
        // https://www.mathworks.com/help/signal/ref/blackmanharris.html?requestedDomain=www.mathworks.com
        float TAAReconstructionFilterWeightBlackmanHarris(const in float x)
        {
            const float WIDTH = 3.3f;
            const float HALF_WIDTH = 0.5f * WIDTH;
            const float SCALE = TWO_PI / WIDTH;
            const float BIAS = PI;

            float x1 = min(HALF_WIDTH, x) * SCALE + BIAS;

            const float C0 =  0.35875f;
            const float C1 = -0.48829f;
            const float C2 =  0.14128f;
            const float C3 = -0.01168f;

            return C0 + C1 * cos(x1) + C2 * cos(2.0 * x1) + C3 * cos(3.0 * x1);
        }

        float TAAReconstructionFilterWeightLanczos(const in float x, const in float widthInverse)
        {
            float c1 = PI * x;
            float c2 = widthInverse * c1;
            return (c2 > PI)
                ? 0.0f
                : (x < 1e-5f)
                    ? 1.0
                    : (sin(c2) * sin(c1) / (c2 * c1));
        }

        float TAAReconstructionFilterWeightDodgsonQuadratic(const in float x, const in float x2, const float r)
        {
            if (x > 1.5) { return 0.0; }

            return  (x > 0.5)
                ? (r * x2 - (2.0 * r + 0.5) * x + (r * 0.75 + 0.75))
                : ((r * 0.5 + 0.5) - 2.0 * r * x2);
        }

        #define BICUBIC_SHARPNESS_C_BSPLINE 0.0
        #define BICUBIC_SHARPNESS_C_MITCHELL 1.0 / 3.0
        #define BICUBIC_SHARPNESS_C_ROUBIDOUX 0.3655305
        #define BICUBIC_SHARPNESS_C_CATMULL_ROM 0.5
        float TAAReconstructionFilterWeightBicubic(const in float x, const in float x2, const float b, const float c)
        {
            float x3 = x2 * x;

            return (x < 2.0)
                ? ((x < 1.0)
                    ? ((2.0 - 1.5 * b - c) * x3 + (-3.0 + 2.0 * b + c) * x2 + (1.0 - b / 3.0))
                    : ((-b / 6.0 - c) * x3 + (b + 5.0 * c) * x2 + (-2.0 * b - 8.0 * c) * x + (4.0 / 3.0 * b + 4.0 * c)))
                : 0.0;
        }

        #define TAA_RECONSTRUCTION_FILTER_UNITY 0
        #define TAA_RECONSTRUCTION_FILTER_GAUSSIAN 1
        #define TAA_RECONSTRUCTION_FILTER_BLACKMAN_HARRIS 2
        #define TAA_RECONSTRUCTION_FILTER_DODGSON_QUADRATIC 3
        #define TAA_RECONSTRUCTION_FILTER_MITCHELL 4
        #define TAA_RECONSTRUCTION_FILTER_ROUBIDOUX 5
        #define TAA_RECONSTRUCTION_FILTER_CATMULL_ROM 6
        #define TAA_RECONSTRUCTION_FILTER_LANCZOS_2 7
        #define TAA_RECONSTRUCTION_FILTER_LANCZOS_3 8
        #define TAA_RECONSTRUCTION_FILTER_LANCZOS_4 9
        #define TAA_RECONSTRUCTION_FILTER_LANCZOS_5 10

        int GetTAAReconstructionFilterRadius(int reconstructionFilter)
        {
            switch (reconstructionFilter)
            {
                case TAA_RECONSTRUCTION_FILTER_GAUSSIAN: return 1;
                case TAA_RECONSTRUCTION_FILTER_BLACKMAN_HARRIS: return 1;
                case TAA_RECONSTRUCTION_FILTER_DODGSON_QUADRATIC: return 1;
                case TAA_RECONSTRUCTION_FILTER_MITCHELL: return 2;
                case TAA_RECONSTRUCTION_FILTER_ROUBIDOUX: return 2;
                case TAA_RECONSTRUCTION_FILTER_CATMULL_ROM: return 2;
                case TAA_RECONSTRUCTION_FILTER_LANCZOS_2: return 2;
                case TAA_RECONSTRUCTION_FILTER_LANCZOS_3: return 3;
                case TAA_RECONSTRUCTION_FILTER_LANCZOS_4: return 4;
                case TAA_RECONSTRUCTION_FILTER_LANCZOS_5: return 5;
                default: return 1;
            }
        }


        float3 EvaluateTAAReconstructionFilter(TEXTURE2D_X(tex), TEXTURE2D_X(texHistory), float2 coords, float2 offset, float2 scale, float2 scaleHistory, float2 motionVector)
        {
            int kernelRadius = GetTAAReconstructionFilterRadius(_TAAReconstructionFilter);

            float3 colorMin = FLT_MAX;
            float3 colorMax = -FLT_MAX;
            float3 colorTotal = 0.0f;
            float weightTotal = 0.0f;
            for (int y = -kernelRadius; y <= kernelRadius; ++y)
            {
                for (int x = -kernelRadius; x <= kernelRadius; ++x)
                {
                    float2 jitter = _TaaJitterStrength.xy;
                    float2 offsetPixels = float2(x, y) + jitter;
                    float distanceSquaredPixels = dot(offsetPixels, offsetPixels);

                    float sampleWeight = 1.0f;
                    switch (_TAAReconstructionFilter)
                    {
                        case TAA_RECONSTRUCTION_FILTER_GAUSSIAN:
                            sampleWeight = TAAReconstructionFilterWeightGaussianFitBlackmanHarris(distanceSquaredPixels);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_BLACKMAN_HARRIS:
                            sampleWeight = TAAReconstructionFilterWeightBlackmanHarris(distanceSquaredPixels);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_DODGSON_QUADRATIC:
                            sampleWeight = TAAReconstructionFilterWeightDodgsonQuadratic(sqrt(distanceSquaredPixels), distanceSquaredPixels, _TaaFrameInfo.x * 0.5f);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_MITCHELL:
                            sampleWeight = TAAReconstructionFilterWeightBicubic(sqrt(distanceSquaredPixels), distanceSquaredPixels, BICUBIC_SHARPNESS_C_MITCHELL * -2.0f + 1.0f, BICUBIC_SHARPNESS_C_MITCHELL);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_ROUBIDOUX:
                            sampleWeight = TAAReconstructionFilterWeightBicubic(sqrt(distanceSquaredPixels), distanceSquaredPixels, BICUBIC_SHARPNESS_C_ROUBIDOUX * -2.0f + 1.0f, BICUBIC_SHARPNESS_C_ROUBIDOUX);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_CATMULL_ROM:
                            sampleWeight = TAAReconstructionFilterWeightBicubic(sqrt(distanceSquaredPixels), distanceSquaredPixels, BICUBIC_SHARPNESS_C_CATMULL_ROM * -2.0f + 1.0f, BICUBIC_SHARPNESS_C_CATMULL_ROM);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_LANCZOS_2:
                            sampleWeight = TAAReconstructionFilterWeightLanczos(sqrt(distanceSquaredPixels), 1.0f / 2.0f);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_LANCZOS_3:
                            sampleWeight = TAAReconstructionFilterWeightLanczos(sqrt(distanceSquaredPixels), 1.0f / 3.0f);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_LANCZOS_4:
                            sampleWeight = TAAReconstructionFilterWeightLanczos(sqrt(distanceSquaredPixels), 1.0f / 4.0f);
                            break;
                        case TAA_RECONSTRUCTION_FILTER_LANCZOS_5:
                            sampleWeight = TAAReconstructionFilterWeightLanczos(sqrt(distanceSquaredPixels), 1.0f / 5.0f);
                            break;
                        default:
                            sampleWeight = 1.0f;
                            break;
                    }

                    float3 sampleColor = Fetch(tex, coords, float2(x, y), scale);

                    sampleColor.xyz = YcocgFromRgb(sampleColor.rgb);

                    sampleColor.xyz *= rcp(1.0f + sampleColor.x);

                    // if (max(abs(x), abs(y)) <= 1)
                    if (distanceSquaredPixels < (1.5f * 1.5f))
                    {
                        colorMin = min(colorMin, sampleColor.xyz);
                        colorMax = max(colorMax, sampleColor.xyz);
                    }

                    colorTotal += sampleColor * sampleWeight;
                    weightTotal += sampleWeight;
                }
            }

            float3 color = (weightTotal == 0.0f) ? 0.0f : (colorTotal * rcp(weightTotal));

            // float2 coordsHistory = (coords - motionVector) * scaleHistory(coords + offset * _ScreenSize.zw) * scale;
            float2 coordsHistory = (coords - motionVector) * _ScreenSize.xy;
            float2 weights[3], uvs[3];
            BicubicFilterFromSharpness(coordsHistory, weights, uvs, _TaaFrameInfo.x * 0.25f);

            // Rather than taking the full 9 hardware-filtered taps to resolve our bicubic filter, we drop the (lowest weight) corner samples, computing our bicubic filter with only 5-taps.
            // Visually, error is low enough to be visually indistinguishable in our test cases.
            // Source:
            // Filmic SMAA: Sharp Morphological and Temporal Antialiasing
            // http://advances.realtimerendering.com/s2016/index.html
            float3 history =
                    (weights[1].x * weights[0].y) * SAMPLE_TEXTURE2D_X_LOD(texHistory, s_linear_clamp_sampler, float2(uvs[1].x, uvs[0].y) * scaleHistory * _ScreenSize.zw, 0).rgb
                    + (weights[0].x * weights[1].y) * SAMPLE_TEXTURE2D_X_LOD(texHistory, s_linear_clamp_sampler, float2(uvs[0].x, uvs[1].y) * scaleHistory * _ScreenSize.zw, 0).rgb
                    + (weights[1].x * weights[1].y) * SAMPLE_TEXTURE2D_X_LOD(texHistory, s_linear_clamp_sampler, float2(uvs[1].x, uvs[1].y) * scaleHistory * _ScreenSize.zw, 0).rgb
                    + (weights[2].x * weights[1].y) * SAMPLE_TEXTURE2D_X_LOD(texHistory, s_linear_clamp_sampler, float2(uvs[2].x, uvs[1].y) * scaleHistory * _ScreenSize.zw, 0).rgb
                    + (weights[1].x * weights[2].y) * SAMPLE_TEXTURE2D_X_LOD(texHistory, s_linear_clamp_sampler, float2(uvs[1].x, uvs[2].y) * scaleHistory * _ScreenSize.zw, 0).rgb;

            // history = Fetch(texHistory, coords - motionVector, 0.0, scaleHistory);

            history.xyz = YcocgFromRgb(history.rgb);
            history *= rcp(1.0f + history.x);
            history = clamp(history, colorMin, colorMax);

            color.xyz = lerp(color.xyz, history.xyz, _TAAHistoryFeedback);

            color.xyz *= rcp(1.0f - color.x);
            color.rgb = RgbFromYcocg(color.xyz);
            color = clamp(color, 0.0, CLAMP_MAX);
            return color;
        }

        TEXTURE2D_X(_InputTexture);
        TEXTURE2D_X(_InputHistoryTexture);
        RW_TEXTURE2D_X(float3, _OutputHistoryTexture);

        struct Attributes
        {
            uint vertexID : SV_VertexID;
            UNITY_VERTEX_INPUT_INSTANCE_ID
        };

        struct Varyings
        {
            float4 positionCS : SV_POSITION;
            float2 texcoord   : TEXCOORD0;
            UNITY_VERTEX_OUTPUT_STEREO
        };

        Varyings Vert(Attributes input)
        {
            Varyings output;
            UNITY_SETUP_INSTANCE_ID(input);
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);
            output.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
            output.texcoord = GetFullScreenTriangleTexCoord(input.vertexID);
            return output;
        }
        
        float3 MainReconstructionFilter(Varyings input)
        {
            float sharpenStrength = _TaaFrameInfo.x;
            float2 jitter = _TaaJitterStrength.zw;

    #if defined(ORTHOGRAPHIC)
            // Don't dilate in ortho
            float2 closest = input.positionCS.xy;
    #else
            float2 closest = GetClosestFragment(input.positionCS.xy);
    #endif

            float2 motionVector;
            DecodeMotionVector(LOAD_TEXTURE2D_X(_CameraMotionVectorsTexture, closest), motionVector);
            float motionVecLength = length(motionVector);

            return EvaluateTAAReconstructionFilter(_InputTexture, _InputHistoryTexture, input.texcoord, 0.0, _RTHandleScale.xy, _RTHandleScaleHistory.xy, motionVector);
        }

        void FragTAA(Varyings input, out float3 outColor : SV_Target0)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            if (_TAAReconstructionFilter != TAA_RECONSTRUCTION_FILTER_UNITY)
            {
                outColor = MainReconstructionFilter(input);
                return;
            }

            float sharpenStrength = _TaaFrameInfo.x;
            float2 jitter = _TaaJitterStrength.zw;

    #if defined(ORTHOGRAPHIC)
            // Don't dilate in ortho
            float2 closest = input.positionCS.xy;
    #else
            float2 closest = GetClosestFragment(input.positionCS.xy);
    #endif

            float2 motionVector;
            DecodeMotionVector(LOAD_TEXTURE2D_X(_CameraMotionVectorsTexture, closest), motionVector);
            float motionVecLength = length(motionVector);

            float2 uv = input.texcoord - jitter;

            float3 color = Fetch(_InputTexture, uv, 0.0, _RTHandleScale.xy);
            float3 history = Fetch(_InputHistoryTexture, input.texcoord - motionVector, 0.0, _RTHandleScaleHistory.zw);

            float3 topLeft = Fetch(_InputTexture, uv, -RADIUS, _RTHandleScale.xy);
            float3 bottomRight = Fetch(_InputTexture, uv, RADIUS, _RTHandleScale.xy);

            float3 corners = 4.0 * (topLeft + bottomRight) - 2.0 * color;

            // Sharpen output
    #if SHARPEN
            float3 topRight = Fetch(_InputTexture, uv, float2(RADIUS, -RADIUS), _RTHandleScale.xy);
            float3 bottomLeft = Fetch(_InputTexture, uv, float2(-RADIUS, RADIUS), _RTHandleScale.xy);
            float3 blur = (topLeft + topRight + bottomLeft + bottomRight) * 0.25;
            color += (color - blur) * sharpenStrength;
    #endif

            color = clamp(color, 0.0, CLAMP_MAX);

            float3 average = Map((corners + color) / 7.0);

            topLeft = Map(topLeft);
            bottomRight = Map(bottomRight);
            color = Map(color);

            float colorLuma = Luminance(color);
            float averageLuma = Luminance(average);
            float nudge = lerp(4.0, 0.25, saturate(motionVecLength * 100.0)) * abs(averageLuma - colorLuma);

            float3 minimum = min(bottomRight, topLeft) - nudge;
            float3 maximum = max(topLeft, bottomRight) + nudge;

            history = Map(history);

            // Clip history samples
    #if CLIP_AABB
            history = ClipToAABB(history, minimum, maximum);
    #else
            history = clamp(history, minimum, maximum);
    #endif

            // Blend color & history
            // Feedback weight from unbiased luminance diff (Timothy Lottes)
            float historyLuma = Luminance(history);
            float diff = abs(colorLuma - historyLuma) / Max3(colorLuma, historyLuma, 0.2);
            float weight = 1.0 - diff;
            float feedback = lerp(FEEDBACK_MIN, FEEDBACK_MAX, weight * weight);

            color = Unmap(lerp(color, history, feedback));
            color = clamp(color, 0.0, CLAMP_MAX);

            _OutputHistoryTexture[COORD_TEXTURE2D_X(input.positionCS.xy)] = color;
            outColor = color; 
        }

        void FragExcludedTAA(Varyings input, out float3 outColor : SV_Target0)
        {
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

            float2 jitter = _TaaJitterStrength.zw;
            float2 uv = input.texcoord - jitter;

            float3 color = Fetch(_InputTexture, uv, 0.0, _RTHandleScale.xy);

            outColor = color;
        }
    ENDHLSL

    SubShader
    {
        Tags{ "RenderPipeline" = "HDRenderPipeline" }

        // TAA
        Pass
        {
            Stencil
            {
                ReadMask 16     // ExcludeFromTAA
                Ref 16          // ExcludeFromTAA
                Comp NotEqual
                Pass Keep
            }

            ZWrite Off ZTest Always Blend Off Cull Off

            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragTAA
            ENDHLSL
        }

        // Excluded from TAA
        // Note: This is a straightup passthrough now, but it would be interesting instead to try to reduce history influence instead.
        Pass
        {
            Stencil
            {
                ReadMask 16     // ExcludeFromTAA
                Ref 16          // ExcludeFromTAA
                Comp Equal
                Pass Keep
            }

            ZWrite Off ZTest Always Blend Off Cull Off

            HLSLPROGRAM
                #pragma vertex Vert
                #pragma fragment FragExcludedTAA
            ENDHLSL
        }
    }
    Fallback Off
}
