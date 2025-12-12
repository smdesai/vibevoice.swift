import Foundation
import MLX
import MLXNN

public class TokenizerEncoder: Module {
    public let dimension: Int
    public let channels: Int
    public let nFilters: Int
    public let ratios: [Int]
    public let depths: [Int]
    public let causal: Bool

    @ModuleInfo(key: "downsample_layers") public var downsampleLayers: [SConv1d]
    @ModuleInfo(key: "stages") public var stages: [[Block1D]]
    public let normFinal: Module
    @ModuleInfo(key: "head") public var head: SConv1d

    public init(_ config: AcousticTokenizerConfiguration) {
        self.dimension = config.vaeDim
        self.channels = config.channels
        self.nFilters = config.encoderNFilters
        self.ratios = config.encoderRatios.reversed()
        self.depths = config.encoderDepthsArray
        self.causal = config.causal

        let numStages = depths.count

        var downsampleLayersList: [SConv1d] = []
        let stem = SConv1d(
            inChannels: channels,
            outChannels: nFilters,
            kernelSize: 7,
            bias: config.convBias,
            causal: causal,
            padMode: config.padMode
        )
        downsampleLayersList.append(stem)

        for i in 0..<ratios.count {
            let inCh = nFilters * Int(pow(2.0, Double(i)))
            let outCh = nFilters * Int(pow(2.0, Double(i + 1)))
            let downsample = SConv1d(
                inChannels: inCh,
                outChannels: outCh,
                kernelSize: ratios[i] * 2,
                stride: ratios[i],
                bias: config.convBias,
                causal: causal,
                padMode: config.padMode
            )
            downsampleLayersList.append(downsample)
        }
        _downsampleLayers.wrappedValue = downsampleLayersList

        var stagesList: [[Block1D]] = []
        for i in 0..<numStages {
            let inCh = nFilters * Int(pow(2.0, Double(i)))
            var stageBlocks: [Block1D] = []
            for _ in 0..<depths[i] {
                stageBlocks.append(Block1D(
                    dim: inCh,
                    kernelSize: 7,
                    mixerLayer: config.mixerLayer,
                    layerScaleInitValue: config.layerScaleInitValue,
                    causal: causal,
                    padMode: config.padMode,
                    bias: config.convBias,
                    eps: config.layernormEps
                ))
            }
            stagesList.append(stageBlocks)
        }
        _stages.wrappedValue = stagesList

        let lastCh = nFilters * Int(pow(2.0, Double(numStages - 1)))
        if config.disableLastNorm {
            self.normFinal = Identity()
        } else {
            self.normFinal = ConvRMSNorm(dimensions: lastCh, eps: config.layernormEps)
        }

        _head.wrappedValue = SConv1d(
            inChannels: lastCh,
            outChannels: dimension,
            kernelSize: 7,
            bias: config.convBias,
            causal: causal,
            padMode: config.padMode
        )

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: StreamingConvCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        var out = forwardFeatures(x, cache: cache, useCache: useCache)
        out = head(out, cache: cache, useCache: useCache)
        return out
    }

    private func forwardFeatures(
        _ x: MLXArray,
        cache: StreamingConvCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        var out = x

        for i in 0..<depths.count {
            out = downsampleLayers[i](out, cache: cache, useCache: useCache)

            for block in stages[i] {
                out = block(out, cache: cache, useCache: useCache)
            }
        }

        if let convNorm = normFinal as? ConvRMSNorm {
            out = convNorm(out)
        }

        return out
    }
}
