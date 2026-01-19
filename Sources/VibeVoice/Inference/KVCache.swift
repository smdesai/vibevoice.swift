import Foundation
import MLX
import MLXFast

public protocol KVCache: AnyObject {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
}

public class KVCacheSimple: KVCache {
    /// Maximum sequence length to bound memory usage (default 2048 tokens)
    public static var maxSequenceLength: Int = 2048

    internal var keys: MLXArray?
    internal var values: MLXArray?

    public private(set) var offset: Int = 0

    public var step: Int = 256

    /// Whether to use sliding window when approaching max length
    public var useSlidingWindow: Bool = true

    /// Track dimensions for pre-allocation
    private var cachedB: Int = 0
    private var cachedKvHeads: Int = 0
    private var cachedKHeadDim: Int = 0
    private var cachedVHeadDim: Int = 0

    public init(step: Int = 256) {
        self.step = step
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset
        let numNewTokens = keys.dim(2)

        // Check if we need to apply sliding window
        if useSlidingWindow && previous + numNewTokens > Self.maxSequenceLength {
            applySlidingWindow(keepTokens: Self.maxSequenceLength / 2)
        }

        let currentOffset = self.offset  // May have changed after sliding window

        let needsAllocation: Bool
        if let currentKeys = self.keys {
            needsAllocation = (currentOffset + numNewTokens) > currentKeys.dim(2)
        } else {
            needsAllocation = true
        }

        if needsAllocation {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            // Cache dimensions for potential reallocation
            cachedB = B
            cachedKvHeads = kvHeads
            cachedKHeadDim = kHeadDim
            cachedVHeadDim = vHeadDim

            // Pre-allocate to bounded size to avoid repeated reallocations
            let allocSize = min(
                ((currentOffset + numNewTokens + step - 1) / step) * step,
                Self.maxSequenceLength
            )
            let kShape = [B, kvHeads, allocSize, kHeadDim]
            let vShape = [B, kvHeads, allocSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values, currentOffset > 0 {
                // Copy existing data to new buffer (only up to current offset)
                newK[.ellipsis, ..<currentOffset, 0...] =
                    currentKeys[.ellipsis, ..<currentOffset, 0...]
                newV[.ellipsis, ..<currentOffset, 0...] =
                    currentValues[.ellipsis, ..<currentOffset, 0...]
            }

            self.keys = newK
            self.values = newV
        }

        let newOffset = currentOffset + numNewTokens
        self.offset = newOffset

        self.keys?[.ellipsis, currentOffset ..< newOffset, 0...] = keys
        self.values?[.ellipsis, currentOffset ..< newOffset, 0...] = values

        guard let k = self.keys, let v = self.values else {
            return (keys, values)
        }

        return (k[.ellipsis, ..<newOffset, 0...], v[.ellipsis, ..<newOffset, 0...])
    }

    /// Apply sliding window to keep only the most recent tokens
    private func applySlidingWindow(keepTokens: Int) {
        guard let currentKeys = self.keys, let currentValues = self.values else { return }
        guard offset > keepTokens else { return }

        let dropTokens = offset - keepTokens

        // Shift data: keep tokens from dropTokens..<offset, move to 0..<keepTokens
        let keptKeys = currentKeys[.ellipsis, dropTokens ..< offset, 0...]
        let keptValues = currentValues[.ellipsis, dropTokens ..< offset, 0...]

        // Write back to beginning of buffer
        self.keys?[.ellipsis, ..<keepTokens, 0...] = keptKeys
        self.values?[.ellipsis, ..<keepTokens, 0...] = keptValues

        self.offset = keepTokens
    }

    public func reset() {
        self.keys = nil
        self.values = nil
        self.offset = 0
    }

    public func initialize(keys: MLXArray, values: MLXArray) {
        self.keys = keys
        self.values = values
        self.offset = keys.dim(2)
    }

    public var sequenceLength: Int {
        keys?.dim(2) ?? 0
    }
}

public func createCausalMask(n: Int, offset: Int) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    return linds .>= rinds
}

public func createAttentionMask(h: MLXArray, cache: KVCache?)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let n = h.dim(1)

    if n == 1 {
        return .none
    }

    let offset = cache?.offset ?? 0
    return .array(createCausalMask(n: n, offset: offset))
}

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let (updatedKeys, updatedValues): (MLXArray, MLXArray)

    if let cache = cache {
        (updatedKeys, updatedValues) = cache.update(keys: keys, values: values)
    } else {
        (updatedKeys, updatedValues) = (keys, values)
    }

    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: updatedKeys,
        values: updatedValues,
        scale: scale,
        mask: mask
    )
}
