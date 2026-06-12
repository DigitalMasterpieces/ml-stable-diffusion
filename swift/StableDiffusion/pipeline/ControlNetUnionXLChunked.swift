// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation
import CoreML
import Accelerate

/// Chunk names for architectural ControlNet Union chunking
/// This 6-chunk architecture is designed for ControlNet Union ProMax which has 3 down_blocks
/// (not 4 like standard SDXL). Total of 10 residual outputs (additional_residual_0 through _9).
/// DeltaDown2 is split at the resnet boundary into DeltaDown2A and DeltaDown2B (~750MB each).
public let CONTROLNET_UNION_CHUNK_NAMES = [
    "ControlNetAlphaTimeEmbed",   // Chunk 0: time/control embedding (sin/cos) - can run on CPU/GPU
    "ControlNetBetaCondFusion",   // Chunk 1: conditioning fusion
    "ControlNetGammaDown01",      // Chunk 2: down_blocks[0,1] with projections
    "ControlNetDeltaDown2A",      // Chunk 3: down_blocks[2] first resnet+attn
    "ControlNetDeltaDown2B",      // Chunk 4: down_blocks[2] second resnet+attn
    "ControlNetEpsilonMid",       // Chunk 5: mid_block with projection
]

/// Architectural chunked ControlNet Union for Neural Engine compatibility on mobile devices
/// Splits ControlNet Union into 6 chunks: AlphaTimeEmbed, BetaCondFusion, GammaDown01, DeltaDown2A, DeltaDown2B, EpsilonMid
///
/// This is the chunked version of `ControlNetUnionXL` for when the single Union model
/// is too large for Neural Engine on mobile devices.
@available(iOS 16.2, macOS 13.1, *)
public struct ControlNetUnionXLChunked: ResourceManaging, ControlNetXLProtocol {

    /// The 6 architectural chunk models
    public var chunks: [ManagedMLModel]

    /// Number of control types supported (e.g., 8 for Union ProMax)
    public let numControlTypes: Int

    public init(chunkURLs: [URL], configuration: MLModelConfiguration, numControlTypes: Int = 8) {
        precondition(chunkURLs.count == 6,
            "ControlNetUnionXLChunked requires exactly 6 chunk URLs (AlphaTimeEmbed, BetaCondFusion, GammaDown01, DeltaDown2A, DeltaDown2B, EpsilonMid). " +
            "This chunked version is only for Union ControlNet models. Got \(chunkURLs.count) URLs.")
        precondition(numControlTypes > 0,
            "ControlNetUnionXLChunked requires numControlTypes > 0. Union ControlNet models typically have 6 control types.")

        // Load chunks with appropriate compute units
        // Chunks 0 (AlphaTimeEmbed) and 1 (BetaCondFusion) need CPU+GPU to avoid Neural Engine issues:
        // - AlphaTimeEmbed: sin/cos operators can fail on Neural Engine (e.g., M1 iPad Pro)
        // - BetaCondFusion: reshape operations in nn.MultiheadAttention cause Neural Engine compiler errors
        self.chunks = chunkURLs.enumerated().map { (index, url) in
            let chunkConfig: MLModelConfiguration
            if index == 0 || index == 1 {
                chunkConfig = MLModelConfiguration()
                chunkConfig.computeUnits = .cpuAndGPU
            } else {
                chunkConfig = configuration
            }
            return ManagedMLModel(modelAt: url, configuration: chunkConfig)
        }
        self.numControlTypes = numControlTypes
    }

    /// Compatibility with ControlNetXLProtocol
    public var models: [ManagedMLModel] {
        return chunks
    }

    public var loadProgressWeights: [Int64] {
        // Approximate weight distribution across 6 chunks
        // AlphaTimeEmbed is small, BetaCondFusion is medium, GammaDown01 is large,
        // DeltaDown2A and DeltaDown2B are medium (~750MB each), EpsilonMid is small
        return [1, 4, 7, 4, 4, 2]  // Total: 22
    }

    public func makeLoadProgress() -> Progress {
        let total = self.loadProgressWeights.reduce(0, +)
        let progress = Progress(totalUnitCount: total)
        progress.localizedDescription = "ControlNet Union (Chunked)"
        return progress
    }

    /// Load resources.
    public func loadResources(progress: Progress, prewarm: Bool) throws {
        for (index, chunk) in chunks.enumerated() {
            let chunkProgress = Progress(totalUnitCount: loadProgressWeights[index], parent: progress, pendingUnitCount: loadProgressWeights[index])
            try chunk.loadResources(progress: chunkProgress)
            if prewarm {
                chunk.unloadResources()
            }
        }
    }

    /// Unload the underlying models to free up memory
    public func unloadResources() {
        for chunk in chunks {
            chunk.unloadResources()
        }
    }

    public var inputImageDescriptions: [MLFeatureDescription] {
        // For chunked ControlNet Union, get all controlnet_cond_* inputs from the BetaCondFusion chunk (index 1)
        try! chunks[1].perform { model in
            model.modelDescription.inputDescriptionsByName
                .filter { $0.key.hasPrefix("controlnet_cond_") }
                .map { $0.value }
        }
    }

    /// The expected shape of the models image input
    public var inputImageShapes: [[Int]] {
        inputImageDescriptions.map { desc in
            desc.multiArrayConstraint!.shape.map { $0.intValue }
        }
    }

    /// The expected output shapes - derived from actual CoreML model outputs
    public var outputShapes: [[String: [Int]]] {
        // Read shapes from outputDescriptions which queries actual CoreML models
        var combinedOutputs: [String: [Int]] = [:]
        for (name, desc) in outputDescriptions[0] {
            if let constraint = desc.multiArrayConstraint {
                combinedOutputs[name] = constraint.shape.map { $0.intValue }
            }
        }
        return [combinedOutputs]
    }

    public var outputDescriptions: [[String : MLFeatureDescription]] {
        // For chunked ControlNet, we combine the residual outputs from chunks 2-5
        var combinedDescriptions: [String: MLFeatureDescription] = [:]

        // Get outputs from GammaDown01 (chunk 2) - residuals 0-6
        try! chunks[2].perform { model in
            for (name, desc) in model.modelDescription.outputDescriptionsByName {
                if name.starts(with: "additional_residual") {
                    combinedDescriptions[name] = desc
                }
            }
        }

        // Get outputs from DeltaDown2A (chunk 3) - residual 7
        try! chunks[3].perform { model in
            for (name, desc) in model.modelDescription.outputDescriptionsByName {
                if name.starts(with: "additional_residual") {
                    combinedDescriptions[name] = desc
                }
            }
        }

        // Get outputs from DeltaDown2B (chunk 4) - residual 8
        try! chunks[4].perform { model in
            for (name, desc) in model.modelDescription.outputDescriptionsByName {
                if name.starts(with: "additional_residual") {
                    combinedDescriptions[name] = desc
                }
            }
        }

        // Get outputs from EpsilonMid (chunk 5) - residual 9
        try! chunks[5].perform { model in
            for (name, desc) in model.modelDescription.outputDescriptionsByName {
                if name.starts(with: "additional_residual") {
                    combinedDescriptions[name] = desc
                }
            }
        }

        return [combinedDescriptions]
    }

    /// **SpatialControlNet** — optional per-control-type **spatial** conditioning-scale maps,
    /// indexed `[controlNetModel][controlType]`, each an `[1, H, W]` (latent-resolution) weight
    /// map (or `H*W` scalars). When set, `execute` builds the model's
    /// `conditioning_scale_spatial` input from these maps: per control type it uses the map if
    /// present, otherwise a constant fill of that type's scalar from `conditioningScales`.
    /// When `nil`, every type is filled uniformly from its scalar — identical to the legacy
    /// scalar `conditioning_scale` behavior. Reset to `nil` to return to global scales.
    ///
    /// Note: the all-zero early-out still reads `conditioningScales`, so when driving with maps
    /// pass representative non-zero scalars (e.g. each map's max) for the active types.
    public var conditioningScaleMaps: [[MLShapedArray<Float32>?]]? = nil

    /// **SpatialControlNet detection.** Whether this ControlNet was converted with SPATIAL
    /// conditioning weights (the converter's `--spatial_controlnet` flag). The spatial
    /// BetaCondFusion chunk declares `conditioning_scale_spatial` as a 4-D `[num, 1, H, W]`
    /// map input; the default scalar chunk declares `conditioning_scale` as a 1-D `[num]`
    /// vector. We detect the variant by querying that input's **rank** on the BetaCondFusion
    /// chunk (rank ≥ 4 ⇒ spatial) and branch `execute` accordingly, so a single Swift build
    /// drives either converted model. The spatial code path — building
    /// `conditioning_scale_spatial`, honoring `conditioningScaleMaps`, and spatial residual
    /// post-scaling — only applies when this is `true`; otherwise the legacy scalar
    /// `conditioning_scale` path runs. Defaults to `false` (scalar) if the descriptor can't be
    /// read (e.g. model not yet loaded).
    public var isSpatialConditioning: Bool {
        (try? chunks[1].perform { model in
            let inputs = model.modelDescription.inputDescriptionsByName
            let desc = inputs["conditioning_scale_spatial"] ?? inputs["conditioning_scale"]
            let rank = desc?.multiArrayConstraint?.shape.count ?? 1
            return rank >= 4
        }) ?? false
    }

    /// Build the `conditioning_scale_spatial` tensor `[numControlTypes, 1, H, W]`. Each control
    /// type's plane is its painted map (when provided and sized `>= H*W`) or a constant fill of
    /// its scalar — so a uniform fill reproduces the legacy scalar `conditioning_scale[i]`.
    private static func buildConditioningScaleSpatial(
        scales: [Float],
        maps: [MLShapedArray<Float32>?]?,
        numControlTypes: Int,
        latentH: Int,
        latentW: Int
    ) -> MLShapedArray<Float32> {
        let plane = latentH * latentW
        var scalars = [Float](repeating: 1.0, count: numControlTypes * plane)
        for t in 0..<numControlTypes {
            let base = t * plane
            if let maps, t < maps.count, let map = maps[t] {
                let mv = map.scalars
                for k in 0..<plane { scalars[base + k] = k < mv.count ? mv[k] : 0 }
            } else {
                let s = t < scales.count ? scales[t] : 1.0
                for k in 0..<plane { scalars[base + k] = s }
            }
        }
        return MLShapedArray<Float32>(scalars: scalars, shape: [numControlTypes, 1, latentH, latentW])
    }

    /// Bilinear-resample a single-channel (row-major `[H, W]`) plane. Used to apply a spatial
    /// conditioning-scale map to ControlNet residuals at each residual's own resolution.
    private static func resamplePlane(_ src: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> [Float] {
        if srcW == dstW && srcH == dstH { return src }
        var s = src
        var dst = [Float](repeating: 0, count: max(dstW * dstH, 1))
        s.withUnsafeMutableBufferPointer { sp in
            dst.withUnsafeMutableBufferPointer { dp in
                var srcBuf = vImage_Buffer(
                    data: sp.baseAddress, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW),
                    rowBytes: srcW * MemoryLayout<Float>.stride)
                var dstBuf = vImage_Buffer(
                    data: dp.baseAddress, height: vImagePixelCount(dstH), width: vImagePixelCount(dstW),
                    rowBytes: dstW * MemoryLayout<Float>.stride)
                _ = vImageScale_PlanarF(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        return dst
    }

    /// Execute chunked ControlNet Union prediction
    ///
    /// Data flow:
    /// - AlphaTimeEmbed: (timestep, control_type) → emb
    /// - BetaCondFusion: (sample, controlnet_cond_*, conditioning_scale_spatial) → fused_sample
    /// - GammaDown01: (fused_sample, emb, encoder_hidden_states) → (residuals 0-6, hidden [B, 640, H/4, W/4])
    /// - DeltaDown2A: (hidden, emb, encoder_hidden_states) → (residual 7, hidden [B, 1280, H/4, W/4])
    /// - DeltaDown2B: (hidden, emb, encoder_hidden_states) → (residual 8, hidden [B, 1280, H/4, W/4])
    /// - EpsilonMid: (hidden, emb, encoder_hidden_states) → (residual 9)
    public func execute(
        latents: [MLShapedArray<Float32>],
        timeStep: Double,
        hiddenStates: MLShapedArray<Float32>,
        pooledStates: MLShapedArray<Float32>,
        geometryConditioning: MLShapedArray<Float32>,
        conditioningScales: [[Float]],
        controlTypes: [[UInt]],
        images: [[MLShapedArray<Float32>?]]
    ) throws -> [[String: MLShapedArray<Float32>]] {
        // Early return if no ControlNet images are provided
        let hasAnyImage = images.contains { modelImages in
            modelImages.contains { $0 != nil }
        }
        if !hasAnyImage {
            return []
        }

        let batchSize = hiddenStates.shape[0]
        let t: MLShapedArray<Float32>
        if batchSize == 2 {
            t = MLShapedArray(scalars: [Float(timeStep), Float(timeStep)], shape: [2])
        } else {
            t = MLShapedArray(scalars: [Float(timeStep)], shape: [1])
        }

        var outputs: [[String: MLShapedArray<Float32>]] = Array(repeating: [:], count: latents.count)

        // Early-out if all conditioning scales are zero (no ControlNet effect)
        let allScalesZero = conditioningScales.first?.allSatisfy { $0 == 0.0 } ?? true
        if allScalesZero {
            return outputs
        }

        // Build control_type tensor from the first model's control types
        let controlTypeArray = controlTypes.first ?? Array(repeating: 0, count: numControlTypes)
        let controlTypeFloats = controlTypeArray.map { Float($0) }
        let controlType: MLShapedArray<Float32>
        if batchSize == 2 {
            controlType = MLShapedArray<Float32>(
                scalars: Array(repeating: controlTypeFloats, count: 2).flatMap { $0 },
                shape: [2, numControlTypes]
            )
        } else {
            controlType = MLShapedArray<Float32>(scalars: controlTypeFloats, shape: [1, numControlTypes])
        }

        // Normalize the per-control-type scalar conditioning scales into `activeScales`
        // (padded/truncated to `numControlTypes`). Both conditioning-weight variants use this:
        // the scalar path feeds it directly as `conditioning_scale`, and the spatial path uses
        // each scalar as the constant fill for any control type that has no painted map.
        let inputScales = conditioningScales.first ?? Array(repeating: 1.0, count: numControlTypes)
        var activeScales: [Float] = Array(repeating: 1.0, count: numControlTypes)
        for (i, scale) in inputScales.enumerated() where i < numControlTypes {
            activeScales[i] = scale
        }
        guard let firstLatentShape = latents.first?.shape, firstLatentShape.count >= 4 else {
            return outputs
        }
        let latentH = firstLatentShape[2]
        let latentW = firstLatentShape[3]
        // Detect once which conditioning-weight contract the converted model expects, and build
        // only that tensor. Spatial (SpatialControlNet): per-type `[num, 1, H, W]` maps. Scalar
        // (default): the legacy `[num]` vector.
        let spatialConditioning = self.isSpatialConditioning
        let condScaleSpatial: MLShapedArray<Float32>? = spatialConditioning
            ? Self.buildConditioningScaleSpatial(
                scales: activeScales,
                maps: conditioningScaleMaps?.first,
                numControlTypes: numControlTypes,
                latentH: latentH,
                latentW: latentW)
            : nil
        let condScaleScalar: MLShapedArray<Float32>? = spatialConditioning
            ? nil
            : MLShapedArray<Float32>(scalars: activeScales, shape: [numControlTypes])

        // Get expected image shape for creating zero placeholders for missing control types
        let expectedImageShape = self.inputImageShapes.first ?? [batchSize, 3, 1024, 1024]

        for (latentIndex, latent) in latents.enumerated() {
            // --- Stage 0: AlphaTimeEmbed ---
            // Inputs: timestep, text_embeds, time_ids, control_type
            // Outputs: emb [B, 1280, 1, 1]
            let alphaInputs: [String: Any] = [
                "timestep": MLMultiArray(t),
                "text_embeds": MLMultiArray(pooledStates),
                "time_ids": MLMultiArray(geometryConditioning),
                "control_type": MLMultiArray(controlType),
            ]

            let alphaProvider = try MLDictionaryFeatureProvider(dictionary: alphaInputs)
            let alphaResult = try chunks[0].perform { model in
                try model.prediction(from: alphaProvider)
            }

            let emb = alphaResult.featureValue(for: "emb_out")!.multiArrayValue!

            // --- Stage 1: BetaCondFusion ---
            // Inputs: sample, controlnet_cond_*, and the conditioning weights under whichever
            // key the converted model declares (`conditioning_scale_spatial` for the spatial /
            // SpatialControlNet variant, `conditioning_scale` for the default scalar variant).
            // Outputs: fused_sample [B, 320, H, W]
            var betaInputs: [String: Any] = [
                "sample": MLMultiArray(latent),
            ]
            if let condScaleSpatial {
                betaInputs["conditioning_scale_spatial"] = MLMultiArray(condScaleSpatial)
            } else if let condScaleScalar {
                betaInputs["conditioning_scale"] = MLMultiArray(condScaleScalar)
            }

            // BetaCondFusion expects ALL numControlTypes controlnet_cond inputs
            // Missing images should be zeros (with the conditioning applied correctly per type)
            let firstImages = images.first ?? []
            for i in 0..<numControlTypes {
                if i < firstImages.count, let image = firstImages[i] {
                    betaInputs["controlnet_cond_\(i)"] = MLMultiArray(image)
                } else {
                    // Create zero tensor for missing control type
                    let zeroImage = MLShapedArray<Float32>(repeating: 0.0, shape: expectedImageShape)
                    betaInputs["controlnet_cond_\(i)"] = MLMultiArray(zeroImage)
                }
            }

            let betaProvider = try MLDictionaryFeatureProvider(dictionary: betaInputs)
            let betaResult = try chunks[1].perform { model in
                try model.prediction(from: betaProvider)
            }

            let fusedSample = betaResult.featureValue(for: "fused_sample_out")!.multiArrayValue!

            // --- Stage 2: GammaDown01 ---
            // Inputs: fused_sample, emb, encoder_hidden_states
            // Outputs: residuals 0-6, hidden_out [B, 640, H/4, W/4]
            let gammaInputs: [String: Any] = [
                "fused_sample": fusedSample,
                "emb": emb,
                "encoder_hidden_states": MLMultiArray(hiddenStates),
            ]

            let gammaProvider = try MLDictionaryFeatureProvider(dictionary: gammaInputs)
            let gammaResult = try chunks[2].perform { model in
                try model.prediction(from: gammaProvider)
            }

            // Collect residuals from GammaDown01 (dynamically)
            for name in gammaResult.featureNames {
                if name.starts(with: "additional_residual") {
                    let residual = gammaResult.featureValue(for: name)!.multiArrayValue!
                    outputs[latentIndex][name] = MLShapedArray<Float32>(residual)
                }
            }
            let hiddenGamma = gammaResult.featureValue(for: "hidden_out")!.multiArrayValue!

            // --- Stage 3: DeltaDown2A ---
            // Inputs: hidden [B, 640, H/4, W/4], emb, encoder_hidden_states
            // Outputs: residual 7, hidden_out [B, 1280, H/4, W/4]
            let deltaAInputs: [String: Any] = [
                "hidden": hiddenGamma,
                "emb": emb,
                "encoder_hidden_states": MLMultiArray(hiddenStates),
            ]

            let deltaAProvider = try MLDictionaryFeatureProvider(dictionary: deltaAInputs)
            let deltaAResult = try chunks[3].perform { model in
                try model.prediction(from: deltaAProvider)
            }

            // Collect residuals from DeltaDown2A (dynamically)
            for name in deltaAResult.featureNames {
                if name.starts(with: "additional_residual") {
                    let residual = deltaAResult.featureValue(for: name)!.multiArrayValue!
                    outputs[latentIndex][name] = MLShapedArray<Float32>(residual)
                }
            }
            let hiddenDeltaA = deltaAResult.featureValue(for: "hidden_out")!.multiArrayValue!

            // --- Stage 4: DeltaDown2B ---
            // Inputs: hidden [B, 1280, H/4, W/4], emb, encoder_hidden_states
            // Outputs: residual 8, hidden_out [B, 1280, H/4, W/4]
            let deltaBInputs: [String: Any] = [
                "hidden": hiddenDeltaA,
                "emb": emb,
                "encoder_hidden_states": MLMultiArray(hiddenStates),
            ]

            let deltaBProvider = try MLDictionaryFeatureProvider(dictionary: deltaBInputs)
            let deltaBResult = try chunks[4].perform { model in
                try model.prediction(from: deltaBProvider)
            }

            // Collect residuals from DeltaDown2B (dynamically)
            for name in deltaBResult.featureNames {
                if name.starts(with: "additional_residual") {
                    let residual = deltaBResult.featureValue(for: name)!.multiArrayValue!
                    outputs[latentIndex][name] = MLShapedArray<Float32>(residual)
                }
            }
            let hiddenDeltaB = deltaBResult.featureValue(for: "hidden_out")!.multiArrayValue!

            // --- Stage 5: EpsilonMid ---
            // Inputs: hidden [B, 1280, H/4, W/4], emb, encoder_hidden_states
            // Outputs: residual 9
            let epsilonInputs: [String: Any] = [
                "hidden": hiddenDeltaB,
                "emb": emb,
                "encoder_hidden_states": MLMultiArray(hiddenStates),
            ]

            let epsilonProvider = try MLDictionaryFeatureProvider(dictionary: epsilonInputs)
            let epsilonResult = try chunks[5].perform { model in
                try model.prediction(from: epsilonProvider)
            }

            // Collect residuals from EpsilonMid (dynamically)
            for name in epsilonResult.featureNames {
                if name.starts(with: "additional_residual") {
                    let residual = epsilonResult.featureValue(for: name)!.multiArrayValue!
                    outputs[latentIndex][name] = MLShapedArray<Float32>(residual)
                }
            }
        }

        // Output-level (residual) scaling — the dominant-weight "max activator", applied PER
        // LATENT COMPONENT so it covers the spatial (SpatialControlNet) case as well as scalar.
        //
        // The per-type conditioning scale is already applied once at fusion by BetaCondFusion.
        // On TOP of that we multiply the residuals by the dominant active weight — restoring the
        // double-application that tuned the look (e.g. Tile@0.5 → `R = 0.5·B(c₀ + 0.5·Tile)`),
        // but made CONTINUOUS across active-unit count by using the MAX active weight rather than
        // the old `== 1` single-type gate (which made the second multiply VANISH the instant a
        // second unit turned on — the ≈1/scale jump when enabling Canny).
        //
        // Same idea in two regimes:
        //   • Scalar weights (default model, or a spatial-converted model in normal generation
        //     with uniform fills, i.e. `conditioningScaleMaps == nil`): the dominant weight is a
        //     single constant `max(active scales)` → one uniform residual multiply (fast path).
        //     Byte-for-byte the prior scalar behavior.
        //   • Spatial maps driving the pass (SpatialControlNet + `conditioningScaleMaps`, e.g.
        //     Level of Abstraction): the dominant weight VARIES per location. Take the
        //     per-location max ACROSS control types — `condScaleSpatial` already holds each
        //     type's painted map / constant-scalar plane (inactive types are 0) — then resample
        //     that single plane to each residual's own resolution and multiply element-wise. With
        //     one active map this is exactly `residual *= map` (the pre-`max()` spatial post-
        //     scaling that was committed, tuned, and working); with a uniform map it collapses to
        //     the scalar case; mixing a map with a scalar unit keeps the per-location max, so the
        //     continuity property holds everywhere.
        if spatialConditioning, conditioningScaleMaps != nil, let condScaleSpatial {
            // Per-location dominant weight: max over the control-type axis of the
            // [numControlTypes, 1, latentH, latentW] fusion tensor.
            let plane = latentH * latentW
            let cs = condScaleSpatial.scalars
            var maxPlane = [Float](repeating: 0.0, count: plane)
            for tIdx in 0..<numControlTypes {
                let base = tIdx * plane
                for k in 0..<plane where cs[base + k] > maxPlane[k] {
                    maxPlane[k] = cs[base + k]
                }
            }
            // A uniformly-1.0 plane is the identity — nothing to rescale.
            if !maxPlane.allSatisfy({ $0 == 1.0 }) {
                for n in 0..<outputs.count {
                    for key in outputs[n].keys {
                        guard let residual = outputs[n][key], residual.shape.count == 4 else { continue }
                        let bc = residual.shape[0] * residual.shape[1]
                        let rh = residual.shape[2], rw = residual.shape[3]
                        let rplane = rh * rw
                        guard rplane > 0 else { continue }
                        // Resample the latent-resolution dominant-weight plane to this residual's h×w.
                        let weights = (rw == latentW && rh == latentH)
                            ? maxPlane
                            : Self.resamplePlane(maxPlane, srcW: latentW, srcH: latentH, dstW: rw, dstH: rh)
                        outputs[n][key] = MLShapedArray<Float32>(unsafeUninitializedShape: residual.shape) { ptr, _ in
                            residual.withUnsafeShapedBufferPointer { src, _, _ in
                                let sp = src.baseAddress!
                                let op = ptr.baseAddress!
                                weights.withUnsafeBufferPointer { wp in
                                    let wb = wp.baseAddress!
                                    // Broadcast the single h×w plane across batch × channels.
                                    for p in 0..<bc {
                                        let b = p * rplane
                                        vDSP_vmul(sp + b, 1, wb, 1, op + b, 1, vDSP_Length(rplane))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            let nonZeroScales = conditioningScales.first?.filter { $0 > 0.0 } ?? []
            if let effectiveScale = nonZeroScales.max(), effectiveScale != 1.0 {
                for n in 0..<outputs.count {
                    for key in outputs[n].keys {
                        if let residual = outputs[n][key] {
                            outputs[n][key] = MLShapedArray<Float32>(unsafeUninitializedShape: residual.shape) { ptr, _ in
                                residual.withUnsafeShapedBufferPointer { src, _, _ in
                                    var scale = effectiveScale
                                    vDSP_vsmul(src.baseAddress!, 1, &scale, ptr.baseAddress!, 1, vDSP_Length(src.count))
                                }
                            }
                        }
                    }
                }
            }
        }

        // Initialize missing outputs (shouldn't happen, but for safety).
        for (outputName, shape) in self.outputShapes[0] {
            for n in 0..<latents.count {
                if outputs[n][outputName] == nil {
                    outputs[n][outputName] = MLShapedArray<Float32>(
                        repeating: 0.0,
                        shape: shape
                    )
                }
            }
        }

        return outputs
    }
}
