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

    /// Execute chunked ControlNet Union prediction
    ///
    /// Data flow:
    /// - AlphaTimeEmbed: (timestep, control_type) → emb
    /// - BetaCondFusion: (sample, controlnet_cond_*, conditioning_scale) → fused_sample
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

        // Get conditioning scale (use first one if available)
        guard let conditioningScale = conditioningScales.first?.first else {
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

        // Build conditioning_scale tensor from actual input scales
        // conditioningScales is [[Float]] where first array has scales per control type
        let inputScales = conditioningScales.first ?? Array(repeating: 1.0, count: numControlTypes)
        var activeScales: [Float] = Array(repeating: 1.0, count: numControlTypes)
        for (i, scale) in inputScales.enumerated() where i < numControlTypes {
            activeScales[i] = scale
        }
        let condScaleArray = MLShapedArray<Float32>(
            scalars: activeScales,
            shape: [numControlTypes]
        )

        // Get expected image shape for creating zero placeholders for missing control types
        let expectedImageShape = self.inputImageShapes.first ?? [batchSize, 3, 1024, 1024]

        for (latentIndex, latent) in latents.enumerated() {
            // --- Stage 0: AlphaTimeEmbed ---
            // Inputs: timestep, control_type
            // Outputs: emb [B, 1280, 1, 1]
            let alphaInputs: [String: Any] = [
                "timestep": MLMultiArray(t),
                "control_type": MLMultiArray(controlType),
            ]

            let alphaProvider = try MLDictionaryFeatureProvider(dictionary: alphaInputs)
            let alphaResult = try chunks[0].perform { model in
                try model.prediction(from: alphaProvider)
            }

            let emb = alphaResult.featureValue(for: "emb_out")!.multiArrayValue!

            // --- Stage 1: BetaCondFusion ---
            // Inputs: sample, controlnet_cond_*, conditioning_scale
            // Outputs: fused_sample [B, 320, H, W]
            var betaInputs: [String: Any] = [
                "sample": MLMultiArray(latent),
                "conditioning_scale": MLMultiArray(condScaleArray),
            ]

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

        // Initialize missing outputs (shouldn't happen, but for safety)
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
