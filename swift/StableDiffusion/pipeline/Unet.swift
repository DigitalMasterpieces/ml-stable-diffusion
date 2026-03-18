// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation
import CoreML
import os

/// U-Net noise prediction model for stable diffusion
@available(iOS 16.2, macOS 13.1, *)
public struct Unet: ResourceManaging {

    /// Model used to predict noise residuals given an input, diffusion time step, and conditional embedding
    ///
    /// It can be in the form of a single model or multiple stages
    var models: [ManagedMLModel]

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - url: Location of single U-Net  compiled Core ML model
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.models = [ManagedMLModel(modelAt: url, configuration: configuration)]
    }

    /// Creates a U-Net noise prediction model
    ///
    /// - Parameters:
    ///   - urls: Location of chunked U-Net via urls to each compiled chunk
    ///   - configuration: Configuration to be used when the model is loaded
    /// - Returns: U-net model that will lazily load its required resources when needed or requested
    public init(chunksAt urls: [URL],
                configuration: MLModelConfiguration) {
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
    }

    public var loadProgressWeights: [Int64] {
        if models.count == 2 {
            return [35, 35]
        } else if models.count == 4 {
            return [15, 20, 20, 15]
        } else if models.count == 11 {
            // Architectural 11-chunk: AlphaA, AlphaB, GammaA, GammaB, Sigma, ThetaA, ThetaB, ThetaC, Lambda, Kappa, Omega
            return [2, 16, 6, 6, 12, 6, 4, 4, 10, 10, 4]  // Total: 80
        } else {
            return [70]
        }
    }

    public func makeLoadProgress() -> Progress {
        let total = self.loadProgressWeights.reduce(0, +)
        let progress = Progress(totalUnitCount: total)
        progress.localizedDescription = "UNet"

        for (index, units) in self.loadProgressWeights.enumerated() {
            let chunkProgress = Progress(totalUnitCount: units)
            chunkProgress.localizedDescription = "UNet block \(index + 1)"
            progress.addTrackedChild(chunkProgress, units: units)
        }

        return progress
    }

    /// Load resources.
    public func loadResources(progress: Progress, prewarm: Bool) throws {
        let chunks = progress.children
        assert(chunks.count == self.loadProgressWeights.count)

        for (index, chunkProgress) in chunks.enumerated() {
            // Update label shown in UI
            progress.rootProgress?.localizedDescription = chunkProgress.localizedDescription

            // Do the actual work
            try models[index].loadResources(progress: chunkProgress)

            if prewarm {
                models[index].unloadResources()
            }

            // Mark chunk complete
            chunkProgress.completedUnitCount = chunkProgress.totalUnitCount
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }

    var latentSampleDescription: MLFeatureDescription {
        if models.count == 11 {
            // AlphaEncoderB is at index 1 in 11-chunk architectural mode
            try! models[1].perform { model in
                model.modelDescription.inputDescriptionsByName["sample"]!
            }
        } else {
            try! models.first!.perform { model in
                model.modelDescription.inputDescriptionsByName["sample"]!
            }
        }
    }

    /// The expected shape of the models latent sample input
    public var latentSampleShape: [Int] {
        latentSampleDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    var latentTimeIdDescription: MLFeatureDescription {
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["time_ids"]!
        }
    }

    /// The expected shape of the geometry conditioning
    public var latentTimeIdShape: [Int] {
        latentTimeIdDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    /// Batch prediction noise from latent samples
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    /// - Returns: Array of predicted noise residuals
    func predictNoise(
        latents: [MLShapedArray<Float32>],
        timeStep: Double,
        hiddenStates: MLShapedArray<Float32>,
        additionalResiduals: [[String: MLShapedArray<Float32>]]? = nil
    ) throws -> [MLShapedArray<Float32>] {
        let noiseState = signposter.beginInterval("Predict Noise")
        defer { signposter.endInterval("Predict Noise", noiseState) }

        // Match time step batch dimension to the model / latent samples
        let t: MLShapedArray<Float32>
        if hiddenStates.shape[0] == 2 {
            t = MLShapedArray(scalars: [Float(timeStep), Float(timeStep)], shape: [2])
        } else {
            t = MLShapedArray(scalars: [Float(timeStep)], shape: [1])
        }

        // Form batch input to model
        let inputs = try latents.enumerated().map {
            var dict: [String: Any] = [
                "sample" : MLMultiArray($0.element),
                "timestep" : MLMultiArray(t),
                "encoder_hidden_states": MLMultiArray(hiddenStates)
            ]
            if let residuals = additionalResiduals?[$0.offset] {
                for (k, v) in residuals {
                    dict[k] = MLMultiArray(v)
                }
            }
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Make predictions
        let results = try models.predictions(from: batch)

        // Pull out the results in Float32 format
        let noise = (0..<results.count).map { i in

            let result = results.features(at: i)
            let outputName = result.featureNames.first!

            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!

            // To conform to this func return type make sure we return float32
            // Use the fact that the concatenating constructor for MLMultiArray
            // can do type conversion:
            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            return MLShapedArray<Float32>(fp32Noise)
        }

        return noise
    }

    /// Batch prediction noise from latent samples, for Stable Diffusion XL
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    ///   - pooledStates: Additional text states to condition on
    ///   - geometryConditioning: Condition on image geometry
    /// - Returns: Array of predicted noise residuals
    @available(iOS 17.0, macOS 14.0, *)
    func predictNoise(
        latents: [MLShapedArray<Float32>],
        timeStep: Double,
        hiddenStates: MLShapedArray<Float32>,
        pooledStates: MLShapedArray<Float32>,
        geometryConditioning: MLShapedArray<Float32>,
        additionalResiduals: [[String: MLShapedArray<Float32>]]? = nil,
        imageEmbeds: MLShapedArray<Float32>? = nil,
        ipAdapterScale: Float? = 1.0,
        reduceMemory: Bool = false
    ) throws -> [MLShapedArray<Float32>] {
        // Determine chunk mode for signpost message.
        let noiseState: OSSignpostIntervalState
        if models.count == 4 {
            noiseState = signposter.beginInterval("Predict Noise", "quad")
        } else if models.count == 11 {
            noiseState = signposter.beginInterval("Predict Noise", "architectural")
        } else {
            noiseState = signposter.beginInterval("Predict Noise", "single")
        }
        defer { signposter.endInterval("Predict Noise", noiseState) }

        // Match time step batch dimension to the model / latent samples
        // Infer batch size from hiddenStates shape (batch size 2 for CFG, 1 for no CFG)
        let batchSize = hiddenStates.shape[0]
        let t: MLShapedArray<Float32>
        if batchSize == 2 {
            t = MLShapedArray<Float32>(scalars: [Float(timeStep), Float(timeStep)], shape: [2])
        } else {
            t = MLShapedArray<Float32>(scalars: [Float(timeStep)], shape: [1])
        }

        // Form batch input to model
        let inputs = try latents.enumerated().map {
            var dict: [String: Any] = [
                "sample" : MLMultiArray($0.element),
                "timestep" : MLMultiArray(t),
                "encoder_hidden_states": MLMultiArray(hiddenStates),
                "text_embeds": MLMultiArray(pooledStates),
                "time_ids": MLMultiArray(geometryConditioning)
            ]
            if let residuals = additionalResiduals?[$0.offset] {
                for (k, v) in residuals {
                    dict[k] = MLMultiArray(v)
                }
            }
            if let ipAdapterScale = ipAdapterScale {
                let ipAdapterScaleArray = MLShapedArray<Float32>(scalars: [Float(ipAdapterScale)],shape: [1])
                dict["ip_adapter_scale"] = MLMultiArray(ipAdapterScaleArray)
            }
            if let imageEmbeds = imageEmbeds {
                dict["image_embeds"] = MLMultiArray(imageEmbeds)
            }
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Make predictions
        let results: MLBatchProvider
        if models.count == 4 {
            results = try models.unet_quad_predictions(from: batch)
        } else if models.count == 11 {
            results = try models.unet_architectural_predictions(from: batch, reduceMemory: reduceMemory)
        } else {
            results = try models.predictions(from: batch)
        }

        // Pull out the results in Float32 format
        let noise = (0..<results.count).map { i in

            let result = results.features(at: i)
            let outputName = result.featureNames.first!

            let outputNoise = result.featureValue(for: outputName)!.multiArrayValue!

            // To conform to this func return type make sure we return float32
            // Use the fact that the concatenating constructor for MLMultiArray
            // can do type conversion:
            let fp32Noise = MLMultiArray(
                concatenating: [outputNoise],
                axis: 0,
                dataType: .float32
            )
            return MLShapedArray<Float32>(fp32Noise)
        }

        return noise
    }
}

@available(iOS 16.2, macOS 13.1, *)
public extension Array where Element == ManagedMLModel {
    /// Performs batch predictions for the Unet using an array `[ManagedMLModel]` with four instances in a pipeline.
    /// - Parameter batch: Inputs for batched predictions.
    /// - Returns: Final prediction results after processing through all models.
    /// - Throws: Errors if the array is empty, predictions fail, or results can't be combined.
    func unet_quad_predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {
        let dependencies: [[Int]] = [
            [],       // stage 0
            [0],      // stage 1 depends on output of stage 0
            [1],      // stage 2 depends on output of stage 1
            [0, 1, 2] // stage 3 depends on outputs of stage 1 and 2, etc.
        ]

        let inputs = batch.arrayOfFeatureValueDictionaries
        var stageOutputs: [[[String: MLFeatureValue]]] = []   // cache outputs of all stages

        // --- Stage 0 ---
        var results = try signposter.withIntervalSignpost("UNet Stage", "Stage 0/4") {
            try self.first!.perform { model in
                try model.predictions(fromBatch: batch)
            }
        }
        stageOutputs.append(results.arrayOfFeatureValueDictionaries)

        // --- Remaining stages ---
        for (stageIndex, stage) in self.dropFirst().enumerated() {
            let actualIndex = stageIndex + 1   // because dropFirst shifts indices

            results = try signposter.withIntervalSignpost("UNet Stage", "Stage \(actualIndex)/4") {
                // Build merged inputs for this stage
                let mergedInputDicts: [[String: MLFeatureValue]] =
                    (0..<inputs.count).map { sampleIndex in
                        var merged = inputs[sampleIndex]   // start with original input

                        // merge outputs of all dependencies (in order)
                        for dep in dependencies[actualIndex] {
                            let depDict = stageOutputs[dep][sampleIndex]
                            for (k, v) in depDict { merged[k] = v }
                        }
                        return merged
                    }

                let providers = try mergedInputDicts.map {
                    try MLDictionaryFeatureProvider(dictionary: $0)
                }
                let nextBatch = MLArrayBatchProvider(array: providers)

                // predict
                return try stage.perform { model in
                    try model.predictions(fromBatch: nextBatch)
                }
            }

            stageOutputs.append(results.arrayOfFeatureValueDictionaries)
        }

        return results
    }

    /// Performs batch predictions for architectural 11-chunk UNet pipeline (SDXL)
    /// Chunks: AlphaEncoderA → AlphaEncoderB → GammaDownblockA → GammaDownblockB → SigmaCore →
    ///         ThetaUpblockA → ThetaUpblockB → ThetaUpblockC → LambdaUpblock → KappaUpblock → OmegaDecoder
    func unet_architectural_predictions(from batch: MLBatchProvider, reduceMemory: Bool = false) throws -> MLBatchProvider {
        let outputToInputNameMap: [String: String] = [
            "emb_out": "emb",
            "ip_hidden_states_out": "ip_hidden_states",
            "hidden_out": "hidden",
        ]

        // Stage-specific skip mappings
        // GammaDownblockA outputs skip_down2_0, GammaDownblockB outputs skip_down2_1
        // These feed into ThetaUpblockA/B/C in reverse order
        let stageSkipMappings: [Int: [String: String]] = [
            // Stage 5 (ThetaUpblockA): needs GammaB's skip_down2_1 as skip_0
            5: ["skip_down2_1": "skip_0"],
            // Stage 6 (ThetaUpblockB): needs GammaA's skip_down2_0 as skip_0
            6: ["skip_down2_0": "skip_0"],
            // Stage 7 (ThetaUpblockC): needs AlphaB's skip_down1_2 as skip_0
            7: ["skip_down1_2": "skip_0"],
            // Stage 8 (LambdaUpblock): needs AlphaB's skip_down1_1, skip_down1_0, skip_down0_2
            8: ["skip_down1_1": "skip_0", "skip_down1_0": "skip_1", "skip_down0_2": "skip_2"],
            // Stage 9 (KappaUpblock): needs AlphaB's skip_down0_1, skip_down0_0, skip_conv_in
            9: ["skip_down0_1": "skip_0", "skip_down0_0": "skip_1", "skip_conv_in": "skip_2"],
        ]

        // Dependencies:
        // Stage 0 (AlphaA): base inputs → emb_out
        // Stage 1 (AlphaB): needs [0] emb → ip_hidden_states_out, skip_conv_in, skip_down0_*, skip_down1_*, hidden_out
        // Stage 2 (GammaDownblockA): needs [1] hidden, [0] emb → skip_down2_0, hidden_out
        // Stage 3 (GammaDownblockB): needs [2] hidden, [0] emb → skip_down2_1, hidden_out
        // Stage 4 (SigmaCore): needs [3] hidden, [0] emb → hidden_out
        // Stage 5 (ThetaUpblockA): needs [4] hidden, [0] emb, [3] skip_down2_1 → hidden_out
        // Stage 6 (ThetaUpblockB): needs [5] hidden, [0] emb, [2] skip_down2_0 → hidden_out
        // Stage 7 (ThetaUpblockC): needs [6] hidden, [0] emb, [1] skip_down1_2 → hidden_out (upsampled)
        // Stage 8 (LambdaUpblock): needs [7] hidden, [0] emb, [1] skips → hidden_out
        // Stage 9 (KappaUpblock): needs [8] hidden, [0] emb, [1] skips → hidden_out
        // Stage 10 (OmegaDecoder): needs [9] hidden, [1] skip_conv_in → noise_pred
        let dependencies: [[Int]] = [
            [],              // 0: AlphaEncoderA
            [0],             // 1: AlphaEncoderB
            [0, 1],          // 2: GammaDownblockA
            [0, 1, 2],       // 3: GammaDownblockB
            [0, 1, 3],       // 4: SigmaCore
            [0, 1, 3, 4],    // 5: ThetaUpblockA (needs GammaB's skip_down2_1)
            [0, 1, 2, 5],    // 6: ThetaUpblockB (needs GammaA's skip_down2_0)
            [0, 1, 6],       // 7: ThetaUpblockC (needs AlphaB's skip_down1_2)
            [0, 1, 7],       // 8: LambdaUpblock
            [0, 1, 8],       // 9: KappaUpblock
            [0, 1, 9]        // 10: OmegaDecoder
        ]

        let inputs = batch.arrayOfFeatureValueDictionaries
        var stageOutputs: [[[String: MLFeatureValue]]] = []

        // --- Stage 0 (AlphaEncoderA) ---
        var results = try signposter.withIntervalSignpost("UNet Stage", "AlphaEncoderA") {
            try self.first!.perform { model in
                try model.predictions(fromBatch: batch)
            }
        }

        // Unload model to free ANE/GPU memory (outputs are retained in stageOutputs)
        if reduceMemory {
            self.first!.unloadResources()
        }
        stageOutputs.append(results.arrayOfFeatureValueDictionaries)

        // --- Remaining stages ---
        for (stageIndex, stage) in self.dropFirst().enumerated() {
            try autoreleasepool {
                let actualIndex = stageIndex + 1

                let skipMap = stageSkipMappings[actualIndex] ?? [:]

                results = try signposter.withIntervalSignpost("UNet Stage", "\(actualIndex)/11") {
                    let mergedInputDicts: [[String: MLFeatureValue]] =
                        (0..<inputs.count).map { sampleIndex in
                            var merged = inputs[sampleIndex]

                            for dep in dependencies[actualIndex] {
                                let depDict = stageOutputs[dep][sampleIndex]
                                for (outputName, value) in depDict {
                                    if let inputName = skipMap[outputName] {
                                        merged[inputName] = value
                                    }
                                    else if let inputName = outputToInputNameMap[outputName] {
                                        merged[inputName] = value
                                    }
                                    else {
                                        merged[outputName] = value
                                    }
                                }
                            }
                            return merged
                        }

                    let providers = try mergedInputDicts.map {
                        try MLDictionaryFeatureProvider(dictionary: $0)
                    }
                    let nextBatch = MLArrayBatchProvider(array: providers)

                    return try stage.perform { model in
                        try model.predictions(fromBatch: nextBatch)
                    }
                }

                // Unload model to free ANE/GPU memory (outputs are retained in stageOutputs)
                if reduceMemory {
                    stage.unloadResources()
                }
                stageOutputs.append(results.arrayOfFeatureValueDictionaries)
            }
        }

        return results
    }
}

@available(iOS 17.4, macOS 14.4, *)
extension Unet: ComputePlanProviding, ManagedModelProviding {
    public var managedModels: [ManagedMLModel] { self.models }
}

