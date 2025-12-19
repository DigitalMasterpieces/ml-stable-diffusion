// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

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
        if models.count == 1 {
            return [70]
        } else if models.count == 2 {
            return [35, 35]
        } else if models.count == 4 {
            return [15, 20, 20, 15]
        } else {
            // Unsupported
            assert(false)
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
        try! models.first!.perform { model in
            model.modelDescription.inputDescriptionsByName["sample"]!
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
        ipAdapterScale: Float? = 1.0
    ) throws -> [MLShapedArray<Float32>] {

        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray<Float32>(scalars:[Float(timeStep), Float(timeStep)],shape:[2])

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
        var results = try self.first!.perform { model in
            try model.predictions(fromBatch: batch)
        }
        stageOutputs.append(results.arrayOfFeatureValueDictionaries)

        // --- Remaining stages ---
        for (stageIndex, stage) in self.dropFirst().enumerated() {
            let actualIndex = stageIndex + 1   // because dropFirst shifts indices

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
            results = try stage.perform { model in
                try model.predictions(fromBatch: nextBatch)
            }
            stageOutputs.append(results.arrayOfFeatureValueDictionaries)
        }

        return results
    }
}

