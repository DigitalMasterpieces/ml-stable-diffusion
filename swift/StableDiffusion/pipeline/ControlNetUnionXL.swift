// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import Accelerate

@available(iOS 16.2, macOS 13.1, *)
public struct ControlNetUnionXL: ResourceManaging, ControlNetXLProtocol {

    public var models: [ManagedMLModel]
    
    public init(modelAt urls: [URL],
                configuration: MLModelConfiguration) {
        self.models = urls.map { ManagedMLModel(modelAt: $0, configuration: configuration) }
    }

    public var loadProgressWeights: [Int64] {
        return [20]
    }

    public func makeLoadProgress() -> Progress {
        let progress = Progress(totalUnitCount: self.loadProgressWeights.first!)
        progress.localizedDescription = "Control Net (Union)"
        return progress
    }

    /// Load resources.
    public func loadResources(progress: Progress, prewarm: Bool) throws {
        for model in models {
            try model.loadResources(progress: progress)
            if prewarm {
                model.unloadResources()
            }
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }

    public var inputImageDescriptions: [MLFeatureDescription] {
        models.flatMap { model in
            try! model.perform {
                $0.modelDescription.inputDescriptionsByName
                    .filter { $0.key.hasPrefix("controlnet_cond_") }
                    .map { $0.value }
            }
        }
    }

    /// The expected shape of the models image input
    public var inputImageShapes: [[Int]] {
        inputImageDescriptions.map { desc in
            desc.multiArrayConstraint!.shape.map { $0.intValue }
        }
    }

    public var outputDescriptions: [[String : MLFeatureDescription]] {
        models.map { model in
            try! model.perform {
                $0.modelDescription.outputDescriptionsByName
            }
        }
    }

    /// The expected shape of the models outputs
    public var outputShapes: [[String: [Int]]] {
        outputDescriptions.map { desc in
            desc.mapValues { $0.multiArrayConstraint!.shape.map { $0.intValue } }
        }
    }

    /// Calculate additional inputs for Unet to generate intended image following provided images
    ///
    /// - Parameters:
    ///   - latents: Batch of latent samples in an array
    ///   - timeStep: Current diffusion timestep
    ///   - hiddenStates: Hidden state to condition on
    ///   - pooledStates: Additional text states to condition on
    ///   - geometryConditioning: Condition on image geometry
    ///   - conditioningScale: Conditioning scale
    ///   - controlTypes: Tensor with values of `0` or `1` depending on whether the control type is used.
    ///   - images: Images for each ControlNet
    /// - Returns: Array of predicted noise residuals
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
        // Match time step batch dimension to the model / latent samples
        // Infer batch size from hiddenStates shape (batch size 2 for CFG, 1 for no CFG)
        let batchSize = hiddenStates.shape[0]
        let t: MLShapedArray<Float32>
        if batchSize == 2 {
            t = MLShapedArray(scalars: [Float(timeStep), Float(timeStep)], shape: [2])
        } else {
            t = MLShapedArray(scalars: [Float(timeStep)], shape: [1])
        }

        var outputs: [[String: MLShapedArray<Float32>]] = Array(repeating: [:], count: latents.count)

        for (modelIndex, model) in models.enumerated() {
            // Run prediction for every control type that is activated.
            for (controlID, controlType) in controlTypes[modelIndex].enumerated() {
                let controlImage = images[modelIndex][controlID]
                let conditioningScale = conditioningScales[modelIndex][controlID]

                // Early out if image is not set or the conditioning scale is zero.
                guard let controlImage = controlImage, conditioningScale > 0.0 else {
                    continue
                }

                // Setup the control net type mask for the specific control ID
                let controlNetTotalIDs = conditioningScales[modelIndex].count
                let controlTypeIDMask: [UInt] = (0..<controlNetTotalIDs).map { $0 == controlID ? 1 : 0 }
                let controlTypeIDMaskArray: MLShapedArray<Float32>
                if batchSize == 2 {
                    controlTypeIDMaskArray = MLShapedArray<Float32>(
                        scalars: Array(repeating: controlTypeIDMask.map { Float32($0) }, count: 2).flatMap { $0 },
                        shape: [2, controlTypeIDMask.count]
                    )
                } else {
                    controlTypeIDMaskArray = MLShapedArray<Float32>(
                        scalars: controlTypeIDMask.map { Float32($0) },
                        shape: [1, controlTypeIDMask.count]
                    )
                }

                let conditioningScalesArray =
                    MLShapedArray<Float32>(
                        scalars: conditioningScales[modelIndex],
                        shape: [controlNetTotalIDs]
                    )

                let inputs = try latents.map { latent in
                    var dict: [String: MLMultiArray] = [
                        "sample": MLMultiArray(latent),
                        "timestep": MLMultiArray(t),
                        "encoder_hidden_states": MLMultiArray(hiddenStates),
                        "control_type": MLMultiArray(controlTypeIDMaskArray),
                        "controlnet_cond_\(controlID)": MLMultiArray(controlImage),
                        "conditioning_scale": MLMultiArray(conditioningScalesArray)
                    ]

                    return try MLDictionaryFeatureProvider(dictionary: dict)
                }

                let batch = MLArrayBatchProvider(array: inputs)

                let results = try model.perform {
                    try $0.predictions(fromBatch: batch)
                }

                for n in 0..<results.count {
                    let result = results.features(at: n)
                    for k in result.featureNames {
                        let newValue = result.featureValue(for: k)!.multiArrayValue!

                        let count = newValue.count
                        let inputPointer = newValue.dataPointer.assumingMemoryBound(to: Float.self)

                        // Accumulate scaled values into outputs.
                        let scaledArray = try! MLMultiArray(shape: newValue.shape, dataType: .float32)
                        let scaledPointer = scaledArray.dataPointer.assumingMemoryBound(to: Float.self)

                        // Direct scaling into MLMultiArray memory.
                        vDSP_vsmul(inputPointer, 1,
                                   [1.0], // conditioningScale
                                    scaledPointer, 1,
                                    vDSP_Length(count))

                        outputs[n][k] = MLShapedArray<Float32>(scaledArray)
                    }
                }
            }
        }

        // Initialize missing outputs.
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

