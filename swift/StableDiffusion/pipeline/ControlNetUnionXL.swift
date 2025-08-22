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
    
    /// Load resources.
    public func loadResources() throws {
        for model in models {
            try model.loadResources()
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
        for model in models {
            model.unloadResources()
        }
    }
    
    /// Pre-warm resources
    public func prewarmResources() throws {
        // Override default to pre-warm each model
        for model in models {
            try model.loadResources()
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
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>,
        pooledStates: MLShapedArray<Float32>,
        geometryConditioning: MLShapedArray<Float32>,
        conditioningScale: Float,
        controlTypes: MLShapedArray<Float32>?,
        images: [MLShapedArray<Float32>]
    ) throws -> [[String: MLShapedArray<Float32>]] {
        // Match time step batch dimension to the model / latent samples
        let t = MLShapedArray(scalars: [Float(timeStep), Float(timeStep)], shape: [2])

        // Initialize tensor for conditioning scale
        let conditioningScaleArray = MLShapedArray<Float32>(scalars: [conditioningScale], shape: [1])
        guard let controlTypeArray = controlTypes else {
            return []
        }
        
        var outputs: [[String: MLShapedArray<Float32>]] = []
        
        for (modelIndex, model) in models.enumerated() {
            let inputs = try latents.map { latent in
                let dict: [String: Any] = [
                    "sample": MLMultiArray(latent),
                    "timestep": MLMultiArray(t),
                    "encoder_hidden_states": MLMultiArray(hiddenStates),
                    "controlnet_cond_3": MLMultiArray(images[modelIndex]),
                    "control_type": MLMultiArray(controlTypeArray)
                ]
                return try MLDictionaryFeatureProvider(dictionary: dict)
            }

            let batch = MLArrayBatchProvider(array: inputs)
            
            let results = try model.perform {
                try $0.predictions(fromBatch: batch)
            }
            
            // pre-allocate MLShapedArray with a specific shape in outputs
            if outputs.isEmpty {
                outputs = initOutputs(
                    batch: latents.count,
                    shapes: results.features(at: 0).featureValueDictionary
                )
            }
            
            for n in 0..<results.count {
                let result = results.features(at: n)
                for k in result.featureNames {
                    let newValue = result.featureValue(for: k)!.multiArrayValue!
                    let count = newValue.count
                    let inputPointer = newValue.dataPointer.assumingMemoryBound(to: Float.self)

                    if modelIndex == 0 {
                        // scale first input directly into outputs
                        var scaled = [Float](repeating: 0, count: count)
                        vDSP_vsmul(inputPointer, 1,
                                    [conditioningScale], &scaled, 1,
                                    vDSP_Length(count))
                        let scaledArray = try! MLMultiArray(shape: newValue.shape, dataType: .float32)
                        let scaledPointer = scaledArray.dataPointer.assumingMemoryBound(to: Float.self)
                        scaled.withUnsafeBufferPointer { buf in
                            scaledPointer.assign(from: buf.baseAddress!, count: count)
                        }
                        outputs[n][k] = MLShapedArray<Float32>(scaledArray)
                    } else {
                        // accumulate scaled values into outputs
                        let outputArray = MLMultiArray(outputs[n][k]!)
                        let outputPointer = outputArray.dataPointer.assumingMemoryBound(to: Float.self)
                        vDSP_vsma(inputPointer, 1, [conditioningScale],
                                    outputPointer, 1,
                                    outputPointer, 1,
                                    vDSP_Length(count))
                                    // `vsma` does: output = input * scale + output
                    }
                }
            }
        }
        
        return outputs
    }
    
    private func initOutputs(batch: Int, shapes: [String: MLFeatureValue]) -> [[String: MLShapedArray<Float32>]] {
        var output: [String: MLShapedArray<Float32>] = [:]
        for (outputName, featureValue) in shapes {
            output[outputName] = MLShapedArray<Float32>(
                repeating: 0.0,
                shape: featureValue.multiArrayValue!.shape.map { $0.intValue }
            )
        }
        return Array(repeating: output, count: batch)
    }
}
