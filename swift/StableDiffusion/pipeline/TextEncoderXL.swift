// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Foundation
import CoreML
import os

@available(iOS 17.0, macOS 14.0, *)
public protocol TextEncoderXLModel: ResourceManaging {
    typealias TextEncoderXLOutput = (hiddenEmbeddings: MLShapedArray<Float32>, pooledOutputs: MLShapedArray<Float32>)

    /// The model's expected input sequence length (total tokens including start/end/pad).
    var inputLength: Int { get }

    func encode(_ text: String) throws -> TextEncoderXLOutput
}

///  A model for encoding text, suitable for SDXL
@available(iOS 17.0, macOS 14.0, *)
public struct TextEncoderXL: TextEncoderXLModel {

    /// Text tokenizer
    public var tokenizer: BPETokenizer

    /// Embedding model
    var model: ManagedMLModel

    /// Fixed input sequence length for CLIP-based text encoders (ViT-L/14 and OpenCLIP ViT-G/14).
    ///
    /// This is the total token count including the start-of-text and end-of-text markers plus
    /// padding. The effective content-token budget is `inputLength - 2`.
    public let inputLength: Int = 77

    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - url: Location of compiled text encoding  Core ML model
    ///   - configuration: configuration to be used when the model is loaded
    /// - Returns: A text encoder that will lazily load its required resources when needed or requested
    public init(tokenizer: BPETokenizer,
                modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.tokenizer = tokenizer
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    public var loadProgressWeights: [Int64] {
        return [15]
    }

    public func makeLoadProgress() -> Progress {
        let progress = Progress(totalUnitCount: self.loadProgressWeights.first!)
        progress.localizedDescription = "Text Encoder"
        return progress
    }

    /// Ensure the model has been loaded into memory
    public func loadResources(progress: Progress, prewarm: Bool) throws {
        try model.loadResources(progress: progress)
        if prewarm {
            self.unloadResources()
        }
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
       model.unloadResources()
    }

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    public func encode(_ text: String) throws -> TextEncoderXLOutput {
        let encodeState = signposter.beginInterval("Encode Text")
        defer { signposter.endInterval("Encode Text", encodeState) }

        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize with prompt weights, padding to the expected length
        var (tokens, ids, weights) = signposter.withIntervalSignpost("Tokenize") {
            tokenizer.tokenizeWithWeights(text, minCount: inputLength)
        }

        // Truncate if necessary
        if ids.count > inputLength {
            tokens = tokens.dropLast(tokens.count - inputLength)
            ids = ids.dropLast(ids.count - inputLength)
            weights = weights.dropLast(weights.count - inputLength)
            let truncated = tokenizer.decode(tokens: tokens)
            print("Needed to truncate input '\(text)' to '\(truncated)'")
        }

        // Use the model to generate the embedding
        return try encode(ids: ids, weights: weights)
    }

    func encode(ids: [Int], weights: [Float]? = nil) throws -> TextEncoderXLOutput {
        let inputName = inputDescription.name
        let inputShape = inputShape

        let floatIds = ids.map { Float32($0) }
        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
        let inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: [inputName: MLMultiArray(inputArray)])

        // Run CoreML model inference.
        let result = try signposter.withIntervalSignpost("Text Encoder Predict") {
            try model.perform { model in
                try model.prediction(from: inputFeatures)
            }
        }

        let embeddingFeature = result.featureValue(for: "hidden_embeds")
        let pooledFeature = result.featureValue(for: "pooled_outputs")
        let pooledOutputs = MLShapedArray<Float32>(converting: pooledFeature!.multiArrayValue!)
        var hiddenEmbeddings = MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!)

        // Apply prompt weights to hidden embeddings (not pooled — pooled has no token dimension)
        if let weights {
            let weightsState = signposter.beginInterval("Apply Prompt Weights")

            let shape = hiddenEmbeddings.shape  // [1, 77, dim]
            let previousMean = hiddenEmbeddings.scalars.withUnsafeBufferPointer { buffer in
                buffer.reduce(0, +)
            } / Float(hiddenEmbeddings.scalars.count)

            // Multiply each token position's embedding vector by its weight
            let newEmbeddings = weights.enumerated().map { index, weight in
                MLShapedArray<Float32>(
                    scalars: hiddenEmbeddings[0][index].scalars.map { Float32($0 * weight) },
                    shape: hiddenEmbeddings[0][index].shape
                )
            }
            hiddenEmbeddings = MLShapedArray<Float32>(concatenating: newEmbeddings, alongAxis: 0)

            let currentMean = hiddenEmbeddings.scalars.withUnsafeBufferPointer { buffer in
                buffer.reduce(0, +)
            } / Float(hiddenEmbeddings.scalars.count)

            // Mean-factor normalization to preserve overall embedding magnitude
            if currentMean != 0 {
                let meanFactor = Float32(previousMean / currentMean)
                hiddenEmbeddings = MLShapedArray<Float32>(unsafeUninitializedShape: shape) { scalars, _ in
                    hiddenEmbeddings.withUnsafeShapedBufferPointer { embeddings, _, _ in
                        for i in 0..<embeddings.count {
                            scalars.initializeElement(at: i, to: embeddings[i] * meanFactor)
                        }
                    }
                }
            }

            signposter.endInterval("Apply Prompt Weights", weightsState)
        }

        return (hiddenEmbeddings, pooledOutputs)
    }

    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.value
        }
    }

    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}

@available(iOS 17.4, macOS 14.4, *)
extension TextEncoderXL: ComputePlanProviding, ManagedModelProviding {
    public var managedModels: [ManagedMLModel] { [self.model] }
}
