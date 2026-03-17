// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

@available(iOS 16.2, macOS 13.1, *)
public protocol TextEncoderModel: ResourceManaging {

    func encode(_ text: String) throws -> MLShapedArray<Float32>
}

///  A model for encoding text
@available(iOS 16.2, macOS 13.1, *)
public struct TextEncoder: TextEncoderModel {

    /// Text tokenizer
    var tokenizer: BPETokenizer

    /// Embedding model
    var model: ManagedMLModel

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
        return [10]
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
    public func encode(_ text: String) throws -> MLShapedArray<Float32> {

        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize with prompt weights, padding to the expected length
        var (tokens, ids, weights) = tokenizer.tokenizeWithWeights(text, minCount: inputLength)

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

    /// Prediction queue
    let queue = DispatchQueue(label: "textencoder.predict")

    func encode(ids: [Int], weights: [Float]? = nil) throws -> MLShapedArray<Float32> {
        let inputName = inputDescription.name
        let inputShape = inputShape

        let floatIds = ids.map { Float32($0) }
        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
        let inputFeatures = try! MLDictionaryFeatureProvider(
            dictionary: [inputName: MLMultiArray(inputArray)])

        let result = try model.perform { model in
            try model.prediction(from: inputFeatures)
        }

        let embeddingFeature = result.featureValue(for: "last_hidden_state")
        var textEmbeddings = MLShapedArray<Float32>(converting: embeddingFeature!.multiArrayValue!)

        // Apply prompt weights to embeddings
        if let weights {
            let shape = textEmbeddings.shape  // [1, 77, dim]
            let previousMean = textEmbeddings.scalars.withUnsafeBufferPointer { buffer in
                buffer.reduce(0, +)
            } / Float(textEmbeddings.scalars.count)

            // Multiply each token position's embedding vector by its weight
            let newEmbeddings = weights.enumerated().map { index, weight in
                MLShapedArray<Float32>(
                    scalars: textEmbeddings[0][index].scalars.map { Float32($0 * weight) },
                    shape: textEmbeddings[0][index].shape
                )
            }
            textEmbeddings = MLShapedArray<Float32>(concatenating: newEmbeddings, alongAxis: 0)

            let currentMean = textEmbeddings.scalars.withUnsafeBufferPointer { buffer in
                buffer.reduce(0, +)
            } / Float(textEmbeddings.scalars.count)

            // Mean-factor normalization to preserve overall embedding magnitude
            if currentMean != 0 {
                let meanFactor = Float32(previousMean / currentMean)
                textEmbeddings = MLShapedArray<Float32>(unsafeUninitializedShape: shape) { scalars, _ in
                    textEmbeddings.withUnsafeShapedBufferPointer { embeddings, _, _ in
                        for i in 0..<embeddings.count {
                            scalars.initializeElement(at: i, to: embeddings[i] * meanFactor)
                        }
                    }
                }
            }
        }

        return textEmbeddings
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
extension TextEncoder: ComputePlanProviding, ManagedModelProviding {
    public var managedModels: [ManagedMLModel] { [self.model] }
}
