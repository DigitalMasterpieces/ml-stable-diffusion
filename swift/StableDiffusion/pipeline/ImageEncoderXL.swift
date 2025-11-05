// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

@available(iOS 17.0, macOS 14.0, *)
public protocol ImageEncoderXLModel: ResourceManaging {
    typealias ImageEncoderXLOutput = MLShapedArray<Float32>
    func encode(_ image: CGImage) throws -> ImageEncoderXLOutput
    var imageEmbedsShape: [Int] { get }
}

///  A model for encoding text, suitable for SDXL
@available(iOS 17.0, macOS 14.0, *)
public struct ImageEncoderXL: ImageEncoderXLModel {

    public enum Error: String, Swift.Error {
        case sampleInputShapeNotCorrect
    }

    /// Embedding model
    var model: ManagedMLModel

    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - url: Location of compiled text encoding  Core ML model
    ///   - configuration: configuration to be used when the model is loaded
    /// - Returns: A text encoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL,
                configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    /// Ensure the model has been loaded into memory
    public func loadResources() throws {
        try model.loadResources()
    }

    /// Unload the underlying model to free up memory
    public func unloadResources() {
       model.unloadResources()
    }

    var imageEmbedsDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.outputDescriptionsByName["image_embeds"]!
        }
    }

    /// The expected shape of the image embeddings.
    public var imageEmbedsShape: [Int] {
        imageEmbedsDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    public func encode(_ image: CGImage) throws -> ImageEncoderXLOutput {
        let imageData = try image.planarRGBShapedArray(minValue: 0.0, maxValue: 1.0)
        guard imageData.shape == self.inputShape else {
            // TODO: Consider auto resizing and croping similar to how Vision or CoreML auto-generated Swift code can accomplish with `MLFeatureValue`
            throw Error.sampleInputShapeNotCorrect
        }
        let dict = [self.inputName: MLMultiArray(imageData)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)

        let result = try model.perform { model in
            try model.prediction(from: input)
        }

        let outputName = result.featureNames.first!
        let embeddingFeature = result.featureValue(for: outputName)!.multiArrayValue!
        return MLShapedArray<Float32>(converting: embeddingFeature)
    }

    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.value
        }
    }

    var inputName: String {
        inputDescription.name
    }

    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
