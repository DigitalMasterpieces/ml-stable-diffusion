// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import CoreML
import CryptoKit

/// A class to manage and gate access to a Core ML model
///
/// It will automatically load a model into memory when needed or requested
/// It allows one to request to unload the model from memory
@available(iOS 16.2, macOS 13.1, *)
public final class ManagedMLModel: ResourceManaging {

    /// The location of the model
    var modelURL: URL

    /// The configuration to be used when the model is loaded
    var configuration: MLModelConfiguration

    /// The loaded model (when loaded)
    var loadedModel: MLModel?

    /// Queue to protect access to loaded model
    var queue: DispatchQueue

    /// Create a managed model given its location and desired loaded configuration
    ///
    /// - Parameters:
    ///     - url: The location of the model
    ///     - configuration: The configuration to be used when the model is loaded/used
    /// - Returns: A managed model that has not been loaded
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.modelURL = url
        self.configuration = configuration
        self.loadedModel = nil
        self.queue = DispatchQueue(label: "managed.\(url.lastPathComponent)")
    }

    /// Instantiation and load model into memory
    public func loadResources() throws {
        try queue.sync {
            try loadModel()
        }
    }

    /// Unload the model if it was loaded
    public func unloadResources() {
        queue.sync {
            loadedModel = nil
        }
    }

    /// Perform an operation with the managed model via a supplied closure.
    ///  The model will be loaded and supplied to the closure and should only be
    ///  used within the closure to ensure all resource management is synchronized
    ///
    /// - Parameters:
    ///     - body: Closure which performs and action on a loaded model
    /// - Returns: The result of the closure
    /// - Throws: An error if the model cannot be loaded or if the closure throws
    public func perform<R>(_ body: (MLModel) throws -> R) throws -> R {
        return try queue.sync {
            try autoreleasepool {
                try loadModel()
                return try body(loadedModel!)
            }
        }
    }

    /// Compute hash for a model at `modelURL`.
    func cacheKey(for modelURL: URL) -> String {
        let data = try! Data(contentsOf: modelURL)
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return modelURL.deletingPathExtension().lastPathComponent + "_" + hash
    }

    /// Compute hash for a packaged model at `url`.
    func cacheDirectoryKey(for url: URL) -> String {
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: nil)!

        var hasher = SHA256()

        for case let fileURL as URL in enumerator {
            if (try? fileURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                continue
            }
            if let data = try? Data(contentsOf: fileURL) {
                hasher.update(data: data)
            }
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func loadModel() throws {
        if loadedModel != nil { return }

        let fm = FileManager.default

        // Create persistent cache folder.
        let supportDir = try fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let cacheDir = supportDir.appendingPathComponent("CompiledModels", isDirectory: true)
        if !fm.fileExists(atPath: cacheDir.path) {
            try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        }

        // Extract base name (e.g., "TextEncoder2").
        let baseName = modelURL.deletingPathExtension().lastPathComponent

        // Compute a hash so updated bundled models force a fresh compile.
        let hash: String
        if modelURL.pathExtension == "mlpackage" {
            hash = self.cacheDirectoryKey(for: modelURL)
        } else {
            hash = self.cacheKey(for: modelURL)
        }

        // Cache filename <BaseName>_<hash>.mlmodelc.
        let cacheName = "\(baseName)_\(hash).mlmodelc"
        let cachedModelURL = cacheDir.appendingPathComponent(cacheName)

        // Cleanup: remove any old cached versions for this base model.
        let cacheContents = try fm.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil)
        for url in cacheContents {
            if url.lastPathComponent.hasPrefix(baseName + "_") &&
                url.lastPathComponent != cacheName {
                try? fm.removeItem(at: url)
            }
        }

        // If cache missing → compile or copy.
        if !fm.fileExists(atPath: cachedModelURL.path) {

            if modelURL.pathExtension == "mlpackage" {
                // Compile .mlpackage → .mlmodelc.
                let compiled = try MLModel.compileModel(at: modelURL)
                try fm.copyItem(at: compiled, to: cachedModelURL)

            } else if modelURL.pathExtension == "mlmodelc" {
                // Already compiled → copy.
                try fm.copyItem(at: modelURL, to: cachedModelURL)

            } else if modelURL.pathExtension == "mlmodel" {
                // Raw → compile then cache.
                let compiled = try MLModel.compileModel(at: modelURL)
                try fm.copyItem(at: compiled, to: cachedModelURL)

            } else {
                throw NSError(
                    domain: "ModelLoader",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Unsupported model type: \(modelURL)"]
                )
            }
        }

        // Load the cached compiled model.
        loadedModel = try MLModel(contentsOf: cachedModelURL, configuration: configuration)
    }
}

@available(iOS 16.2, macOS 13.1, *)
public extension Array where Element == ManagedMLModel {
    /// Performs batch predictions using an array of `[ManagedMLModel]` instances in a pipeline.
    /// - Parameter batch: Inputs for btached predictions.
    /// - Returns: Final prediction results after processing through all models.
    /// - Throws: Errors if the array is empty, predictions fail, or results can't be combined.
    func predictions(from batch: MLBatchProvider) throws -> MLBatchProvider {
        var results = try self.first!.perform { model in
            try model.predictions(fromBatch: batch)
        }

        if self.count == 1 {
            return results
        }

        // Manual pipeline batch prediction
        let inputs = batch.arrayOfFeatureValueDictionaries
        for stage in self.dropFirst() {
            // Combine the original inputs with the outputs of the last stage
            let next = try results.arrayOfFeatureValueDictionaries
                .enumerated().map { index, dict in
                    let nextDict = dict.merging(inputs[index]) { out, _ in out }
                    return try MLDictionaryFeatureProvider(dictionary: nextDict)
                }
            let nextBatch = MLArrayBatchProvider(array: next)

            // Predict
            results = try stage.perform { model in
                try model.predictions(fromBatch: nextBatch)
            }
        }

        return results
    }
}

extension MLFeatureProvider {
    var featureValueDictionary: [String : MLFeatureValue] {
        self.featureNames.reduce(into: [String : MLFeatureValue]()) { result, name in
            result[name] = self.featureValue(for: name)
        }
    }
}

extension MLBatchProvider {
    var arrayOfFeatureValueDictionaries: [[String : MLFeatureValue]] {
        (0..<self.count).map {
            self.features(at: $0).featureValueDictionary
        }
    }
}
