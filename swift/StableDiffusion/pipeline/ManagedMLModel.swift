// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

@preconcurrency import CoreML
import os
import Synchronization

/// Manages loading and access to a single Core ML model.
///
/// The caller is responsible for compiling model sources (`.mlpackage` / `.mlmodel`) to
/// `.mlmodelc` bundles before passing the URL to this class. `ManagedMLModel` loads
/// compiled models directly via `MLModel(contentsOf:)` and does not perform any
/// compilation or caching itself.
///
/// Thread safety is provided by `Mutex`, which protects all access to the loaded model.
/// This keeps the API synchronous — important because CoreML predictions (`MLModel.prediction(from:)`)
/// are sync-only and can block for 100-280ms per call.
@available(iOS 16.2, macOS 13.1, *)
public final class ManagedMLModel: Sendable {

    /// The location of the model
    let modelURL: URL

    /// The configuration to be used when the model is loaded
    let configuration: MLModelConfiguration

    /// The loaded model state, protected by a Mutex.
    private let state: Mutex<MLModel?>

    /// Create a managed model given its location and desired loaded configuration
    ///
    /// - Parameters:
    ///     - url: URL of a compiled `.mlmodelc` bundle (or a symlink to one).
    ///       Source formats like `.mlpackage` are not supported — compile them first.
    ///     - configuration: The configuration to be used when the model is loaded/used
    /// - Returns: A managed model that has not been loaded
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.modelURL = url
        // Defensive copy: MLModelConfiguration is a reference type, so the caller could
        // mutate it after init. Copying ensures our configuration is stable.
        self.configuration = configuration.copy() as! MLModelConfiguration
        self.state = Mutex(nil)
    }

    /// Instantiation and load model into memory
    public func loadResources(progress: Progress) throws {
        try state.withLock { loadedModel in
            if loadedModel != nil { return }

            let loadState = signposter.beginInterval(
                "Load Model",
                "\(self.modelURL.lastPathComponent, privacy: .public)"
            )
            defer { signposter.endInterval("Load Model", loadState) }

            let resolvedURL = self.modelURL.resolvingSymlinksInPath()
            loadedModel = try MLModel(contentsOf: resolvedURL, configuration: self.configuration)
        }
    }

    /// Unload the model if it was loaded
    public func unloadResources() {
        state.withLock { loadedModel in
            loadedModel = nil
            signposter.emitEvent("Unload Model", "\(self.modelURL.lastPathComponent, privacy: .public)")
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
        // Drain CoreML ObjC intermediates after each prediction to limit peak memory
        // during tight denoising loops. The pool wraps the lock rather than nesting inside
        // it, because Mutex.withLock's task-isolated closure prevents nested closures in
        // Swift 6 strict concurrency.
        try autoreleasepool {
            try state.withLock { loadedModel in
                if loadedModel == nil {
                    let loadState = signposter.beginInterval(
                        "Load Model",
                        "\(self.modelURL.lastPathComponent, privacy: .public)"
                    )
                    defer { signposter.endInterval("Load Model", loadState) }

                    let resolvedURL = self.modelURL.resolvingSymlinksInPath()
                    loadedModel = try MLModel(contentsOf: resolvedURL, configuration: self.configuration)
                }
                return try body(loadedModel!)
            }
        }
    }

    // MARK: - Compute Plan

    /// Loads the compute plan for this model using its URL and configuration.
    ///
    /// Requires `loadResources` or `perform` to have been called first so that the model
    /// is available.
    @available(iOS 17.4, macOS 14.4, *)
    public var computePlan: MLComputePlan {
        get async throws {
            try await MLComputePlan.load(contentsOf: self.modelURL, configuration: self.configuration)
        }
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
