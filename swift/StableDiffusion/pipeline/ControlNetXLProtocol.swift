// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

@available(iOS 16.2, macOS 13.1, *)
public protocol ControlNetXLProtocol {
    var models: [ManagedMLModel] { get }

    func loadResources() throws

    /// Unload the underlying model to free up memory
    func unloadResources()

    /// Pre-warm resources
    func prewarmResources() throws

    var inputImageDescriptions: [MLFeatureDescription] { get }

    var inputImageShapes: [[Int]] { get }

    func execute(
        latents: [MLShapedArray<Float32>],
        timeStep: Int,
        hiddenStates: MLShapedArray<Float32>,
        pooledStates: MLShapedArray<Float32>,
        geometryConditioning: MLShapedArray<Float32>,
        conditioningScale: Float,
        controlTypes: MLShapedArray<Float32>?,
        images: [MLShapedArray<Float32>]
    ) throws -> [[String: MLShapedArray<Float32>]]
}
