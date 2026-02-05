// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation
import CoreML

/// A decoder model which produces RGB images from latent samples
@available(iOS 16.2, macOS 13.1, *)
public struct Decoder: ResourceManaging {

    /// VAE decoder model
    var model: ManagedMLModel

    /// Create decoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE decoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: A decoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    public var loadProgressWeights: [Int64] {
        return [20]
    }

    public func makeLoadProgress() -> Progress {
        let progress = Progress(totalUnitCount: self.loadProgressWeights.first!)
        progress.localizedDescription = "Decoder"
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

    /// Batch decode latent samples into images
    ///
    ///  - Parameters:
    ///    - latents: Batch of latent samples to decode
    ///    - scaleFactor: scalar divisor on latents before decoding
    ///  - Returns: decoded images
    public func decode(
        _ latents: [MLShapedArray<Float32>],
        scaleFactor: Float32,
        shiftFactor: Float32 = 0.0
    ) throws -> [CGImage] {

        // Form batch inputs for model
        let inputs: [MLFeatureProvider] = try latents.map { sample in
            // Reference pipeline scales the latent samples before decoding
            let sampleScaled = MLShapedArray<Float32>(
                scalars: sample.scalars.map { $0 / scaleFactor + shiftFactor },
                shape: sample.shape)

            let dict = [inputName: MLMultiArray(sampleScaled)]
            return try MLDictionaryFeatureProvider(dictionary: dict)
        }
        let batch = MLArrayBatchProvider(array: inputs)

        // Batch predict with model
        let results = try model.perform { model in
            try model.predictions(fromBatch: batch)
        }

        // Transform the outputs to CGImages
        let images: [CGImage] = try (0..<results.count).map { i in
            let result = results.features(at: i)
            let outputName = result.featureNames.first!
            let output = result.featureValue(for: outputName)!.multiArrayValue!
            return try CGImage.fromShapedArray(MLShapedArray<Float32>(converting: output))
        }

        return images
    }

    /// Batch decode latent samples into images using tiled processing
    ///
    /// This method processes latents in tiles to reduce memory usage,
    /// enabling Neural Engine execution on devices with memory constraints.
    ///
    ///  - Parameters:
    ///    - latents: Batch of latent samples to decode
    ///    - scaleFactor: scalar divisor on latents before decoding
    ///    - shiftFactor: shift factor for latent preprocessing
    ///    - tilingConfig: Configuration for tile-based processing
    ///  - Returns: decoded images
    public func decodeTiled(
        _ latents: [MLShapedArray<Float32>],
        scaleFactor: Float32,
        shiftFactor: Float32 = 0.0,
        tilingConfig: TilingConfiguration
    ) throws -> [CGImage] {
        // If tiling disabled, use standard decoding
        if !tilingConfig.enabled {
            return try decode(latents, scaleFactor: scaleFactor, shiftFactor: shiftFactor)
        }

        var images: [CGImage] = []

        for latent in latents {
            // Scale latent first (before tiling)
            let sampleScaled = MLShapedArray<Float32>(
                scalars: latent.scalars.map { $0 / scaleFactor + shiftFactor },
                shape: latent.shape
            )

            let latentHeight = latent.shape[2]
            let latentWidth = latent.shape[3]
            let imageHeight = latentHeight * 8
            let imageWidth = latentWidth * 8

            // Check if tiling is actually needed
            if latentHeight <= tilingConfig.latentTileSize &&
               latentWidth <= tilingConfig.latentTileSize {
                // Process without tiling
                let dict = [inputName: MLMultiArray(sampleScaled)]
                let input = try MLDictionaryFeatureProvider(dictionary: dict)
                let result = try model.perform { model in
                    try model.prediction(from: input)
                }
                let outputName = result.featureNames.first!
                let output = result.featureValue(for: outputName)!.multiArrayValue!
                let image = try CGImage.fromShapedArray(MLShapedArray<Float32>(converting: output))
                images.append(image)
                continue
            }

            // Generate tile grid
            let grid = TilingUtils.generateTileGrid(
                latentHeight: latentHeight,
                latentWidth: latentWidth,
                config: tilingConfig
            )

            // Process each tile
            var decodedTiles: [[MLShapedArray<Float32>]] = []

            for row in grid {
                var decodedRow: [MLShapedArray<Float32>] = []

                for tileInfo in row {
                    // Extract latent tile from the already-scaled latent
                    let latentTile = TilingUtils.extractLatentTile(
                        from: sampleScaled,
                        tile: tileInfo
                    )

                    // Run VAE decoder on tile
                    let dict = [inputName: MLMultiArray(latentTile)]
                    let input = try MLDictionaryFeatureProvider(dictionary: dict)

                    let result = try model.perform { model in
                        try model.prediction(from: input)
                    }

                    let outputName = result.featureNames.first!
                    let output = result.featureValue(for: outputName)!.multiArrayValue!
                    let imageTile = MLShapedArray<Float32>(converting: output)

                    decodedRow.append(imageTile)
                }
                decodedTiles.append(decodedRow)
            }

            // Stitch image tiles together with blending
            let stitchedImage = TilingUtils.stitchImageTiles(
                tiles: decodedTiles,
                grid: grid,
                config: tilingConfig,
                outputHeight: imageHeight,
                outputWidth: imageWidth
            )

            // Convert to CGImage
            let cgImage = try CGImage.fromShapedArray(stitchedImage)
            images.append(cgImage)
        }

        return images
    }

    var inputName: String {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.key
        }
    }

}
