// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation
import CoreML

/// A encoder model which produces latent samples from RGB images
@available(iOS 16.2, macOS 13.1, *)
public struct Encoder: ResourceManaging {
    
    public enum Error: String, Swift.Error {
        case sampleInputShapeNotCorrect
        case tileExtractionFailed
        case tiledEncodingFailed
    }
    
    /// VAE encoder model + post math and adding noise from schedular
    var model: ManagedMLModel
    
    /// Create encoder from Core ML model
    ///
    /// - Parameters:
    ///     - url: Location of compiled VAE encoder Core ML model
    ///     - configuration: configuration to be used when the model is loaded
    /// - Returns: An encoder that will lazily load its required resources when needed or requested
    public init(modelAt url: URL, configuration: MLModelConfiguration) {
        self.model = ManagedMLModel(modelAt: url, configuration: configuration)
    }

    public var loadProgressWeights: [Int64] {
        return [15]
    }

    public func makeLoadProgress() -> Progress {
        let progress = Progress(totalUnitCount: self.loadProgressWeights.first!)
        progress.localizedDescription = "Encoder"
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
    
    /// Prediction queue
    let queue = DispatchQueue(label: "encoder.predict")

    /// Encode image into latent sample
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///    - scaleFactor: scalar multiplier on latents before encoding image
    ///    - random
    ///  - Returns: The encoded latent space as MLShapedArray
    public func encode(
        _ image: CGImage,
        scaleFactor: Float32,
        random: inout RandomSource
    ) throws -> MLShapedArray<Float32> {
        let imageData = try image.planarRGBShapedArray(minValue: -1.0, maxValue: 1.0)
        guard imageData.shape == inputShape else {
            // TODO: Consider auto resizing and croping similar to how Vision or CoreML auto-generated Swift code can accomplish with `MLFeatureValue`
            throw Error.sampleInputShapeNotCorrect
        }
        let dict = [inputName: MLMultiArray(imageData)]
        let input = try MLDictionaryFeatureProvider(dictionary: dict)
        
        let result = try model.perform { model in
            try model.prediction(from: input)
        }
        let outputName = result.featureNames.first!
        let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
        let output = MLShapedArray<Float32>(converting: outputValue)
        
        // DiagonalGaussianDistribution
        let mean = output[0][0..<4]
        let logvar = MLShapedArray<Float32>(
            scalars: output[0][4..<8].scalars.map { min(max($0, -30), 20) },
            shape: mean.shape
        )
        let std = MLShapedArray<Float32>(
            scalars: logvar.scalars.map { exp(0.5 * $0) },
            shape: logvar.shape
        )
        let latent = MLShapedArray<Float32>(
            scalars: zip(mean.scalars, std.scalars).map {
                Float32(random.nextNormal(mean: Double($0), stdev: Double($1)))
            },
            shape: logvar.shape
        )
        
        // Reference pipeline scales the latent after encoding
        let latentScaled = MLShapedArray<Float32>(
            scalars: latent.scalars.map { $0 * scaleFactor },
            shape: [1] + latent.shape
        )

        return latentScaled
    }

    /// Encode image into latent sample using tiled processing
    ///
    /// This method processes the image in tiles to reduce memory usage,
    /// enabling Neural Engine execution on devices with memory constraints.
    /// The mean latents are stitched first, then noise is sampled on the full
    /// result for cross-tile consistency.
    ///
    ///  - Parameters:
    ///    - image: Input image
    ///    - scaleFactor: scalar multiplier on latents after encoding
    ///    - random: Random source for sampling
    ///    - tilingConfig: Configuration for tile-based processing
    ///  - Returns: The encoded latent space as MLShapedArray
    public func encodeTiled(
        _ image: CGImage,
        scaleFactor: Float32,
        random: inout RandomSource,
        tilingConfig: TilingConfiguration
    ) throws -> MLShapedArray<Float32> {
        // If tiling disabled, use standard encoding
        if !tilingConfig.enabled {
            return try encode(image, scaleFactor: scaleFactor, random: &random)
        }

        let imageWidth = image.width
        let imageHeight = image.height
        let latentWidth = imageWidth / 8
        let latentHeight = imageHeight / 8

        // Check if tiling is actually needed (image fits in one tile)
        if latentHeight <= tilingConfig.latentTileSize &&
           latentWidth <= tilingConfig.latentTileSize {
            return try encode(image, scaleFactor: scaleFactor, random: &random)
        }

        // Generate tile grid
        let grid = TilingUtils.generateTileGrid(
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            config: tilingConfig
        )

        // Process each tile and collect mean latents (defer sampling until stitched)
        var meanTiles: [[MLShapedArray<Float32>]] = []
        var logvarTiles: [[MLShapedArray<Float32>]] = []

        for row in grid {
            var meanRow: [MLShapedArray<Float32>] = []
            var logvarRow: [MLShapedArray<Float32>] = []

            for tileInfo in row {
                // Extract image tile
                guard let imageTile = TilingUtils.extractImageTile(from: image, tile: tileInfo) else {
                    throw Error.tileExtractionFailed
                }

                // Convert tile to MLShapedArray
                let tileData = try imageTile.planarRGBShapedArray(minValue: -1.0, maxValue: 1.0)

                // Run VAE encoder on tile
                let dict = [inputName: MLMultiArray(tileData)]
                let input = try MLDictionaryFeatureProvider(dictionary: dict)

                let result = try model.perform { model in
                    try model.prediction(from: input)
                }

                let outputName = result.featureNames.first!
                let outputValue = result.featureValue(for: outputName)!.multiArrayValue!
                let output = MLShapedArray<Float32>(converting: outputValue)

                // Extract mean (first 4 channels) - defer sampling
                let mean = MLShapedArray<Float32>(
                    scalars: Array(output[0][0..<4].scalars),
                    shape: [4, tileInfo.latentHeight, tileInfo.latentWidth]
                )

                // Extract logvar (next 4 channels) and clamp
                let logvar = MLShapedArray<Float32>(
                    scalars: output[0][4..<8].scalars.map { min(max($0, -30), 20) },
                    shape: [4, tileInfo.latentHeight, tileInfo.latentWidth]
                )

                meanRow.append(mean)
                logvarRow.append(logvar)
            }
            meanTiles.append(meanRow)
            logvarTiles.append(logvarRow)
        }

        // Stitch mean tiles together with blending
        let stitchedMean = TilingUtils.stitchLatentTiles(
            tiles: meanTiles,
            grid: grid,
            config: tilingConfig,
            outputHeight: latentHeight,
            outputWidth: latentWidth
        )

        // Stitch logvar tiles together with blending
        let stitchedLogvar = TilingUtils.stitchLatentTiles(
            tiles: logvarTiles,
            grid: grid,
            config: tilingConfig,
            outputHeight: latentHeight,
            outputWidth: latentWidth
        )

        // Now sample noise on the full stitched result for consistency
        let std = MLShapedArray<Float32>(
            scalars: stitchedLogvar.scalars.map { exp(0.5 * $0) },
            shape: stitchedLogvar.shape
        )

        let latent = MLShapedArray<Float32>(
            scalars: zip(stitchedMean.scalars, std.scalars).map {
                Float32(random.nextNormal(mean: Double($0), stdev: Double($1)))
            },
            shape: stitchedMean.shape
        )

        // Apply scale factor
        let latentScaled = MLShapedArray<Float32>(
            scalars: latent.scalars.map { $0 * scaleFactor },
            shape: latent.shape
        )

        return latentScaled
    }

    var inputDescription: MLFeatureDescription {
        try! model.perform { model in
            model.modelDescription.inputDescriptionsByName.first!.value
        }
    }
    
    var inputName: String {
        inputDescription.name
    }
    
    /// The expected shape of the models latent sample input
    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }
}
