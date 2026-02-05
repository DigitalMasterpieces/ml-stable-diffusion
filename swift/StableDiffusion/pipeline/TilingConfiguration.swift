// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation

/// Configuration for tiled VAE encoding/decoding to enable Neural Engine execution
/// on memory-constrained devices like iPhone 14 Pro at high resolutions (1024x1024).
public struct TilingConfiguration: Hashable, Sendable {

    /// Whether tiling is enabled
    public var enabled: Bool

    /// Tile size in latent space (will be multiplied by 8 for image space)
    /// Default: 64 (512 pixels in image space)
    public var latentTileSize: Int

    /// Overlap size in latent space (will be multiplied by 8 for image space)
    /// Default: 16 (128 pixels, 25% of tile size)
    /// Note: Overlap regions provide context for VAE convolutions but are discarded
    /// during stitching - only the "safe" center regions are copied to the output.
    public var latentOverlap: Int

    // MARK: - Computed Properties

    /// Computed stride in latent space (tile size minus overlap)
    public var latentStride: Int {
        latentTileSize - latentOverlap
    }

    /// Tile size in image space (latent * 8)
    public var imageTileSize: Int {
        latentTileSize * 8
    }

    /// Overlap size in image space (latent * 8)
    public var imageOverlap: Int {
        latentOverlap * 8
    }

    /// Stride in image space (latent * 8)
    public var imageStride: Int {
        latentStride * 8
    }

    // MARK: - Presets

    /// Default configuration optimized for Neural Engine on iPhone 14 Pro
    /// Uses 25% overlap (16 latent pixels) for VAE context, but only safe center
    /// regions are copied during stitching (no blending needed).
    public static let `default` = TilingConfiguration(
        enabled: true,
        latentTileSize: 64,
        latentOverlap: 16
    )

    /// Disabled tiling (process full image at once)
    public static let disabled = TilingConfiguration(
        enabled: false,
        latentTileSize: 64,
        latentOverlap: 16
    )

    // MARK: - Initialization

    public init(
        enabled: Bool = true,
        latentTileSize: Int = 64,
        latentOverlap: Int = 16
    ) {
        precondition(latentTileSize > 0, "Tile size must be positive")
        precondition(latentOverlap >= 0, "Overlap must be non-negative")
        precondition(latentOverlap < latentTileSize, "Overlap must be less than tile size")

        self.enabled = enabled
        self.latentTileSize = latentTileSize
        self.latentOverlap = latentOverlap
    }

    // MARK: - Tile Calculation

    /// Calculate number of tiles needed for a given latent dimension
    /// - Parameter size: The size of the latent dimension (height or width)
    /// - Returns: Number of tiles needed to cover the dimension
    public func tilesNeeded(forLatentDimension size: Int) -> Int {
        if size <= latentTileSize {
            return 1
        }
        // Number of tiles = ceil((size - tileSize) / stride) + 1
        return Int(ceil(Double(size - latentTileSize) / Double(latentStride))) + 1
    }

    /// Calculate tile start positions for a given latent dimension
    /// - Parameter size: The size of the latent dimension (height or width)
    /// - Returns: Array of starting positions for each tile
    public func tilePositions(forLatentDimension size: Int) -> [Int] {
        let count = tilesNeeded(forLatentDimension: size)
        var positions: [Int] = []
        positions.reserveCapacity(count)

        for i in 0..<count {
            // Calculate position, ensuring last tile fits within bounds
            let pos = min(i * latentStride, size - latentTileSize)
            positions.append(max(0, pos))
        }

        return positions
    }

    /// Calculate number of tiles needed for a given image dimension
    /// - Parameter size: The size of the image dimension in pixels (height or width)
    /// - Returns: Number of tiles needed to cover the dimension
    public func tilesNeeded(forImageDimension size: Int) -> Int {
        tilesNeeded(forLatentDimension: size / 8)
    }

    /// Calculate tile start positions for a given image dimension
    /// - Parameter size: The size of the image dimension in pixels (height or width)
    /// - Returns: Array of starting positions in pixels for each tile
    public func tilePositions(forImageDimension size: Int) -> [Int] {
        tilePositions(forLatentDimension: size / 8).map { $0 * 8 }
    }
}
