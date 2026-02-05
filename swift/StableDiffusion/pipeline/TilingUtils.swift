// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation
import CoreML
import CoreGraphics
import Accelerate

/// Information about a single tile for processing
@available(iOS 16.2, macOS 13.1, *)
public struct TileInfo {
    /// Position in latent space
    public let latentX: Int
    public let latentY: Int

    /// Size in latent space
    public let latentWidth: Int
    public let latentHeight: Int

    /// Position in image space (latent * 8)
    public var imageX: Int { latentX * 8 }
    public var imageY: Int { latentY * 8 }

    /// Size in image space (latent * 8)
    public var imageWidth: Int { latentWidth * 8 }
    public var imageHeight: Int { latentHeight * 8 }

    /// Whether this tile has overlap with adjacent tiles
    public let hasLeftOverlap: Bool
    public let hasTopOverlap: Bool
    public let hasRightOverlap: Bool
    public let hasBottomOverlap: Bool

    public init(
        latentX: Int,
        latentY: Int,
        latentWidth: Int,
        latentHeight: Int,
        hasLeftOverlap: Bool,
        hasTopOverlap: Bool,
        hasRightOverlap: Bool,
        hasBottomOverlap: Bool
    ) {
        self.latentX = latentX
        self.latentY = latentY
        self.latentWidth = latentWidth
        self.latentHeight = latentHeight
        self.hasLeftOverlap = hasLeftOverlap
        self.hasTopOverlap = hasTopOverlap
        self.hasRightOverlap = hasRightOverlap
        self.hasBottomOverlap = hasBottomOverlap
    }
}

/// Utilities for tiled VAE processing
@available(iOS 16.2, macOS 13.1, *)
public struct TilingUtils {

    // MARK: - Tile Grid Generation

    /// Generate a 2D grid of tile information for processing
    /// - Parameters:
    ///   - latentHeight: Height of the full latent in pixels
    ///   - latentWidth: Width of the full latent in pixels
    ///   - config: Tiling configuration
    /// - Returns: 2D array of TileInfo [row][col]
    public static func generateTileGrid(
        latentHeight: Int,
        latentWidth: Int,
        config: TilingConfiguration
    ) -> [[TileInfo]] {
        let yPositions = config.tilePositions(forLatentDimension: latentHeight)
        let xPositions = config.tilePositions(forLatentDimension: latentWidth)

        var grid: [[TileInfo]] = []
        grid.reserveCapacity(yPositions.count)

        for (rowIdx, y) in yPositions.enumerated() {
            var row: [TileInfo] = []
            row.reserveCapacity(xPositions.count)

            for (colIdx, x) in xPositions.enumerated() {
                // Determine actual overlap by checking spatial positions
                // A tile has overlap with neighbor if their regions actually intersect
                let tileWidth = min(config.latentTileSize, latentWidth - x)
                let tileHeight = min(config.latentTileSize, latentHeight - y)

                // Check left overlap: does this tile's start overlap with previous tile's region?
                let hasLeftOverlap = colIdx > 0 && x < (xPositions[colIdx - 1] + config.latentTileSize)

                // Check right overlap: does next tile's start overlap with this tile's region?
                let hasRightOverlap = colIdx < xPositions.count - 1 && xPositions[colIdx + 1] < (x + tileWidth)

                // Check top overlap: does this tile's start overlap with previous tile's region?
                let hasTopOverlap = rowIdx > 0 && y < (yPositions[rowIdx - 1] + config.latentTileSize)

                // Check bottom overlap: does next tile's start overlap with this tile's region?
                let hasBottomOverlap = rowIdx < yPositions.count - 1 && yPositions[rowIdx + 1] < (y + tileHeight)

                let tile = TileInfo(
                    latentX: x,
                    latentY: y,
                    latentWidth: tileWidth,
                    latentHeight: tileHeight,
                    hasLeftOverlap: hasLeftOverlap,
                    hasTopOverlap: hasTopOverlap,
                    hasRightOverlap: hasRightOverlap,
                    hasBottomOverlap: hasBottomOverlap
                )
                row.append(tile)
            }
            grid.append(row)
        }

        return grid
    }

    // MARK: - Image Tile Extraction

    /// Extract a tile region from a CGImage
    /// - Parameters:
    ///   - image: Source image
    ///   - tile: Tile information with position and size
    /// - Returns: Cropped CGImage tile, or nil if extraction fails
    public static func extractImageTile(
        from image: CGImage,
        tile: TileInfo
    ) -> CGImage? {
        let rect = CGRect(
            x: tile.imageX,
            y: tile.imageY,
            width: tile.imageWidth,
            height: tile.imageHeight
        )
        return image.cropping(to: rect)
    }

    // MARK: - Latent Tile Extraction

    /// Extract a tile from a latent MLShapedArray
    /// - Parameters:
    ///   - latent: Source latent array with shape [1, C, H, W] or [C, H, W]
    ///   - tile: Tile information with position and size
    /// - Returns: Extracted tile as MLShapedArray
    public static func extractLatentTile(
        from latent: MLShapedArray<Float32>,
        tile: TileInfo
    ) -> MLShapedArray<Float32> {
        let shape = latent.shape
        let hasBatchDim = shape.count == 4
        let channels = hasBatchDim ? shape[1] : shape[0]
        let srcHeight = hasBatchDim ? shape[2] : shape[1]
        let srcWidth = hasBatchDim ? shape[3] : shape[2]

        var tileScalars: [Float32] = []
        tileScalars.reserveCapacity(channels * tile.latentHeight * tile.latentWidth)

        latent.withUnsafeShapedBufferPointer { ptr, _, strides in
            for c in 0..<channels {
                for y in 0..<tile.latentHeight {
                    for x in 0..<tile.latentWidth {
                        let srcY = tile.latentY + y
                        let srcX = tile.latentX + x

                        // Bounds check
                        guard srcY < srcHeight && srcX < srcWidth else {
                            tileScalars.append(0)
                            continue
                        }

                        let idx: Int
                        if hasBatchDim {
                            idx = 0 * strides[0] + c * strides[1] + srcY * strides[2] + srcX * strides[3]
                        } else {
                            idx = c * strides[0] + srcY * strides[1] + srcX * strides[2]
                        }
                        tileScalars.append(ptr[idx])
                    }
                }
            }
        }

        if hasBatchDim {
            return MLShapedArray<Float32>(
                scalars: tileScalars,
                shape: [1, channels, tile.latentHeight, tile.latentWidth]
            )
        } else {
            return MLShapedArray<Float32>(
                scalars: tileScalars,
                shape: [channels, tile.latentHeight, tile.latentWidth]
            )
        }
    }

    // MARK: - Tile Stitching

    /// Stitch latent tiles back together using safe region cropping (no blending)
    ///
    /// Instead of blending overlapping regions, only copies the "safe" center region
    /// of each tile where no overlap exists. The overlap regions act purely as context
    /// for the VAE convolutions but are discarded during stitching.
    ///
    /// This approach is faster than blending and produces seamless results because
    /// the VAE's receptive field is much smaller than the tile size.
    ///
    /// Uses `vDSP_mmov` for efficient 2D block copying - one call per channel instead
    /// of one call per row, resulting in ~32x fewer function calls for interior tiles.
    ///
    /// - Parameters:
    ///   - tiles: 2D array of tile data [row][col]
    ///   - grid: 2D array of tile info matching tiles
    ///   - config: Tiling configuration (unused, kept for API compatibility)
    ///   - outputHeight: Target output height
    ///   - outputWidth: Target output width
    /// - Returns: Stitched latent as MLShapedArray with shape [1, C, H, W]
    public static func stitchLatentTiles(
        tiles: [[MLShapedArray<Float32>]],
        grid: [[TileInfo]],
        config: TilingConfiguration,
        outputHeight: Int,
        outputWidth: Int
    ) -> MLShapedArray<Float32> {
        guard let firstTile = tiles.first?.first else {
            fatalError("No tiles to stitch")
        }

        let hasBatchDim = firstTile.shape.count == 4
        let channels = hasBatchDim ? firstTile.shape[1] : firstTile.shape[0]
        let numCols = grid[0].count
        let numRows = grid.count

        // Use unsafeUninitializedShape to write directly into the MLShapedArray's
        // backing buffer - avoids intermediate allocations and final copy
        return MLShapedArray<Float32>(
            unsafeUninitializedShape: [1, channels, outputHeight, outputWidth]
        ) { outputPtr, _ in

            for (rowIdx, row) in tiles.enumerated() {
                for (colIdx, tile) in row.enumerated() {
                    let info = grid[rowIdx][colIdx]

                    tile.withUnsafeShapedBufferPointer { srcPtr, _, strides in
                        // Calculate the output region this tile is responsible for
                        // by finding midpoints between overlapping tiles
                        let dstStartX: Int
                        let dstEndX: Int
                        let dstStartY: Int
                        let dstEndY: Int

                        // X boundaries
                        if colIdx == 0 {
                            dstStartX = 0
                        } else {
                            // Midpoint between this tile's start and previous tile's end
                            let prevInfo = grid[rowIdx][colIdx - 1]
                            let prevTileEnd = prevInfo.latentX + prevInfo.latentWidth
                            dstStartX = (info.latentX + prevTileEnd) / 2
                        }

                        if colIdx == numCols - 1 {
                            dstEndX = outputWidth
                        } else {
                            // Midpoint between this tile's end and next tile's start
                            let nextInfo = grid[rowIdx][colIdx + 1]
                            let thisTileEnd = info.latentX + info.latentWidth
                            dstEndX = (thisTileEnd + nextInfo.latentX) / 2
                        }

                        // Y boundaries
                        if rowIdx == 0 {
                            dstStartY = 0
                        } else {
                            // Midpoint between this tile's start and previous tile's end
                            let prevInfo = grid[rowIdx - 1][colIdx]
                            let prevTileEnd = prevInfo.latentY + prevInfo.latentHeight
                            dstStartY = (info.latentY + prevTileEnd) / 2
                        }

                        if rowIdx == numRows - 1 {
                            dstEndY = outputHeight
                        } else {
                            // Midpoint between this tile's end and next tile's start
                            let nextInfo = grid[rowIdx + 1][colIdx]
                            let thisTileEnd = info.latentY + info.latentHeight
                            dstEndY = (thisTileEnd + nextInfo.latentY) / 2
                        }

                        // Calculate source region within tile corresponding to destination
                        let srcStartX = dstStartX - info.latentX
                        let srcStartY = dstStartY - info.latentY
                        let safeWidth = dstEndX - dstStartX
                        let safeHeight = dstEndY - dstStartY

                        // Source row stride (elements between row starts)
                        let srcRowStride = hasBatchDim ? strides[2] : strides[1]

                        // Copy safe region using vDSP_mmov for efficient 2D block copy
                        // One call per channel instead of one call per row
                        for c in 0..<channels {
                            // Source: start of safe region for this channel
                            let srcStart: Int
                            if hasBatchDim {
                                srcStart = c * strides[1] + srcStartY * strides[2] + srcStartX
                            } else {
                                srcStart = c * strides[0] + srcStartY * strides[1] + srcStartX
                            }

                            // Destination: start of safe region for this channel
                            let dstStart = c * outputHeight * outputWidth + dstStartY * outputWidth + dstStartX

                            // Copy entire 2D safe region in ONE call per channel
                            // vDSP_mmov parameters: (src, dst, columns, rows, srcRowStride, dstRowStride)
                            vDSP_mmov(
                                srcPtr.baseAddress! + srcStart,
                                outputPtr.baseAddress! + dstStart,
                                vDSP_Length(safeWidth),   // M = number of columns to copy
                                vDSP_Length(safeHeight),  // N = number of rows to copy
                                vDSP_Length(srcRowStride),
                                vDSP_Length(outputWidth)
                            )
                        }
                    }
                }
            }
        }
    }

    /// Stitch image tiles back together using safe region cropping (no blending)
    /// This is a convenience wrapper that converts image-space parameters
    /// - Parameters:
    ///   - tiles: 2D array of decoded image tiles as MLShapedArray [1, 3, H, W]
    ///   - grid: 2D array of tile info (in latent space coordinates)
    ///   - config: Tiling configuration
    ///   - outputHeight: Target output height in pixels
    ///   - outputWidth: Target output width in pixels
    /// - Returns: Stitched image as MLShapedArray with shape [1, 3, H, W]
    public static func stitchImageTiles(
        tiles: [[MLShapedArray<Float32>]],
        grid: [[TileInfo]],
        config: TilingConfiguration,
        outputHeight: Int,
        outputWidth: Int
    ) -> MLShapedArray<Float32> {
        // Convert grid to image-space tile info
        let imageGrid = grid.map { row in
            row.map { tile in
                TileInfo(
                    latentX: tile.imageX,
                    latentY: tile.imageY,
                    latentWidth: tile.imageWidth,
                    latentHeight: tile.imageHeight,
                    hasLeftOverlap: tile.hasLeftOverlap,
                    hasTopOverlap: tile.hasTopOverlap,
                    hasRightOverlap: tile.hasRightOverlap,
                    hasBottomOverlap: tile.hasBottomOverlap
                )
            }
        }

        // Create image-space config (overlap is scaled by 8)
        let imageConfig = TilingConfiguration(
            enabled: config.enabled,
            latentTileSize: config.imageTileSize,
            latentOverlap: config.imageOverlap
        )

        return stitchLatentTiles(
            tiles: tiles,
            grid: imageGrid,
            config: imageConfig,
            outputHeight: outputHeight,
            outputWidth: outputWidth
        )
    }
}
