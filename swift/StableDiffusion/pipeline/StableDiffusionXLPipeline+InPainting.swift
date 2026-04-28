// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import CoreGraphics
import CoreML
import Foundation

/// InPainting helpers for `StableDiffusionXLPipeline`.
///
/// To keep `generateImages(...)` readable we lift all inpainting-specific state and math
/// into a dedicated context type plus a few small helpers, so the denoising loop
/// only has to deal with two thin call sites:
///
/// ```swift
/// let inpaintCtx = try makeInPaintingContext(config: config, encoder: encoder)
/// // ...
/// if let inpaintCtx {
///     latents[i]         = inpaintCtx.blendStepLatents(latents[i], scheduler: scheduler[i],
///                                                      timeSteps: timeSteps, stepIndex: step)
///     denoisedLatents[i] = inpaintCtx.blendPreview(predictedX0: scheduler[i].modelOutputs.last)
/// } else {
///     denoisedLatents[i] = scheduler[i].modelOutputs.last ?? latents[i]
/// }
/// ```
///
/// Behavior matches the diffusers `StableDiffusionXLControlNetUnionInpaintPipeline`
/// reference: at each step the unmasked region is replaced with a re-noised version
/// of the original image latent, while the masked region keeps the denoised result.
@available(iOS 16.2, macOS 13.1, *)
extension StableDiffusionXLPipeline {

    /// Per-step latent blending state for inpainting. Computed once at the start of
    /// `generateImages` and consumed by the denoising loop. `nil` when not inpainting.
    public struct InPaintingContext: Sendable {
        /// VAE-encoded latent of the CLEAN original image. Used both for re-noising
        /// the unmasked region every step and for the preview blend.
        let imageLatent: MLShapedArray<Float32>
        /// Fixed noise tensor used to re-noise the original at each timestep
        /// (the diffusers reference reuses a single noise sample across steps).
        let noise: MLShapedArray<Float32>
        /// Binary mask in latent space: shape `[1, 1, latentH, latentW]`.
        /// `1` = regenerate (denoised wins), `0` = preserve (re-noised original wins).
        let maskLatent: MLShapedArray<Float32>
    }

    /// Build an `InPaintingContext` if all prerequisites (`startingImage`, `maskImage`,
    /// encoder) are present. Returns `nil` otherwise so the caller can keep a single
    /// `if let` gate at each call site.
    ///
    /// We don't gate on `config.mode == .inPainting` because that mode is itself
    /// derived from `maskImage != nil` (see `PipelineConfiguration.mode`), so the
    /// presence of `startingImage` and `maskImage` *is* the inpainting signal.
    func makeInPaintingContext(
        config: Configuration,
        encoder: Encoder?
    ) throws -> InPaintingContext? {
        guard let image = config.startingImage,
              let maskCGImage = config.maskImage,
              let encoder = encoder
        else {
            return nil
        }

        var sampleShape = unet.latentSampleShape
        sampleShape[0] = 1

        // Encode the CLEAN original image (not the masked composite) — this latent is
        // what we re-noise per step in the unmasked region.
        var encRandom = randomSource(from: config.rngType, seed: config.seed)
        let imageLatent: MLShapedArray<Float32>
        if config.tilingConfig.enabled {
            imageLatent = try encoder.encodeTiled(
                image,
                scaleFactor: config.encoderScaleFactor,
                random: &encRandom,
                tilingConfig: config.tilingConfig
            )
        } else {
            imageLatent = try encoder.encode(image, scaleFactor: config.encoderScaleFactor, random: &encRandom)
        }

        // Same-seed noise tensor — reused across steps to re-noise the original.
        var noiseRandom = randomSource(from: config.rngType, seed: config.seed)
        let noise = MLShapedArray<Float32>(
            converting: noiseRandom.normalShapedArray(sampleShape, mean: 0.0, stdev: 1.0)
        )

        // Downsample the user-supplied mask image to latent resolution with a binary threshold.
        let maskLatent = try Self.makeBinaryLatentMask(
            from: maskCGImage,
            latentH: sampleShape[2],
            latentW: sampleShape[3]
        )

        return InPaintingContext(imageLatent: imageLatent, noise: noise, maskLatent: maskLatent)
    }

    /// Renders `maskCGImage` into a 1-channel grayscale buffer, area-averages each
    /// latent cell, and thresholds at 0.5 to produce a binary `[1, 1, H, W]` latent mask.
    private static func makeBinaryLatentMask(
        from maskCGImage: CGImage,
        latentH: Int,
        latentW: Int
    ) throws -> MLShapedArray<Float32> {
        let maskW = maskCGImage.width
        let maskH = maskCGImage.height
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let maskCtx = CGContext(
            data: nil, width: maskW, height: maskH,
            bitsPerComponent: 8, bytesPerRow: maskW,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            throw PipelineError.resourceLoadError
        }
        maskCtx.draw(maskCGImage, in: CGRect(x: 0, y: 0, width: maskW, height: maskH))
        guard let maskData = maskCtx.data else { throw PipelineError.resourceLoadError }
        let maskPixels = maskData.bindMemory(to: UInt8.self, capacity: maskW * maskH)

        var maskScalars = [Float32](repeating: 0, count: latentH * latentW)
        let blockW = Float(maskW) / Float(latentW)
        let blockH = Float(maskH) / Float(latentH)
        for ly in 0..<latentH {
            for lx in 0..<latentW {
                let startX = Int(Float(lx) * blockW)
                let endX = min(Int(Float(lx + 1) * blockW), maskW)
                let startY = Int(Float(ly) * blockH)
                let endY = min(Int(Float(ly + 1) * blockH), maskH)
                var sum: Float = 0
                var count: Float = 0
                for py in startY..<endY {
                    for px in startX..<endX {
                        sum += Float(maskPixels[py * maskW + px]) / 255.0
                        count += 1
                    }
                }
                let avg = count > 0 ? sum / count : 0
                maskScalars[ly * latentW + lx] = avg >= 0.5 ? 1.0 : 0.0
            }
        }
        return MLShapedArray<Float32>(scalars: maskScalars, shape: [1, 1, latentH, latentW])
    }
}

// MARK: - Per-step blending

@available(iOS 16.2, macOS 13.1, *)
extension StableDiffusionXLPipeline.InPaintingContext {

    /// Per-step blend that overrides the scheduler's stepped latents.
    ///
    /// `(1 - mask) * re_noised(original) + mask * denoised`
    ///
    /// On the final step we use the CLEAN original (no re-noise) so the unmasked
    /// region decodes to the source pixels exactly.
    func blendStepLatents(
        _ denoised: MLShapedArray<Float32>,
        scheduler: Scheduler,
        timeSteps: [Double],
        stepIndex: Int
    ) -> MLShapedArray<Float32> {
        let background: MLShapedArray<Float32>
        if stepIndex < timeSteps.count - 1 {
            // Re-noise the original at the *next* timestep so it matches the noise
            // level of `denoised` after the scheduler step we just took.
            let nextTimeStep = timeSteps[stepIndex + 1]
            if let euler = scheduler as? DiscreteEulerScheduler {
                background = euler.addNoise(
                    originalSample: imageLatent, noise: [noise], timeStep: nextTimeStep
                )[0]
            } else {
                background = scheduler.addNoise(
                    originalSample: imageLatent, noise: noise, timeStep: nextTimeStep
                )
            }
        } else {
            background = imageLatent
        }
        return Self.blend(denoised: denoised, background: background, mask: maskLatent)
    }

    /// Preview blend used to drive the intermediate decode shown to the user, and
    /// for the final-step decoded image.
    ///
    /// We use the scheduler's `pred_x0` (which is *clean*) on the masked side
    /// instead of `latents` (which carries `sigma * N(0,1)` noise in early steps
    /// and would decode to rainbow noise). On the unmasked side we use the
    /// CLEAN original latent so the preserved pixels are correct from step 0.
    ///
    /// Returns `predictedX0` unchanged when there is no `pred_x0` to blend with.
    func blendPreview(predictedX0: MLShapedArray<Float32>?) -> MLShapedArray<Float32>? {
        guard let predictedX0 else { return nil }
        return Self.blend(denoised: predictedX0, background: imageLatent, mask: maskLatent)
    }

    /// `(1 - mask) * background + mask * denoised`, with `mask` broadcast over the
    /// channel dimension. `denoised` and `background` must share shape `[B, C, H, W]`;
    /// `mask` has shape `[1, 1, H, W]`.
    private static func blend(
        denoised: MLShapedArray<Float32>,
        background: MLShapedArray<Float32>,
        mask: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let shape = denoised.shape
        let channels = shape[1]
        let latH = shape[2]
        let latW = shape[3]
        return MLShapedArray<Float32>(unsafeUninitializedShape: shape) { scalars, _ in
            denoised.withUnsafeShapedBufferPointer { den, _, _ in
                background.withUnsafeShapedBufferPointer { bg, _, _ in
                    mask.withUnsafeShapedBufferPointer { m, _, _ in
                        for c in 0..<channels {
                            for y in 0..<latH {
                                for x in 0..<latW {
                                    let idx = c * latH * latW + y * latW + x
                                    let mv = m[y * latW + x]
                                    scalars.initializeElement(
                                        at: idx,
                                        to: (1.0 - mv) * bg[idx] + mv * den[idx]
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
