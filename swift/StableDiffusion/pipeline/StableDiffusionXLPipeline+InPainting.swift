// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Accelerate
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
        /// **Continuous** mask in latent space: shape `[1, 1, latentH, latentW]`, values in
        /// `[0, 1]`. `1` = regenerate (denoised wins), `0` = preserve (re-noised original
        /// wins); intermediate values blend proportionally. Mirrors the diffusers
        /// `StableDiffusionXLControlNetUnionInpaintPipeline` reference, which interpolates
        /// the user mask to latent resolution **without** thresholding so the blend has a
        /// feathered ring of fractional-weight latent pixels at the mask boundary. Hard
        /// binarising at latent resolution (previous implementation) produced a
        /// 1-latent-pixel ≈ 8-output-pixel step transition that the VAE had to materialise,
        /// which read as boundary smearing / desaturation in the decoded image.
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

        // Downsample the user-supplied mask image to latent resolution as a continuous
        // [0, 1] float field (no threshold — see `makeLatentMask` for why).
        let maskLatent = try Self.makeLatentMask(
            from: maskCGImage,
            latentH: sampleShape[2],
            latentW: sampleShape[3]
        )

        return InPaintingContext(imageLatent: imageLatent, noise: noise, maskLatent: maskLatent)
    }

    /// Downsamples `maskCGImage` to a 1-channel grayscale buffer at latent resolution
    /// (via `CGImage.withGrayscaleResampledPixels`, which uses CoreGraphics + vImage with
    /// `interpolationQuality = .high`), then normalises to `[0, 1]` Float **without
    /// thresholding**. The high-quality resample produces interpolated values along the
    /// mask boundary; we keep those fractional weights so the per-step blend in
    /// `blendStepLatents` is a feathered transition at latent resolution.
    ///
    /// Matches the diffusers `StableDiffusionXLControlNetUnionInpaintPipeline` semantic
    /// (`F.interpolate` of the pixel-space-binarised mask to latent resolution yields
    /// continuous boundary weights). The previous implementation hard-binarised at
    /// latent resolution, which produced a 1-latent-pixel-wide step transition the VAE
    /// had to materialise as a smeared / desaturated ring in pixel space.
    private static func makeLatentMask(
        from maskCGImage: CGImage,
        latentH: Int,
        latentW: Int
    ) throws -> MLShapedArray<Float32> {
        let count = latentH * latentW
        var maskScalars = [Float32](repeating: 0, count: count)

        // UInt8 (0..255) → Float32, scaled into [0, 1] in a single vDSP pass.
        // `vDSP.convertElements` widens the bytes to Float32 in-place; `vDSP_vsmul` with
        // `1/255` scales into the unit interval. We clamp defensively at the end since the
        // CoreGraphics resample could in theory emit values slightly outside [0, 255]
        // under bizarre colour-management conditions (in practice it doesn't, but the
        // clamp is one vDSP call and removes the precondition from the blend kernel).
        do {
            try maskCGImage.withGrayscaleResampledPixels(width: latentW, height: latentH) { pixels in
                maskScalars.withUnsafeMutableBufferPointer { buf in
                    var bufLocal = buf
                    vDSP.convertElements(of: UnsafeBufferPointer(start: pixels, count: count), to: &bufLocal)
                    var scale: Float = 1.0 / 255.0
                    vDSP_vsmul(buf.baseAddress!, 1, &scale, buf.baseAddress!, 1, vDSP_Length(count))
                    var low: Float = 0
                    var high: Float = 1
                    vDSP_vclip(buf.baseAddress!, 1, &low, &high, buf.baseAddress!, 1, vDSP_Length(count))
                }
            }
        } catch {
            throw PipelineError.errorMaskBinarization
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
    func blendPreview(denoisedLatents: MLShapedArray<Float32>?) -> MLShapedArray<Float32>? {
        guard let denoisedLatents else { return nil }
        return Self.blend(denoised: denoisedLatents, background: imageLatent, mask: maskLatent)
    }

    /// `(1 - mask) * background + mask * denoised`, with `mask` broadcast over the
    /// channel dimension. `denoised` and `background` must share shape `[B, C, H, W]`;
    /// `mask` has shape `[1, 1, H, W]`.
    ///
    /// Refactored as `background + mask * (denoised - background)` so each channel
    /// slice reduces to a `vDSP_vsub` + `vDSP_vma` pair on Accelerate.
    private static func blend(
        denoised: MLShapedArray<Float32>,
        background: MLShapedArray<Float32>,
        mask: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        let shape = denoised.shape
        let channels = shape[1]
        let plane = shape[2] * shape[3]
        let n = vDSP_Length(plane)
        return MLShapedArray<Float32>(unsafeUninitializedShape: shape) { scalars, _ in
            denoised.withUnsafeShapedBufferPointer { den, _, _ in
                background.withUnsafeShapedBufferPointer { bg, _, _ in
                    mask.withUnsafeShapedBufferPointer { m, _, _ in
                        let mPtr = m.baseAddress!
                        let denBase = den.baseAddress!
                        let bgBase = bg.baseAddress!
                        let outBase = scalars.baseAddress!
                        for c in 0..<channels {
                            let off = c * plane
                            // tmp = denoised - background  (written into output slice)
                            vDSP_vsub(bgBase + off, 1, denBase + off, 1, outBase + off, 1, n)
                            // out = mask * tmp + background
                            vDSP_vma(outBase + off, 1, mPtr, 1, bgBase + off, 1, outBase + off, 1, n)
                        }
                    }
                }
            }
        }
    }
}
