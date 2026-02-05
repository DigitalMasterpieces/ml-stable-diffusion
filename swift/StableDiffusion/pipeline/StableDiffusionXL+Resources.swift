// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2023 Apple Inc. All Rights Reserved.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation
import CoreML
import NaturalLanguage

@available(iOS 17.0, macOS 14.0, *)
public extension StableDiffusionXLPipeline {

    struct ResourceURLs {

        public let textEncoderURL: URL
        public let textEncoder2URL: URL
        public let unetURL: URL
        public let unetChunk1URL: URL
        public let unetChunk2URL: URL
        public let unetRefinerURL: URL
        public let unetRefinerChunk1URL: URL
        public let unetRefinerChunk2URL: URL
        public let decoderURL: URL
        public let encoderURL: URL
        /// Tiled VAE decoder for Neural Engine execution (512x512 image tiles)
        public let decoderTiledURL: URL
        /// Tiled VAE encoder for Neural Engine execution (512x512 image tiles)
        public let encoderTiledURL: URL
        public let vocabURL: URL
        public let mergesURL: URL
        public let controlNetDirURL: URL
        public let controlledUnetURL: URL
        public let controlledUnetChunk1URL: URL
        public let controlledUnetChunk2URL: URL

        public init(resourcesAt baseURL: URL) {
            textEncoderURL = baseURL.appending(path: "TextEncoder.mlmodelc")
            textEncoder2URL = baseURL.appending(path: "TextEncoder2.mlmodelc")
            unetURL = baseURL.appending(path: "Unet.mlmodelc")
            unetChunk1URL = baseURL.appending(path: "UnetChunk1.mlmodelc")
            unetChunk2URL = baseURL.appending(path: "UnetChunk2.mlmodelc")
            unetRefinerURL = baseURL.appending(path: "UnetRefiner.mlmodelc")
            unetRefinerChunk1URL = baseURL.appending(path: "UnetRefinerChunk1.mlmodelc")
            unetRefinerChunk2URL = baseURL.appending(path: "UnetRefinerChunk2.mlmodelc")
            decoderURL = baseURL.appending(path: "VAEDecoder.mlmodelc")
            encoderURL = baseURL.appending(path: "VAEEncoder.mlmodelc")
            decoderTiledURL = baseURL.appending(path: "VAEDecoderTiled.mlmodelc")
            encoderTiledURL = baseURL.appending(path: "VAEEncoderTiled.mlmodelc")
            vocabURL = baseURL.appending(path: "vocab.json")
            mergesURL = baseURL.appending(path: "merges.txt")
            controlNetDirURL = baseURL.appending(path: "controlnet")
            controlledUnetURL = baseURL.appending(path: "ControlledUnet.mlmodelc")
            controlledUnetChunk1URL = baseURL.appending(path: "ControlledUnetChunk1.mlmodelc")
            controlledUnetChunk2URL = baseURL.appending(path: "ControlledUnetChunk2.mlmodelc")
        }
    }

    /// Create stable diffusion pipeline using model resources at a
    /// specified URL
    ///
    /// - Parameters:
    ///   - baseURL: URL pointing to directory holding all model and tokenization resources
    ///   - configuration: The configuration to load model resources with
    ///   - reduceMemory: Setup pipeline in reduced memory mode
    /// - Returns:
    ///  Pipeline ready for image generation if all  necessary resources loaded
    init(
        resourcesAt baseURL: URL,
        controlNet controlNetModelNames: [String],
        controlNetUnion: Bool = false,
        configuration config: MLModelConfiguration = .init(),
        reduceMemory: Bool = false
    ) throws {

        /// Expect URL of each resource
        let urls = ResourceURLs(resourcesAt: baseURL)
        let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
        let textEncoder: TextEncoderXL?
        if FileManager.default.fileExists(atPath: urls.textEncoderURL.path) {
            textEncoder = TextEncoderXL(tokenizer: tokenizer, modelAt: urls.textEncoderURL, configuration: config)
        } else {
            textEncoder = nil
        }
        
        // padToken is different in the second XL text encoder
        let tokenizer2 = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL, padToken: "!")
        let textEncoder2 = TextEncoderXL(tokenizer: tokenizer2, modelAt: urls.textEncoder2URL, configuration: config)

        // ControlNet model
        var controlNet: ControlNetXLProtocol? = nil
        let controlNetURLs = controlNetModelNames.map { model in
            let fileName = model + ".mlmodelc"
            return urls.controlNetDirURL.appending(path: fileName)
        }
        if !controlNetURLs.isEmpty {
            if controlNetUnion {
                controlNet = ControlNetUnionXL(modelAt: controlNetURLs, configuration: config)
            } else {
                controlNet = ControlNetXL(modelAt: controlNetURLs, configuration: config)
            }
        }

        // Unet model
        let unet: Unet
        if FileManager.default.fileExists(atPath: urls.unetChunk1URL.path) &&
            FileManager.default.fileExists(atPath: urls.unetChunk2URL.path) {
            unet = Unet(chunksAt: [urls.unetChunk1URL, urls.unetChunk2URL],
                        configuration: config)
        } else {
            unet = Unet(modelAt: urls.unetURL, configuration: config)
        }

        // Refiner Unet model
        let unetRefiner: Unet?
        if FileManager.default.fileExists(atPath: urls.unetRefinerChunk1URL.path) &&
            FileManager.default.fileExists(atPath: urls.unetRefinerChunk2URL.path) {
            unetRefiner = Unet(chunksAt: [urls.unetRefinerChunk1URL, urls.unetRefinerChunk2URL],
                               configuration: config)
        } else if FileManager.default.fileExists(atPath: urls.unetRefinerURL.path) {
            unetRefiner = Unet(modelAt: urls.unetRefinerURL, configuration: config)
        } else {
            unetRefiner = nil
        }


        // Image Decoder
        // Use cpuAndGPU by default since ANE doesn't support FLOAT32 at full resolution.
        // When VAE tiling is enabled (via TilingConfiguration), each tile is small enough
        // for ANE, but the compute units are configured at model load time, not at inference.
        // For Neural Engine support with tiling, use the enableVAETiling initializer.
        let vaeConfig = config.copy() as! MLModelConfiguration
        vaeConfig.computeUnits = .cpuAndGPU
        let decoder = Decoder(modelAt: urls.decoderURL, configuration: vaeConfig)

        // Optional Image Encoder
        let encoder: Encoder?
        if FileManager.default.fileExists(atPath: urls.encoderURL.path) {
            encoder = Encoder(modelAt: urls.encoderURL, configuration: vaeConfig)
        } else {
            encoder = nil
        }

        // Construct pipeline
        self.init(
            textEncoder: textEncoder,
            textEncoder2: textEncoder2,
            imageEncoder: nil,
            unet: unet,
            unetRefiner: unetRefiner,
            decoder: decoder,
            encoder: encoder,
            controlNet: controlNet,
            reduceMemory: reduceMemory
        )
    }

    /// Create stable diffusion pipeline with Neural Engine support via VAE tiling
    ///
    /// When VAE tiling is enabled, the encoder and decoder process images in smaller tiles
    /// that fit within Neural Engine memory constraints. This allows the VAE to use
    /// Neural Engine compute units instead of being forced to CPU+GPU.
    ///
    /// - Parameters:
    ///   - baseURL: URL pointing to directory holding all model and tokenization resources
    ///   - controlNetModelNames: Names of ControlNet models to load
    ///   - controlNetUnion: Whether to use ControlNet Union
    ///   - config: The configuration to load model resources with
    ///   - reduceMemory: Setup pipeline in reduced memory mode
    ///   - enableVAETiling: When true, configures VAE for Neural Engine with tiled processing
    /// - Returns: Pipeline ready for image generation if all necessary resources loaded
    init(
        resourcesAt baseURL: URL,
        controlNet controlNetModelNames: [String],
        controlNetUnion: Bool = false,
        configuration config: MLModelConfiguration = .init(),
        reduceMemory: Bool = false,
        enableVAETiling: Bool
    ) throws {

        /// Expect URL of each resource
        let urls = ResourceURLs(resourcesAt: baseURL)
        let tokenizer = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL)
        let textEncoder: TextEncoderXL?
        if FileManager.default.fileExists(atPath: urls.textEncoderURL.path) {
            textEncoder = TextEncoderXL(tokenizer: tokenizer, modelAt: urls.textEncoderURL, configuration: config)
        } else {
            textEncoder = nil
        }

        // padToken is different in the second XL text encoder
        let tokenizer2 = try BPETokenizer(mergesAt: urls.mergesURL, vocabularyAt: urls.vocabURL, padToken: "!")
        let textEncoder2 = TextEncoderXL(tokenizer: tokenizer2, modelAt: urls.textEncoder2URL, configuration: config)

        // ControlNet model
        var controlNet: ControlNetXLProtocol? = nil
        let controlNetURLs = controlNetModelNames.map { model in
            let fileName = model + ".mlmodelc"
            return urls.controlNetDirURL.appending(path: fileName)
        }
        if !controlNetURLs.isEmpty {
            if controlNetUnion {
                controlNet = ControlNetUnionXL(modelAt: controlNetURLs, configuration: config)
            } else {
                controlNet = ControlNetXL(modelAt: controlNetURLs, configuration: config)
            }
        }

        // Unet model
        let unet: Unet
        if FileManager.default.fileExists(atPath: urls.unetChunk1URL.path) &&
            FileManager.default.fileExists(atPath: urls.unetChunk2URL.path) {
            unet = Unet(chunksAt: [urls.unetChunk1URL, urls.unetChunk2URL],
                        configuration: config)
        } else {
            unet = Unet(modelAt: urls.unetURL, configuration: config)
        }

        // Refiner Unet model
        let unetRefiner: Unet?
        if FileManager.default.fileExists(atPath: urls.unetRefinerChunk1URL.path) &&
            FileManager.default.fileExists(atPath: urls.unetRefinerChunk2URL.path) {
            unetRefiner = Unet(chunksAt: [urls.unetRefinerChunk1URL, urls.unetRefinerChunk2URL],
                               configuration: config)
        } else if FileManager.default.fileExists(atPath: urls.unetRefinerURL.path) {
            unetRefiner = Unet(modelAt: urls.unetRefinerURL, configuration: config)
        } else {
            unetRefiner = nil
        }

        // Image Decoder and Encoder
        // When VAE tiling is enabled, use the tiled models (512x512 image tiles)
        // which can run on Neural Engine
        let vaeConfig = config.copy() as! MLModelConfiguration
        let decoder: Decoder
        let encoder: Encoder?

        if enableVAETiling {
            // Use tiled VAE models for Neural Engine execution
            vaeConfig.computeUnits = .all

            // Check if tiled decoder exists, fallback to standard decoder
            if FileManager.default.fileExists(atPath: urls.decoderTiledURL.path) {
                decoder = Decoder(modelAt: urls.decoderTiledURL, configuration: vaeConfig)
            } else {
                // Fallback: use standard decoder with tiled processing
                // Note: This may not work optimally on ANE without the tiled model
                decoder = Decoder(modelAt: urls.decoderURL, configuration: vaeConfig)
            }

            // Check if tiled encoder exists
            if FileManager.default.fileExists(atPath: urls.encoderTiledURL.path) {
                encoder = Encoder(modelAt: urls.encoderTiledURL, configuration: vaeConfig)
            } else if FileManager.default.fileExists(atPath: urls.encoderURL.path) {
                // Fallback: use standard encoder with tiled processing
                encoder = Encoder(modelAt: urls.encoderURL, configuration: vaeConfig)
            } else {
                encoder = nil
            }
        } else {
            // Standard full-resolution VAE models (CPU+GPU only)
            vaeConfig.computeUnits = .cpuAndGPU
            decoder = Decoder(modelAt: urls.decoderURL, configuration: vaeConfig)

            if FileManager.default.fileExists(atPath: urls.encoderURL.path) {
                encoder = Encoder(modelAt: urls.encoderURL, configuration: vaeConfig)
            } else {
                encoder = nil
            }
        }

        // Construct pipeline
        self.init(
            textEncoder: textEncoder,
            textEncoder2: textEncoder2,
            imageEncoder: nil,
            unet: unet,
            unetRefiner: unetRefiner,
            decoder: decoder,
            encoder: encoder,
            controlNet: controlNet,
            reduceMemory: reduceMemory
        )
    }
}
