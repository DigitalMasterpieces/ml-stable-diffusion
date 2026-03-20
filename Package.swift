// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "stable-diffusion",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "StableDiffusion",
            targets: ["StableDiffusion"]),
        .executable(
            name: "StableDiffusionSample",
            targets: ["StableDiffusionCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.7.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "StableDiffusion",
            dependencies:  [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "swift/StableDiffusion",
            swiftSettings: [.swiftLanguageMode(.v6)]),
        .executableTarget(
            name: "StableDiffusionCLI",
            dependencies: [
                "StableDiffusion",
                .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "swift/StableDiffusionCLI",
            swiftSettings: [.swiftLanguageMode(.v6)]),
        .testTarget(
            name: "StableDiffusionTests",
            dependencies: ["StableDiffusion"],
            path: "swift/StableDiffusionTests",
            resources: [
                .copy("Resources/vocab.json"),
                .copy("Resources/merges.txt")
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]),
    ]
)
