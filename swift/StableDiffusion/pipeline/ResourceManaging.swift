// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

/// Protocol for managing internal resources
public protocol ResourceManaging {

    /// Request resources to be loaded and ready if possible
    func loadResources(progress: Progress, prewarm: Bool) throws

    /// Request resources are unloaded / remove from memory if possible
    func unloadResources()

    /// 
    func makeLoadProgress() -> Progress

    ///
    var loadProgressWeights: [Int64] { get }
}
