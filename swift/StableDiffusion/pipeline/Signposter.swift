// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import os

/// Shared signposter for Instruments profiling of pipeline operations.
///
/// Uses the `os_signpost` instrument under the `com.apple.StableDiffusion` subsystem.
/// The OS short-circuits signpost calls when Instruments isn't attached, so overhead
/// is negligible in production.
let signposter = OSSignposter(
    subsystem: "com.apple.StableDiffusion",
    category: "Performance"
)
