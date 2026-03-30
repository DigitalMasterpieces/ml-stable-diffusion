// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import CoreML

/// A named compute plan for a single model component.
@available(iOS 17.4, macOS 14.4, *)
public struct ComputePlanEntry: @unchecked Sendable {
    public let name: String
    public let plan: MLComputePlan

    public init(name: String, plan: MLComputePlan) {
        self.name = name
        self.plan = plan
    }
}

// MARK: - ComputePlanProviding

/// A type that can provide compute plans describing how Core ML schedules its operations.
@available(iOS 17.4, macOS 14.4, *)
public protocol ComputePlanProviding {
    var computePlans: [ComputePlanEntry] { get async throws }
}

// MARK: - ManagedModelProviding

/// A type that holds one or more `ManagedMLModel` instances.
///
/// When combined with `ComputePlanProviding`, provides a default `computePlans`
/// implementation that loads an `MLComputePlan` for each managed model.
@available(iOS 17.4, macOS 14.4, *)
public protocol ManagedModelProviding {
    var managedModels: [ManagedMLModel] { get }
}

@available(iOS 17.4, macOS 14.4, *)
extension ComputePlanProviding where Self: ManagedModelProviding {
    public var computePlans: [ComputePlanEntry] {
        get async throws {
            try await withThrowingTaskGroup(of: ComputePlanEntry.self) { group in
                for model in self.managedModels {
                    group.addTask {
                        let name = model.modelURL.deletingPathExtension().lastPathComponent
                        let plan = try await model.computePlan
                        return ComputePlanEntry(name: name, plan: plan)
                    }
                }
                var results: [ComputePlanEntry] = []
                for try await entry in group {
                    results.append(entry)
                }
                return results
            }
        }
    }
}
