// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Digital Masterpieces GmbH. All Rights Reserved.

import Foundation

extension Progress {

    private static var rootKey: UInt8 = 0
    private static var progressChildrenKey: UInt8 = 1

    /// Reference to the root progress (UI-facing)
    var rootProgress: Progress? {
        get {
            objc_getAssociatedObject(self, &Progress.rootKey) as? Progress
        }
        set {
            objc_setAssociatedObject(
                self,
                &Progress.rootKey,
                newValue,
                .OBJC_ASSOCIATION_ASSIGN // avoid retain cycles
            )
        }
    }

    /// Stored children progress objects (in insertion order)
    var children: [Progress] {
        get {
            objc_getAssociatedObject(self, &Progress.progressChildrenKey) as? [Progress] ?? []
        }
        set {
            objc_setAssociatedObject(
                self,
                &Progress.progressChildrenKey,
                newValue,
                .OBJC_ASSOCIATION_RETAIN_NONATOMIC
            )
        }
    }

    /// Adds a child progress and tracks it
    func addTrackedChild(_ child: Progress, units: Int64) {
        addChild(child, withPendingUnitCount: units)
        children.append(child)

        // Propagate root reference
        child.rootProgress = self.rootProgress ?? self
    }
}
