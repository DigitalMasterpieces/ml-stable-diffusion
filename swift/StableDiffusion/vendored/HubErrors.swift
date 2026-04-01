//
//  HubErrors.swift
//
//  Vendored from https://github.com/huggingface/swift-transformers
//  (Sources/Hub/Hub.swift) — only the error types needed by StableDiffusion.

import Foundation

/// A namespace for Hub-related error types.
public enum Hub {}

public extension Hub {
    /// Errors that can occur during Hub client operations.
    enum HubClientError: LocalizedError {
        /// Failed to parse server response or configuration data.
        case parse
        /// Expected json response could not be parsed as json.
        case jsonSerialization(fileURL: URL, message: String)
        /// Failed to parse data with the specified error message.
        case parseError(String)

        public var errorDescription: String? {
            switch self {
            case .parse:
                String(localized: "Failed to parse server response.")
            case .jsonSerialization(_, let message):
                message
            case let .parseError(message):
                String(localized: "Parse error: \(message)")
            }
        }
    }
}
