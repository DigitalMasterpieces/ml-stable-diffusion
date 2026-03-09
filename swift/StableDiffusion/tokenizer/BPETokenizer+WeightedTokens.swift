// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation

/// Prompt weighting support for BPETokenizer.
/// Parses A1111-style prompt attention syntax and returns per-token weights.
@available(iOS 16.2, macOS 13.1, *)
extension BPETokenizer {

    /// Tokenizes an input string with A1111-style prompt weighting.
    ///
    /// Parses prompt attention syntax like `(text:1.3)`, `((text))`, `[text]`,
    /// tokenizes each segment, and assigns per-token weights.
    ///
    /// - Parameters:
    ///   - input: A string with optional weight syntax.
    ///   - minCount: The minimum number of tokens to return (pads with weight 1.0).
    /// - Returns: Parallel arrays of tokens, token IDs, and per-token weights.
    public func tokenizeWithWeights(
        _ input: String,
        minCount: Int? = nil
    ) -> (tokens: [String], tokenIDs: [Int], weights: [Float]) {
        var tokens: [String] = []
        var weights: [Float] = []

        tokens.append(startToken)
        weights.append(1)

        let normalizedInput = input.trimmingCharacters(in: .whitespacesAndNewlines)
        let textsAndWeights = parsePromptAttention(input: normalizedInput)
        for (text, weight) in textsAndWeights {
            let textTokens = encode(input: text)
            tokens.append(contentsOf: textTokens)
            weights.append(contentsOf: [Float](repeating: weight, count: textTokens.count))
        }

        tokens.append(endToken)
        weights.append(1)

        // Pad if there was a min length specified
        if let minLen = minCount, minLen > tokens.count {
            tokens.append(contentsOf: repeatElement(padToken, count: minLen - tokens.count))
            weights.append(contentsOf: repeatElement(1 as Float, count: minLen - weights.count))
        }

        let ids = tokens.map({ vocabulary[$0, default: unknownTokenID] })
        return (tokens, ids, weights)
    }

    /// Parses A1111-style prompt attention syntax into text segments with weights.
    ///
    /// Supported syntax:
    /// - `(text)` → multiplies weight by 1.1
    /// - `((text))` → multiplies weight by 1.1 × 1.1 = 1.21 (nesting)
    /// - `[text]` → multiplies weight by 1/1.1
    /// - `(text:2.5)` → explicit weight 2.5
    /// - `\(`, `\)`, `\[`, `\]` → escaped literal brackets
    ///
    /// - Parameter input: The raw prompt string with optional weight syntax.
    /// - Returns: Array of (text, weight) tuples with adjacent same-weight segments merged.
    func parsePromptAttention(input: String) -> [(String, Float)] {
        let pattern = #"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|]|[^\\()\[\]:]+|:"#
        let regex = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        let range = NSRange(location: 0, length: input.utf16.count)
        let matches = regex.matches(in: input, options: [], range: range)

        var result: [(String, Float)] = []
        var roundBrackets: [Int] = []
        var squareBrackets: [Int] = []

        let roundBracketMultiplier: Float = 1.1
        let squareBracketMultiplier: Float = 1 / 1.1

        func multiplyRange(start: Int, multiplier: Float) {
            for p in (start..<result.count) {
                result[p].1 *= multiplier
            }
        }

        for match in matches {
            let text = Range(match.range(at: 0), in: input).map { String(input[$0]) } ?? ""
            let weight = Range(match.range(at: 1), in: input).map { String(input[$0]) }

            if text.starts(with: "\\") {
                result.append((String(text.dropFirst()), 1))
            } else if text == "(" {
                roundBrackets.append(result.count)
            } else if text == "[" {
                squareBrackets.append(result.count)
            } else if let weight, let start = roundBrackets.popLast() {
                multiplyRange(start: start, multiplier: Float(weight) ?? 1)
            } else if text == ")", let start = roundBrackets.popLast() {
                multiplyRange(start: start, multiplier: roundBracketMultiplier)
            } else if text == "]", let start = squareBrackets.popLast() {
                multiplyRange(start: start, multiplier: squareBracketMultiplier)
            } else {
                result.append((text, 1))
            }
        }

        // Handle unclosed brackets
        for pos in roundBrackets {
            multiplyRange(start: pos, multiplier: roundBracketMultiplier)
        }
        for pos in squareBrackets {
            multiplyRange(start: pos, multiplier: squareBracketMultiplier)
        }

        if result.isEmpty {
            result.append(("", 1))
        }

        // Merge consecutive segments with equal weights
        var index = 0
        while index + 1 < result.count {
            if result[index].1 == result[index + 1].1 {
                result[index].0 += result[index + 1].0
                result.remove(at: index + 1)
            } else {
                index += 1
            }
        }

        return result
    }
}
