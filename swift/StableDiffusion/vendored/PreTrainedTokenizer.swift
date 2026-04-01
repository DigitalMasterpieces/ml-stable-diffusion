//
//  PreTrainedTokenizer.swift
//
//  Vendored from https://github.com/huggingface/swift-transformers
//  (Sources/Tokenizers/Tokenizer.swift) — Jinja/chat template and
//  AutoTokenizer/LlamaPreTrainedTokenizer support removed.

import Foundation

/// Errors that can occur during tokenizer operations.
public enum TokenizerError: LocalizedError {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab
    case chatTemplate(String)
    case missingChatTemplate
    case tooLong(String)
    case mismatchedConfig(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig:
            String(localized: "Tokenizer configuration is missing.", comment: "Error when tokenizer config cannot be found")
        case .missingTokenizerClassInConfig:
            String(localized: "The tokenizer class is not specified in the configuration.", comment: "Error when tokenizer_class is missing in config")
        case let .unsupportedTokenizer(name):
            String(localized: "The tokenizer type '\(name)' is not supported.", comment: "Error when tokenizer type is not supported")
        case .missingVocab:
            String(localized: "Vocabulary file is missing from the tokenizer configuration.", comment: "Error when vocab file is missing")
        case .malformedVocab:
            String(localized: "The vocabulary file is malformed or corrupted.", comment: "Error when vocab file is malformed")
        case let .chatTemplate(message):
            String(localized: "Chat template error: \(message)", comment: "Error with chat template")
        case .missingChatTemplate:
            String(localized: "This tokenizer does not have a chat template, and no template was passed.")
        case let .tooLong(message):
            String(localized: "Input is too long: \(message)", comment: "Error when input exceeds maximum length")
        case let .mismatchedConfig(message):
            String(localized: "Tokenizer configuration mismatch: \(message)", comment: "Error when tokenizer configuration is inconsistent")
        }
    }
}

/// A protocol defining the core tokenization functionality.
///
/// This protocol defines the fundamental operations that any tokenization model must support,
/// including converting between text and tokens, and between tokens and their numeric IDs.
public protocol TokenizingModel {
    /// Tokenizes the input text into a sequence of tokens.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func tokenize(text: String) -> [String]

    /// Alias for `tokenize` that allows the instance to be called as a function.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of tokens as strings
    func callAsFunction(_ text: String) -> [String]

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if the token is not in the vocabulary
    func convertTokenToId(_ token: String) -> Int?

    /// Converts multiple token strings to their corresponding numeric IDs.
    ///
    /// - Parameter tokens: An array of token strings to convert
    /// - Returns: An array of numeric IDs, with nil values for tokens not in the vocabulary
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is not valid
    func convertIdToToken(_ id: Int) -> String?

    /// Converts multiple numeric token IDs back to their string representations.
    ///
    /// - Parameter ids: An array of numeric token IDs to convert
    /// - Returns: An array of token strings, with nil values for invalid IDs
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    /// The beginning-of-sequence token string, if defined.
    var bosToken: String? { get }

    /// The numeric ID of the beginning-of-sequence token, if defined.
    var bosTokenId: Int? { get }

    /// The end-of-sequence token string, if defined.
    var eosToken: String? { get }

    /// The numeric ID of the end-of-sequence token, if defined.
    var eosTokenId: Int? { get }

    /// The unknown token string used for out-of-vocabulary words.
    var unknownToken: String? { get }

    /// The numeric ID of the unknown token.
    var unknownTokenId: Int? { get }

    /// Whether consecutive unknown tokens should be fused together.
    var fuseUnknownTokens: Bool { get }
}

/// Helper - possibly to be moved somewhere else
func addedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken else { return nil }
    if let stringValue = addedToken.string() {
        return stringValue
    }
    // This is possibly a serialization of the AddedToken class
    // TODO: support lstrip, rstrip, normalized, etc.
    return addedToken.content.string()
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] {
        tokenize(text: text)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

/// A tokenizer model that can be initialized from Hugging Face Hub configuration data.
///
/// This protocol extends `TokenizingModel` with the ability to be created from configuration
/// files typically found in tokenizer repositories on the Hugging Face Hub.
public protocol PreTrainedTokenizerModel: TokenizingModel {
    /// Initializes a tokenizer model from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration (typically from tokenizer_config.json)
    ///   - tokenizerData: The tokenizer data (typically from tokenizer.json)
    ///   - addedTokens: A dictionary mapping added token strings to their IDs
    /// - Throws: `TokenizerError` if the configuration is invalid or missing required data
    init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws
}

enum TokenizerModel {
    static let knownTokenizers: [String: PreTrainedTokenizerModel.Type] = [
        "BertTokenizer": BertTokenizer.self,
        "CodeGenTokenizer": HFBPETokenizer.self,
        "CodeLlamaTokenizer": HFBPETokenizer.self,
        "CohereTokenizer": HFBPETokenizer.self,
        "DistilbertTokenizer": BertTokenizer.self,
        "DistilBertTokenizer": BertTokenizer.self,
        "FalconTokenizer": HFBPETokenizer.self,
        "GemmaTokenizer": HFBPETokenizer.self,
        "GPT2Tokenizer": HFBPETokenizer.self,
        "LlamaTokenizer": HFBPETokenizer.self,
        "RobertaTokenizer": HFBPETokenizer.self,
        "T5Tokenizer": T5Tokenizer.self,
        "TokenizersBackend": HFBPETokenizer.self,
        "PreTrainedTokenizer": HFBPETokenizer.self,
        "Qwen2Tokenizer": HFBPETokenizer.self,
        "WhisperTokenizer": HFBPETokenizer.self,
        "XLMRobertaTokenizer": UnigramTokenizer.self,
        "Xlm-RobertaTokenizer": UnigramTokenizer.self,
    ]

    static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }

    static func from(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int], strict: Bool = true) throws -> TokenizingModel {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            throw TokenizerError.missingTokenizerClassInConfig
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        // Fallback to BPETokenizer if class is not explicitly registered
        let tokenizerClass = TokenizerModel.knownTokenizers[tokenizerName] ?? HFBPETokenizer.self
        if TokenizerModel.knownTokenizers[tokenizerName] == nil {
            if strict {
                throw TokenizerError.unsupportedTokenizer(tokenizerName)
            } else {
                print("Warning: Tokenizer model class \(tokenizerName) is not registered, falling back to a standard BPE implementation.")
            }
        }
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }
}

/// A complete tokenizer interface supporting encoding, decoding, and special token handling.
public protocol Tokenizer: Sendable {
    func tokenize(text: String) -> [String]
    func encode(text: String) -> [Int]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int]
    func decode(tokens: [Int]) -> String
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String
    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]
    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
}

public extension Tokenizer {
    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, skipSpecialTokens: false)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

let specialTokenAttributes: [String] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
]

/// A comprehensive tokenizer implementation supporting pre-trained models from Hugging Face.
///
/// This class provides a complete tokenizer implementation that can be initialized from
/// Hugging Face Hub configuration files and supports all standard tokenization operations
/// including chat template application, normalization, pre-tokenization, and post-processing.
public class PreTrainedTokenizer: @unchecked Sendable, Tokenizer {
    let model: TokenizingModel

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int? { model.bosTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int? { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }
    public var fuseUnknownTokens: Bool { model.fuseUnknownTokens }

    let addedTokens: Set<String>
    let specialTokens: [String: Int]
    let addedTokensRegex: NSRegularExpression?

    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: TokenDecoder?
    private let tokenizerConfig: Config

    private let cleanUpTokenizationSpaces: Bool

    /// Initializes a tokenizer from Hugging Face configuration files.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: Configuration from tokenizer_config.json
    ///   - tokenizerData: Configuration from tokenizer.json
    ///   - strict: Whether to enforce strict validation of tokenizer types
    /// - Throws: `TokenizerError` if configuration is invalid or tokenizer type is unsupported
    public required init(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws {
        var addedTokens: [String: Int] = [:]
        var specialTokens: [String: Int] = [:]
        for addedToken in tokenizerData["addedTokens"].array(or: []) {
            guard let id = addedToken["id"].integer() else { continue } // malformed: token with no id
            guard let content = addedToken.content.string() else { continue } // malformed: token with no content
            addedTokens[content] = id

            if addedToken["special"].boolean(or: false) {
                specialTokens[content] = id
            }
        }

        // Convert to tuples for easier access, then sort by length (descending) to avoid early partial matches
        // (https://github.com/xenova/transformers.js/commit/c305c3824f628f1f02806a6310bd3b18b0f7f8f5)
        let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = (tokenizerData["addedTokens"].array(or: [])).compactMap { addedToken -> (String, Bool, Bool)? in
            guard let content = addedToken.content.string() else { return nil }
            let prefix = addedToken["lstrip"].boolean(or: false)
            let suffix = addedToken["rstrip"].boolean(or: false)
            return (content: content, prefix: prefix, suffix: suffix)
        }.sorted {
            $0.content.count > $1.content.count
        }

        // then concatenate into regular expression
        let addedTokensRegexString = unwrappedAddedTokens.map {
            let token = NSRegularExpression.escapedPattern(for: $0.content)
            let prefix = $0.prefix ? #"\s*"# : ""
            let suffix = $0.suffix ? #"\s*"# : ""
            return "\(prefix)(\(token))\(suffix)"
        }.joined(separator: "|")
        addedTokensRegex = try? NSRegularExpression(pattern: addedTokensRegexString, options: [])

        self.specialTokens = specialTokens
        self.addedTokens = Set(addedTokens.keys)

        preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        normalizer = NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        decoder = DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: self.addedTokens)
        cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: true)
        self.tokenizerConfig = tokenizerConfig

        model = try TokenizerModel.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens, strict: strict)
    }

    func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer else { return [text] }
        return preTokenizer(text: text, options: options)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer else { return text }
        return normalizer(text: text)
    }

    func postProcess(_ tokens: [String], addSpecialTokens: Bool = true) -> [String] {
        guard let postProcessor else { return tokens }
        return postProcessor(tokens: tokens, addSpecialTokens: addSpecialTokens)
    }

    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

        return
            text
            .replacingOccurrences(of: " .", with: ".")
            .replacingOccurrences(of: " ?", with: "?")
            .replacingOccurrences(of: " !", with: "!")
            .replacingOccurrences(of: " ,", with: ",")
            .replacingOccurrences(of: " ' ", with: "'")
            .replacingOccurrences(of: " n't", with: "n't")
            .replacingOccurrences(of: " 'm", with: "'m")
            .replacingOccurrences(of: " 's", with: "'s")
            .replacingOccurrences(of: " 've", with: "'ve")
            .replacingOccurrences(of: " 're", with: "'re")
    }

    func fuseUnknown(_ tokens: [String]) -> [String] {
        guard fuseUnknownTokens else { return tokens }
        let (fused, _) = tokens.reduce((fused: [String](), previousIsUnknown: false)) { result, token in
            var (fused, previousIsUnknown) = result
            let isUnknown = model.convertTokenToId(token) == model.unknownTokenId
            if isUnknown {
                if !previousIsUnknown { fused.append(token) }
            } else {
                fused.append(token)
            }
            return (fused, isUnknown)
        }
        return fused
    }

    /// Tokenizes input text using the configured normalization and pre-tokenization steps.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of token strings
    public func tokenize(text: String) -> [String] {
        // Take care of special tokens first
        let sections: [String] =
            if let regex = addedTokensRegex {
                text.split(by: regex)
            } else {
                [text]
            }
        return sections.enumerated().map { section, x in
            if addedTokens.contains(x) { return [x] }
            return preTokenize(normalize(x), options: section == 0 ? [.firstSection] : []).flatMap { model($0) }
        }.flatMap { fuseUnknown($0) }
    }

    /// Encodes input text into token IDs with optional special token handling.
    ///
    /// This is the main entry point for text encoding operations.
    ///
    /// - Parameters:
    ///   - text: The input text to encode
    ///   - addSpecialTokens: Whether to add special tokens during post-processing
    /// - Returns: An array of token IDs
    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).map { model.convertTokenToId($0)! }
    }

    /// Encodes input text into token IDs with special tokens included by default.
    ///
    /// - Parameter text: The input text to encode
    /// - Returns: An array of token IDs
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    /// Decodes token IDs back into human-readable text.
    ///
    /// - Parameters:
    ///   - tokens: The token IDs to decode
    ///   - skipSpecialTokens: Whether to exclude special tokens from the output text
    /// - Returns: The decoded text string
    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        // IDs to tokens
        let tokenStrings: [String]
        if skipSpecialTokens {
            let specialTokenIDs = Set(specialTokens.values)
            tokenStrings =
                tokens
                .filter { !specialTokenIDs.contains($0) }
                .compactMap { model.convertIdToToken($0) }
        } else {
            tokenStrings = tokens.compactMap { model.convertIdToToken($0) }
        }
        let decoded = decodeTokens(tokenStrings)
        // At this point we should have a single String
        return cleanUp(text: decoded.joined(separator: ""))
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID of the token, or nil if not found in the vocabulary
    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }

}

// MARK: - Tokenizer model classes

class T5Tokenizer: UnigramTokenizer, @unchecked Sendable {}
