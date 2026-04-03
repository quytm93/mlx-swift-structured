//
//  GrammarMatcherFactory.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 20.09.2025.
//

import MLXLMCommon
import Hub

extension GrammarMaskedLogitProcessor {

    static let tokenizerInfoCache = Cache<ModelConfiguration, TokenizerInfo>()

    public static func from(
        hub: HubApi = .shared,  // TODO: Request changes in swift-transformers to make the tokenizer vocab (and some other properties) public
        configuration: ModelConfiguration,
        grammar: Grammar
    ) async throws -> GrammarMaskedLogitProcessor {
        let tokenizerInfo = try await tokenizerInfo(for: configuration, hub: hub)
        let grammarMatcher = try XGrammar(tokenizerInfo: tokenizerInfo, grammar: grammar)
        let processor = GrammarMaskedLogitProcessor(grammarMatcher: grammarMatcher)
        return processor
    }

    private static func tokenizerInfo(
        for configuration: ModelConfiguration,
        hub: HubApi
    ) async throws -> TokenizerInfo {
        if let cached = await tokenizerInfoCache.value(for: configuration) {
            return cached
        }

        let configurations =
            switch configuration.id {
            case .id(let id, let revision):
                LanguageModelConfigurationFromHub(modelName: id, revision: revision, hubApi: hub)
            case .directory(let directory):
                LanguageModelConfigurationFromHub(modelFolder: directory, hubApi: hub)
            }

        let (modelConfig, tokenizerConfig, tokenizerData) = try await (
            configurations.modelConfig,
            configurations.tokenizerConfig,
            configurations.tokenizerData
        )

        let modelVocab: [(token: String, id: Int)] = tokenizerData
            .model.vocab.dictionary(or: [:])
            .compactMap { key, value in
                if let id = value.integer() {
                    return (token: key.string, id: id)
                } else {
                    return nil
                }
            }

        let addedTokens: [(token: String, id: Int)] = tokenizerData
            .addedTokens.array(or: [])
            .compactMap { value in
                if let id = value.id.integer(), let token = value.content.string() {
                    return (token: token, id: id)
                } else {
                    return nil
                }
            }

        let configuredVocabSize =
            [
                modelConfig?.vocabSize.integer(),
                modelConfig?.textConfig.vocabSize.integer(),
                modelConfig?.textConfiguration.vocabSize.integer(),
            ]
            .compactMap { $0 }
            .max() ?? 0

        let derivedVocabSize =
            [
                modelVocab.map(\.id).max(),
                addedTokens.map(\.id).max(),
            ]
            .compactMap { $0 }
            .map { $0 + 1 }
            .max() ?? 0

        let vocabSize = max(configuredVocabSize, derivedVocabSize)
        var vocab = Array(repeating: "", count: vocabSize)
        for (token, id) in (modelVocab + addedTokens) where vocab.indices.contains(id) {
            vocab[id] = token
        }

        let decoders: [Config] =
            switch tokenizerData.decoder.type.string() {
            case "Sequence":
                tokenizerData.decoder.decoders.array(or: [])
            default:
                [tokenizerData.decoder]
            }

        var vocabType: Int32 = 0
        loop: for decoder in decoders {
            switch decoder.type.string() {
            case "ByteFallback":
                vocabType = 1
                break loop
            case "ByteLevel":
                vocabType = 2
                break loop
            default:
                continue
            }
        }

        var stopTokenIds: [Int32] = configuration.extraEOSTokens.compactMap(vocab.firstIndex).map(Int32.init)
        if let tokenizerConfig, let eosToken = tokenizerConfig.eosToken.string(), let eosTokenId = vocab.firstIndex(of: eosToken) {
            stopTokenIds.append(Int32(eosTokenId))
        }

        let tokenizerInfo = TokenizerInfo(vocab: vocab, vocabType: vocabType, stopTokenIds: stopTokenIds)
        await tokenizerInfoCache.set(tokenizerInfo, for: configuration)
        return tokenizerInfo
    }
}

extension ModelConfiguration: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(tokenizerId)
        hasher.combine(overrideTokenizer)
        hasher.combine(defaultPrompt)
        hasher.combine(extraEOSTokens)
        hasher.combine(eosTokenIds)
        hasher.combine(toolCallFormat?.rawValue)
    }
}

extension ModelConfiguration.Identifier: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .id(let id, let revision):
            hasher.combine(0)
            hasher.combine(id)
            hasher.combine(revision)
        case .directory(let directory):
            hasher.combine(1)
            hasher.combine(directory.path)
        }
    }
}
