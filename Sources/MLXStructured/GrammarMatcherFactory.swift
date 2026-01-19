//
//  GrammarMatcherFactory.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 20.09.2025.
//

import MLXLMCommon
import Hub

public extension GrammarMaskedLogitProcessor {
    static func from(
        hub: HubApi = .shared, // TODO: Request changes in swift-transformers to make the tokenizer vocab (and some other properties) public
        configuration: ModelConfiguration,
        grammar: Grammar
    ) async throws -> GrammarMaskedLogitProcessor {
        let configurations = switch configuration.id {
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
        
        // Collect all vocab entries and find the maximum index
        let vocabDict = tokenizerData.model.vocab.dictionary(or: [:])
        let maxVocabIndex = vocabDict.values.compactMap { $0.integer() }.max() ?? -1
        
        // Check added tokens for even higher indices
        let addedTokens = tokenizerData.addedTokens.array(or: [])
        let maxAddedTokenIndex = addedTokens.compactMap { $0.id.integer() }.max() ?? -1
        
        // Calculate vocab size from tokenizer data
        let calculatedVocabSize = max(maxVocabIndex, maxAddedTokenIndex) + 1
        
        // Try multiple sources for vocab size, in order of preference:
        // 1. Tokenizer config (most reliable for actual model vocab size)
        // 2. Model config
        // 3. Calculated from tokenizer data
        var vocabSize = 0
        
        // First, try tokenizer config vocab_size
        if let tokenizerConfig = tokenizerConfig {
            vocabSize = tokenizerConfig.vocabSize.integer(or: 0)
        }
        
        // If not found, try model config
        if vocabSize == 0, let modelConfig = modelConfig {
            vocabSize = modelConfig.vocabSize.integer(or: 0)
        }
        
        // Use calculated size as final fallback, but always ensure we're at least
        // as large as the calculated size (in case config is outdated)
        vocabSize = max(vocabSize, calculatedVocabSize)
        
        // Safety check: ensure vocab size is reasonable
        if vocabSize == 0 || vocabSize > 1_000_000 {
            vocabSize = calculatedVocabSize
        }
        
        var vocab = Array(repeating: "", count: vocabSize)
        
        for (key, value) in vocabDict {
            if let index = value.integer(), index < vocabSize {
                vocab[index] = key.string
            }
        }
        
        for value in addedTokens {
            if let index = value.id.integer(), let token = value.content.string() {
                // Expand vocab array if needed to accommodate this token
                if index >= vocab.count {
                    vocab.append(contentsOf: Array(repeating: "", count: index - vocab.count + 1))
                }
                if index < vocab.count {
                    vocab[index] = token
                }
            }
        }
        
        let decoders: [Config] = switch tokenizerData.decoder.type.string() {
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
        
//        print("Vocab size:", vocab.count)
//        print("Vocab type:", vocabType)
//        print("Stop tokens Ids:", stopTokenIds)
//        print("Grammar:", grammar)
              
        let grammarMatcher = try XGrammar(vocab: vocab, vocabType: vocabType, stopTokenIds: stopTokenIds, grammar: grammar)
        let processor = GrammarMaskedLogitProcessor(grammarMatcher: grammarMatcher)
        return processor
    }
}
