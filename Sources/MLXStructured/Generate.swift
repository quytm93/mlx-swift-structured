//
//  Generate.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 27.09.2025.
//

import Foundation
import JSONSchema
import MLXLMCommon
import MLX

func makeGrammarIterator(
    input: LMInput,
    cache: [KVCache]?,
    parameters: GenerateParameters,
    context: ModelContext,
    grammar: Grammar
) async throws -> TokenIterator {
    let sampler = parameters.sampler()
    let processor = try await GrammarMaskedLogitProcessor.from(configuration: context.configuration, grammar: grammar)
    return try TokenIterator(
        input: input,
        model: context.model,
        cache: cache,
        processor: processor,
        sampler: sampler,
        prefillStepSize: parameters.prefillStepSize,
        maxTokens: parameters.maxTokens
    )
}

public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = GenerateParameters(),
    context: ModelContext,
    grammar: Grammar
) async throws -> AsyncStream<Generation> {
    let iterator = try await makeGrammarIterator(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        grammar: grammar
    )
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    )
    return stream
}

public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = GenerateParameters(),
    context: ModelContext,
    grammar: Grammar
) async throws -> AsyncStream<TokenGeneration> {
    let iterator = try await makeGrammarIterator(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        grammar: grammar
    )
    let (stream, _) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    )
    return stream
}

public func generate<Content: Decodable>(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = GenerateParameters(),
    context: ModelContext,
    schema: JSONSchema,
    generating: Content.Type,
    decoder: JSONDecoder = JSONDecoder()
) async throws -> Content {
    let grammar = try Grammar.schema(schema)
    let iterator = try await makeGrammarIterator(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        grammar: grammar
    )

    let (stream, task) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    )

    var tokens = [Int]()
    for await generation in stream {
        if let token = generation.token {
            tokens.append(token)
        }
    }

    await task.value

    let output = context.tokenizer.decode(tokens: tokens)
    let content = try decoder.decode(Content.self, from: Data(output.utf8))
    return content
}
