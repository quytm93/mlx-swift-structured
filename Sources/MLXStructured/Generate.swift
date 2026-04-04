//
//  Generate.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 27.09.2025.
//

import Foundation
import JSONSchema
import MLXLMCommon

public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = GenerateParameters(),
    context: ModelContext,
    ebnf: String,
) async throws -> AsyncStream<Generation> {
    let grammar = Grammar.ebnf(ebnf)
    return try await generate(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        grammar: grammar
    )
}

public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = GenerateParameters(),
    context: ModelContext,
    regex: String,
) async throws -> AsyncStream<Generation> {
    let grammar = Grammar.regex(regex)
    return try await generate(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        grammar: grammar
    )
}

public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = .init(),
    context: ModelContext,
    schema: JSONSchema,
    options: JSONSchemaFormatOptions = .init(),
) async throws -> AsyncStream<Generation> {
    let grammar = try Grammar.schema(schema, options: options)
    return try await generate(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        grammar: grammar
    )
}

public func generate<Content: Decodable>(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = .init(),
    context: ModelContext,
    schema: JSONSchema,
    options: JSONSchemaFormatOptions = .init(),
    generating: Content.Type,
    decoder: JSONDecoder = .init()
) async throws -> Content {
    let grammar = try Grammar.schema(schema)
    let stream = try await generate(
        input: input,
        cache: cache,
        parameters: parameters,
        context: context,
        grammar: grammar
    )

    let output = await stream.compactMap(\.chunk).reduce("", +)
    let content = try decoder.decode(Content.self, from: Data(output.utf8))
    return content
}

public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    parameters: GenerateParameters = .init(),
    context: ModelContext,
    grammar: Grammar,
) async throws -> AsyncStream<Generation> {
    let sampler = parameters.sampler()
    let processor = try await GrammarMaskedLogitProcessor.from(
        configuration: context.configuration,
        grammar: grammar
    )

    let iterator = try TokenIterator(
        input: input,
        model: context.model,
        cache: cache,
        processor: processor,
        sampler: sampler,
        prefillStepSize: parameters.prefillStepSize,
        maxTokens: parameters.maxTokens
    )

    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    )

    return stream
}
