//
//  Generate+Generable.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 28.03.2026.
//

#if canImport(FoundationModels)
    import Foundation
    import FoundationModels
    import AsyncAlgorithms
    import MLXLMCommon

    @available(macOS 26.0, iOS 26.0, *)
    public func generate<Content: Generable>(
        input: LMInput,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters = GenerateParameters(),
        context: ModelContext,
        generating: Content.Type
    ) async throws -> Content {
        let grammar = try Grammar.generable(Content.self)
        let stream = try await generate(
            input: input,
            cache: cache,
            parameters: parameters,
            context: context,
            grammar: grammar
        )

        let output = await stream.compactMap(\.chunk).reduce("", +)
        let generatedContent = try GeneratedContent(json: output)
        let content = try Content(generatedContent)
        return content
    }

    @available(macOS 26.0, iOS 26.0, *)
    public func generate<Content: Generable>(
        input: LMInput,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters = GenerateParameters(),
        context: ModelContext,
        partially: Content.Type,
    ) async throws -> some AsyncSequence<Content.PartiallyGenerated, any Error> {
        let grammar = try Grammar.generable(Content.self)
        let stream = try await generate(
            input: input,
            cache: cache,
            parameters: parameters,
            context: context,
            grammar: grammar
        )

        return stream.compactMap(\.chunk).reductions("", +).map {
            let generatedContent = try GeneratedContent(json: $0)
            let partiallyGenerated = try Content.PartiallyGenerated(generatedContent)
            return partiallyGenerated
        }
    }
#endif
