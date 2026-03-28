//
//  Generate+Generable.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 28.03.2026.
//

import MLXLMCommon

#if canImport(FoundationModels)
    import FoundationModels
#endif

#if compiler(>=6.2)
    @available(macOS 26.0, iOS 26.0, *)
    public func generate<Content: Generable>(
        input: LMInput,
        parameters: GenerateParameters = GenerateParameters(),
        context: ModelContext,
        generating: Content.Type
    ) async throws -> Content {
        let grammar = try Grammar.generable(Content.self)
        let iterator = try await makeGrammarIterator(
            input: input,
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
        let content = try Content(GeneratedContent(json: output))
        return content
    }

    @available(macOS 26.0, iOS 26.0, *)
    public func generatePartially<Content: Generable>(
        input: LMInput,
        parameters: GenerateParameters = GenerateParameters(),
        context: ModelContext,
        generating: Content.Type,
        indent: Int? = nil
    ) async throws -> AsyncStream<Content.PartiallyGenerated> {
        let grammar = try Grammar.generable(Content.self, indent: indent)
        let iterator = try await makeGrammarIterator(
            input: input,
            parameters: parameters,
            context: context,
            grammar: grammar
        )

        let (partialStream, continuation) = AsyncStream<Content.PartiallyGenerated>.makeStream()
        let (stream, task) = generateTask(
            promptTokenCount: input.text.tokens.size,
            modelConfiguration: context.configuration,
            tokenizer: context.tokenizer,
            iterator: iterator
        )

        Task {
            var output = ""
            for await generation in stream {
                if let chunk = generation.chunk {
                    output.append(chunk)
                    let generatedContent = try GeneratedContent(json: output)
                    let partiallyGenerated = try Content.PartiallyGenerated(generatedContent)
                    continuation.yield(partiallyGenerated)
                }
            }

            await task.value
            continuation.finish()
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return partialStream
    }
#endif
